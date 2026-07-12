"""`MDX23CInference`: load a DrumSep checkpoint (from cache or by downloading
it) and run chunked, overlap-averaged separation over arbitrary-length audio.

`separate()` does the actual work: pads the mix, slices it into overlapping
`chunk_size`-sample windows sized from `config.inference.dim_t`, batches them
through `TFC_TDF_net`, and accumulates overlapping predictions before
dividing by `overlap` to average them back down — this is what lets a model
trained on a few seconds of audio process an arbitrary-length track.
`KNOWN_MODELS` hard-codes the one currently-downloadable aufr33/jarredou
DrumSep checkpoint's org-hosted release URLs and sha256 digests; see
README's Weights provenance section for provenance and for the 5-stem
model's lost-upstream status. `download_model()` verifies sha256 on every
fresh download and re-verifies cached files before reuse. `separate_drums()`
is the file-in/files-out convenience wrapper the CLI and top-level
`separate` alias call.

Reads: .config, .model, .utils.download, .utils.cache
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from tqdm import tqdm

from .config import MDX23CConfig
from .model import TFC_TDF_net
from .utils.download import download_file, sha256sum
from .utils.cache import get_cache_dir


class MDX23CInference:
    """
    MDX23C inference engine for drum audio source separation.

    Handles loading models, processing audio in overlapping chunks, and
    returning separated stems as numpy arrays.

    Supported pretrained models:

    - ``drumsep-6stem``: kick, snare, toms, hh, ride, crash

    The original ``drumsep-5stem`` checkpoint has no surviving original
    source anywhere on the web (see README's Weights provenance section)
    and has been removed from this registry; it may return if a verified
    mirror surfaces. Its architecture config (:meth:`MDX23CConfig.drumsep_5stem`)
    remains available for anyone supplying their own checkpoint file.

    Example::

        engine = MDX23CInference.from_pretrained("drumsep-6stem", device="cuda")
        stems = engine.separate(audio, sample_rate=44100)
        # stems -> {"kick": array, "snare": array, ...}
    """

    # Known model configurations with download URLs. Hosted under org
    # control (openmirlab/mdxnet-infer GitHub Release), not a third-party
    # account — see org constitution art.4 and README's Weights provenance
    # section for the cross-mirror sha256 verification story.
    KNOWN_MODELS = {
        'drumsep-6stem': {
            'config': 'drumsep_6stem',
            'stems': ['kick', 'snare', 'toms', 'hh', 'ride', 'crash'],
            'ckpt_url': (
                'https://github.com/openmirlab/mdxnet-infer/releases/download/'
                'weights-drumsep-v1/'
                'aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'
            ),
            'yaml_url': (
                'https://github.com/openmirlab/mdxnet-infer/releases/download/'
                'weights-drumsep-v1/'
                'aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'
            ),
            'ckpt_sha256': (
                'd2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0'
            ),
            'yaml_sha256': (
                '17d1649a227f841165bdb4c11a42082898192a1ea3ceab7e7e0b9293d6589dd6'
            ),
        },
    }

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[MDX23CConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize MDX23C inference engine.

        Args:
            model_path: Path to .ckpt model weights file.
            config: MDX23CConfig object. If not provided, loaded from
                ``config_path`` or inferred from ``model_name``.
            config_path: Path to YAML config file.
            device: Inference device (``'cuda'``, ``'cpu'``, or ``'mps'``).
                Auto-detects if ``None``.
            model_name: Known model name (currently only ``'drumsep-6stem'``;
                see class docstring). Used to select built-in config when
                ``config`` and ``config_path`` are both ``None``. Raises
                ``ValueError`` if given a name not in :attr:`KNOWN_MODELS`.
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.model_name = model_name

        # Load config
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = MDX23CConfig.from_yaml(Path(config_path))
        elif model_name is not None:
            if model_name not in self.KNOWN_MODELS:
                raise ValueError(
                    f"Unknown model_name: {model_name!r}. "
                    f"Known models: {list(self.KNOWN_MODELS.keys())}. "
                    "Pass an explicit `config` or `config_path` to use a "
                    "custom or no-longer-registered architecture "
                    "(e.g. MDX23CConfig.drumsep_5stem())."
                )
            config_method = getattr(
                MDX23CConfig,
                self.KNOWN_MODELS[model_name]['config']
            )
            self.config = config_method()
        else:
            # Default to 6-stem config
            self.config = MDX23CConfig.drumsep_6stem()

        # Initialize model
        self.model = TFC_TDF_net(self.config, self.device)

        # Load weights
        if model_path is not None:
            self._load_weights(Path(model_path))

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, model_path: Path) -> None:
        """Load model weights from checkpoint file."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

        self.model.load_state_dict(state_dict)

    @property
    def stem_names(self) -> List[str]:
        """List of stem names produced by this model."""
        return list(self.config.training.instruments)

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        batch_size: Optional[int] = None,
        overlap: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio into individual drum stems.

        Args:
            audio: Audio array of shape ``(samples,)`` for mono or
                ``(samples, 2)`` for stereo. Also accepts ``(2, samples)``
                channel-first layout which will be transposed automatically.
            sample_rate: Sample rate of input audio. Resampled to 44100
                if different.
            batch_size: Batch size for inference. Defaults to config value.
            overlap: Number of overlapping segments per chunk. Defaults to
                config value.
            progress: Show tqdm progress bar.

        Returns:
            Dictionary mapping stem names to audio arrays of shape
            ``(samples, 2)``.
        """
        import librosa

        # Ensure stereo (samples, 2)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        elif audio.shape[0] == 2 and audio.shape[1] != 2:
            audio = audio.T

        # Resample if needed
        if sample_rate != self.config.audio.sample_rate:
            audio = librosa.resample(
                audio.T,
                orig_sr=sample_rate,
                target_sr=self.config.audio.sample_rate
            ).T

        # Convert to tensor (channels, samples)
        mix = torch.tensor(audio.T, dtype=torch.float32)

        # Inference parameters
        if batch_size is None:
            batch_size = self.config.inference.batch_size
        if overlap is None:
            overlap = self.config.inference.num_overlap

        # Chunk size calculation
        mdx_segment_size = self.config.inference.dim_t
        chunk_size = self.config.audio.hop_length * (mdx_segment_size - 1)
        hop_size = chunk_size // overlap

        # Pad audio
        mix_shape = mix.shape[1]
        pad_size = hop_size - (mix_shape - chunk_size) % hop_size
        mix = torch.cat([
            torch.zeros(2, chunk_size - hop_size),
            mix,
            torch.zeros(2, pad_size + chunk_size - hop_size)
        ], 1)

        # Split into overlapping chunks
        chunks = mix.unfold(1, chunk_size, hop_size).transpose(0, 1)
        batches = [
            chunks[i:i + batch_size]
            for i in range(0, len(chunks), batch_size)
        ]

        # Initialize output accumulator
        num_stems = self.model.num_target_instruments
        if num_stems > 1:
            accumulated = torch.zeros(num_stems, *mix.shape)
        else:
            accumulated = torch.zeros_like(mix)

        # Process batches
        with torch.no_grad():
            count = 0
            iterator = tqdm(batches, desc="Separating") if progress else batches

            for batch in iterator:
                batch_result = self.model(batch.to(self.device))

                for result in batch_result:
                    result_cpu = result.cpu()
                    accumulated[
                        ...,
                        count * hop_size:count * hop_size + chunk_size
                    ] += result_cpu
                    count += 1

        # Remove padding and normalize by overlap
        output = accumulated[
            ...,
            chunk_size - hop_size:-(pad_size + chunk_size - hop_size)
        ] / overlap

        # Build stem dict
        stems: Dict[str, np.ndarray] = {}
        output_np = output.cpu().detach().numpy()

        for i, stem_name in enumerate(self.config.training.instruments):
            if num_stems > 1:
                stem_audio = output_np[i]  # (channels, samples)
            else:
                stem_audio = output_np

            stems[stem_name] = stem_audio.T  # (samples, channels)

        return stems

    @classmethod
    def download_model(
        cls,
        model_name: str,
        cache_dir: Optional[Union[str, Path]] = None,
        progress: bool = True,
    ) -> tuple[Path, Path]:
        """
        Download a known pretrained model to cache.

        Args:
            model_name: ``'drumsep-6stem'`` (currently the only known model;
                see class docstring).
            cache_dir: Directory to cache model files. Defaults to
                ``~/.cache/mdxnet-infer/`` (override via the
                ``MDXNET_INFER_CACHE_DIR`` env var; see
                :func:`mdxnet_infer.utils.cache.get_cache_dir`).
            progress: Show download progress bar.

        Returns:
            Tuple of ``(ckpt_path, yaml_path)``.

        Raises:
            ValueError: ``model_name`` is unknown, or a downloaded/cached
                file fails sha256 verification.
        """
        if model_name not in cls.KNOWN_MODELS:
            raise ValueError(
                f"Unknown model: {model_name!r}. "
                f"Known models: {list(cls.KNOWN_MODELS.keys())}"
            )

        model_info = cls.KNOWN_MODELS[model_name]

        if cache_dir is None:
            cache_dir = get_cache_dir()
        else:
            cache_dir = Path(cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = cls._fetch_verified(
            model_info['ckpt_url'], cache_dir,
            model_info.get('ckpt_sha256'), progress,
        )
        yaml_path = cls._fetch_verified(
            model_info['yaml_url'], cache_dir,
            model_info.get('yaml_sha256'), progress,
        )

        return ckpt_path, yaml_path

    @staticmethod
    def _fetch_verified(
        url: str,
        cache_dir: Path,
        expected_sha256: Optional[str],
        progress: bool,
    ) -> Path:
        """Download ``url`` into ``cache_dir`` (or reuse the cached copy),
        verifying sha256 against ``expected_sha256`` when given.

        A cached file that fails verification is treated as corrupt and
        re-downloaded once; a fresh download that fails verification raises
        (see :func:`mdxnet_infer.utils.download.download_file`).
        """
        filename = url.split('/')[-1]
        path = cache_dir / filename

        if path.exists():
            if expected_sha256 and sha256sum(path).lower() != expected_sha256.lower():
                print(
                    f"Cached {filename} failed sha256 verification; "
                    "re-downloading..."
                )
                download_file(url, path, progress=progress, expected_sha256=expected_sha256)
            else:
                print(f"Using cached {filename}")
        else:
            print(f"Downloading {filename}...")
            download_file(url, path, progress=progress, expected_sha256=expected_sha256)

        return path

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = 'drumsep-6stem',
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        progress: bool = True,
    ) -> 'MDX23CInference':
        """
        Load a pretrained model by name, downloading weights if needed.

        Args:
            model_name: ``'drumsep-6stem'`` or ``'drumsep-5stem'``.
            cache_dir: Directory for cached model files.
            device: Inference device.
            progress: Show download progress.

        Returns:
            Initialized :class:`MDX23CInference` instance.
        """
        ckpt_path, yaml_path = cls.download_model(
            model_name, cache_dir=cache_dir, progress=progress
        )

        return cls(
            model_path=ckpt_path,
            config_path=yaml_path,
            device=device,
            model_name=model_name,
        )


def separate_drums(
    audio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    model_name: str = 'drumsep-6stem',
    combine_cymbals: bool = False,
    device: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    progress: bool = True,
) -> Dict[str, Path]:
    """
    Separate a drum audio file into individual stem files.

    Downloads the pretrained model automatically on first use.

    Args:
        audio_path: Path to input audio file.
        output_dir: Directory for output WAV files. Defaults to same
            directory as ``audio_path``.
        model_name: ``'drumsep-6stem'`` (default; currently the only known
            pretrained model, see :class:`MDX23CInference`).
        combine_cymbals: If ``True`` and using the 6-stem model, merge
            ride + crash into a single ``cymbals`` stem.
        device: Inference device. Auto-detected if ``None``.
        cache_dir: Directory for cached model weights. Defaults to
            ``~/.cache/mdxnet-infer/`` (override via the
            ``MDXNET_INFER_CACHE_DIR`` env var).
        progress: Show progress messages and bars.

    Returns:
        Dictionary mapping stem names to output file paths.
    """
    import soundfile as sf
    import librosa

    audio_path = Path(audio_path)
    if output_dir is None:
        output_dir = audio_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    if progress:
        print(f"Loading audio: {audio_path.name}")
    audio, sr = librosa.load(str(audio_path), sr=None, mono=False)

    # Handle mono
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)

    # Transpose to (samples, channels)
    audio = audio.T

    # Load model
    if progress:
        print(f"Loading model: {model_name}")
    engine = MDX23CInference.from_pretrained(
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
        progress=progress,
    )

    # Separate
    if progress:
        print("Separating stems...")
    stems = engine.separate(audio, sample_rate=sr, progress=progress)

    # Optionally combine ride+crash into cymbals
    if combine_cymbals and model_name == 'drumsep-6stem':
        from .utils.stems import combine_cymbal_stems
        combine_cymbal_stems(stems)

    # Save outputs
    output_paths: Dict[str, Path] = {}
    base_name = audio_path.stem

    for stem_name, stem_audio in stems.items():
        output_path = output_dir / f"{base_name}_{stem_name}.wav"
        sf.write(str(output_path), stem_audio, sr, subtype='PCM_16')
        output_paths[stem_name] = output_path
        if progress:
            print(f"  Saved: {output_path.name}")

    return output_paths
