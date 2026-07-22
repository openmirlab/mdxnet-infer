"""`MDX23CInference`: load a DrumSep checkpoint (from cache or by downloading
it) and run chunked, overlap-averaged separation over arbitrary-length audio.

`separate()` does the actual work: pads the mix, slices it into overlapping
`chunk_size`-sample windows sized from `config.inference.dim_t`, batches them
through `TFC_TDF_net`, and accumulates overlapping predictions before
dividing by `overlap` to average them back down — this is what lets a model
trained on a few seconds of audio process an arbitrary-length track.
`KNOWN_MODELS` is a compatibility view over the package-owned TOML catalog
for the one currently-downloadable aufr33/jarredou DrumSep checkpoint; see
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
from .checkpoint_catalog import get_checkpoint_metadata, list_model_names


def _known_models() -> dict:
    """Build the legacy URL/digest view from the registry at import time."""
    models = {}
    for name in list_model_names():
        metadata = get_checkpoint_metadata(name)
        if metadata is not None:
            models[name] = {
                "stems": metadata["stems"],
                "api_family": metadata["api_family"],
                "ckpt_url": metadata["checkpoint_url"],
                "yaml_url": metadata["config_url"],
                "ckpt_sha256": metadata["checkpoint_sha256"],
                "yaml_sha256": metadata["config_sha256"],
            }
    return models


def _resolve_device(device: Optional[str]) -> torch.device:
    """Resolve and validate a public inference-device request.

    ``None`` and ``"auto"`` retain the legacy preference order (CUDA, MPS,
    CPU). Explicit requests must name an available ``cpu``, ``cuda``,
    ``cuda:N``, or ``mps`` device; no explicit request is silently downgraded.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cpu":
        return torch.device("cpu")
    if not isinstance(device, str):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is not available")
        return torch.device("mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available")
        return torch.device("cuda")
    if not device.startswith("cuda:"):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    index_text = device[5:]
    if not index_text.isdigit():
        raise ValueError("CUDA device index must be a non-negative integer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available")
    index = int(index_text)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index} is not available")
    return torch.device(device)


class MDX23CInference:
    """
    MDX23C inference engine for source separation.

    Handles loading models, processing audio in overlapping chunks, and
    returning separated stems as numpy arrays.

    Supported pretrained models are the stable names in the package-owned
    checkpoint registry.  They include the DrumSep model and generic
    vocals/instrumental, dereverb, four-stem, and SFX recipes.  The legacy
    ``drumsep-5stem`` architecture remains constructible for local weights,
    but is intentionally not a downloadable registry entry.

    Example::

        engine = MDX23CInference.from_pretrained("drumsep-6stem", device="cuda")
        stems = engine.separate(audio, sample_rate=44100)
        # stems -> {"kick": array, "snare": array, ...}
    """

    # Compatibility view derived from the registry -- never a second source
    # of truth.  Legacy callers still receive the old URL/digest key names.
    KNOWN_MODELS = _known_models()

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
            device: Inference device (``'cpu'``, ``'cuda'``, ``'cuda:N'``,
                or ``'mps'``). Auto-detects if ``None`` or the literal string
                ``'auto'``.
            model_name: Known model name (currently only ``'drumsep-6stem'``;
                see class docstring). Used to select built-in config when
                ``config`` and ``config_path`` are both ``None``. Raises
                ``ValueError`` if given a name not in :attr:`KNOWN_MODELS`.
        """
        self.device = _resolve_device(device)
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
            metadata = get_checkpoint_metadata(model_name)
            assert metadata is not None  # guarded by KNOWN_MODELS above
            recipe = dict(metadata["recipe"])
            recipe["training"] = {
                "instruments": metadata["stems"],
                "target_instrument": metadata["target_instrument"],
            }
            self.config = MDX23CConfig.from_mapping(recipe)
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
        target = self.config.training.target_instrument
        return [target] if target is not None else list(self.config.training.instruments)

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

        stem_names = self.stem_names
        if len(stem_names) != num_stems:
            raise RuntimeError(
                "MDX23C output heads do not match configured stem names; "
                "provide an explicit target-instrument output contract"
            )
        for i, stem_name in enumerate(stem_names):
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
            model_name: Stable name returned by :func:`list_model_names`.
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

    @classmethod
    def is_cached(
        cls,
        model_name: str = 'drumsep-6stem',
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Check whether a known model's checkpoint and config are already on
        disk, without downloading or touching the network.

        Resolves target paths via the same :meth:`_target_path` helper
        :meth:`_fetch_verified` uses, so this can never disagree with
        :meth:`download_model`/:meth:`from_pretrained` about where a
        model's files live.

        Deliberately skips sha256 verification -- this is a cheap,
        conscious tradeoff for a status-only check, not an oversight. A
        corrupt-but-present file reports ``True`` here even though
        :meth:`_fetch_verified` would reject it on hash mismatch and
        re-download.

        Args:
            model_name: Stable name returned by :func:`list_model_names`.
            cache_dir: Directory to check. Defaults to
                ``~/.cache/mdxnet-infer/`` (override via the
                ``MDXNET_INFER_CACHE_DIR`` env var; see
                :func:`mdxnet_infer.utils.cache.get_cache_dir`).

        Returns:
            ``True`` only if both the checkpoint and config files exist.

        Raises:
            ValueError: ``model_name`` is unknown.
        """
        if model_name not in cls.KNOWN_MODELS:
            raise ValueError(
                f"Unknown model: {model_name!r}. "
                f"Known models: {list(cls.KNOWN_MODELS.keys())}"
            )

        model_info = cls.KNOWN_MODELS[model_name]
        cache_dir = Path(cache_dir) if cache_dir else get_cache_dir()

        ckpt_path = cls._target_path(cache_dir, model_info['ckpt_url'])
        yaml_path = cls._target_path(cache_dir, model_info['yaml_url'])

        return ckpt_path.is_file() and yaml_path.is_file()

    @staticmethod
    def _target_path(cache_dir: Path, url: str) -> Path:
        """Compute the on-disk path a download of ``url`` into ``cache_dir``
        resolves to. Pure path arithmetic, no I/O -- shared by
        :meth:`_fetch_verified` (which downloads there) and :meth:`is_cached`
        (which only checks for the file), so the two can never disagree on
        where a checkpoint or config lives.
        """
        from urllib.parse import urlparse

        return cache_dir / Path(urlparse(url).path).name

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
        path = MDX23CInference._target_path(cache_dir, url)
        filename = path.name

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
            model_name: Stable name returned by :func:`list_model_names`.
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


def separate_file(
    audio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    model_name: str = 'drumsep-6stem',
    combine_cymbals: bool = False,
    device: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    progress: bool = True,
) -> Dict[str, Path]:
    """
    Separate an audio file into the registered model's stem files.

    Downloads the pretrained model automatically on first use.

    Args:
        audio_path: Path to input audio file.
        output_dir: Directory for output WAV files. Defaults to same
            directory as ``audio_path``.
        model_name: Stable name returned by :func:`list_model_names`.
        combine_cymbals: DrumSep-only compatibility option. Generic models
            reject it rather than inheriting drum semantics.
        device: Inference device. ``None``/``'auto'`` auto-detect; explicit
            values must be ``'cpu'``, ``'cuda'``, ``'cuda:N'``, or ``'mps'``.
        cache_dir: Directory for cached model weights. Defaults to
            ``~/.cache/mdxnet-infer/`` (override via the
            ``MDXNET_INFER_CACHE_DIR`` env var).
        progress: Show progress messages and bars.

    Returns:
        Dictionary mapping stem names to output file paths.
    """
    metadata = get_checkpoint_metadata(model_name)
    if metadata is None:
        raise ValueError(f"Unknown model: {model_name!r}. Known models: {list_model_names()}")
    if combine_cymbals and metadata["api_family"] != "drumsep":
        raise ValueError("combine_cymbals is only valid for drumsep-6stem")

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


def separate_drums(
    audio_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    model_name: str = "drumsep-6stem",
    combine_cymbals: bool = False,
    device: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    progress: bool = True,
) -> Dict[str, Path]:
    """DrumSep-only file convenience wrapper.

    Generic MDX23C models must use :func:`separate_file`, making it impossible
    for their stems to silently acquire DrumSep-only cymbal semantics.
    """
    metadata = get_checkpoint_metadata(model_name)
    if metadata is None or metadata["api_family"] != "drumsep":
        raise ValueError(
            f"{model_name!r} is not a DrumSep model; use separate_file() instead"
        )
    return separate_file(
        audio_path,
        output_dir=output_dir,
        model_name=model_name,
        combine_cymbals=combine_cymbals,
        device=device,
        cache_dir=cache_dir,
        progress=progress,
    )
