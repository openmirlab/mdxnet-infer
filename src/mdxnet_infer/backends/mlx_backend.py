"""MLX backend -- native Apple Silicon execution behind the same seam.

Builds the vendored MLX `TfcTdfV3MLX` from the package's own config and its
own sha256-verified PyTorch checkpoint, so the package-owned checkpoint
contract is unchanged: no second catalog, no converted-weight cache the
loader does not control. Every registry checkpoint in this package shares
one architecture (unlike the sibling `bs-roformer-infer` package, which has
per-checkpoint mask-estimator variants) -- see `mdxnet_infer.backends`
module docstring -- so this backend supports every checkpoint the Torch path
does; there is no `supports_variation` refusal to write here.

`separate()` mirrors `MDX23CInference.separate()`'s exact arithmetic --
`chunk_size = hop_length * (dim_t - 1)`, `hop_size = chunk_size // overlap`,
constant zero-padding at both ends sized from `hop_size` and the mixture's
own length, batched forward passes, additive accumulation, divide by
`overlap` at the end. This is deliberately **not** the sibling
package's fade-windowed overlap-add: `MDX23CInference.separate()` never
built or needed a fade window, because with `chunk_size == hop_size *
overlap` every interior output sample is covered by exactly `overlap`
chunks, so uniform division is already exact. Porting the sibling's
windowed accumulator here would silently change the math this package's
Torch path has always computed.

The vendored model's `exact_zero_safe_rfft()` guard (see `mdxnet_infer.mlx.
model` module docstring) is measured **inert** for this architecture --
kept for consistency and cheap insurance, not because removing it moves
parity outside noise here.

Reads: .base (ChunkingPlan, BackendUnavailable), ..mlx (TfcTdfV3MLX,
conversion), ..utils.checkpoint (load_checkpoint_state), numpy
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .base import BackendUnavailable, ChunkingPlan


def _config_to_mapping(config) -> dict:
    """`MDX23CConfig` (dataclass tree) -> the plain dict `TfcTdfV3MLX` expects.

    `TfcTdfV3MLX.__init__` reads `config["model"]`, `config["audio"]`, and
    `config["training"]` exactly as `TFC_TDF_net.__init__` reads
    `config.model`, `config.audio`, `config.training` -- this is a format
    conversion, not a second source of architecture truth.
    """
    return {
        "audio": dataclasses.asdict(config.audio),
        "model": dataclasses.asdict(config.model),
        "training": dataclasses.asdict(config.training),
        "inference": dataclasses.asdict(config.inference),
    }


class MLXBackend:
    """Runs TFC_TDF_v3 (MDX23C) natively on Apple Silicon through MLX."""

    name = "mlx"

    def __init__(self, model, config, device="mps"):
        self._model = model
        self._config = config
        self._device = device

    # ------------------------------------------------------------- availability

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mlx.core  # noqa: F401
            import mlx_spectro  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def _require(cls) -> None:
        if not cls.is_available():
            raise BackendUnavailable(
                "the MLX backend needs the optional extra: "
                "pip install 'mdxnet-infer[mlx]' (Apple Silicon)"
            )

    # ------------------------------------------------------------- construction

    @staticmethod
    def _select_device(device):
        """MLX owns its own execution target; a Torch device string is refused.

        Silently ignoring `device="cuda"` here -- which this backend used to do,
        and which its CLI help still advertised -- discards what the caller
        explicitly asked for. They would only discover it by noticing the wrong
        hardware was busy, the same failure article 4b forbids for devices.
        """
        if device in (None, "auto", "mps"):
            return "mps"
        raise BackendUnavailable(
            f"backend 'mlx' cannot honour device {device!r}; it executes on Apple "
            f"Silicon and accepts None, 'auto', or 'mps'. Use backend='torch' to "
            f"select a Torch device."
        )

    @classmethod
    def from_checkpoint(cls, *, config, checkpoint_path, device="mps"):
        """Build an MLX model from this package's own checkpoint and config."""
        cls._require()
        device = cls._select_device(device)

        from ..mlx import TfcTdfV3MLX, convert_torch_to_mlx_weights, load_converted_weights
        from ..utils.checkpoint import load_checkpoint_state

        state = load_checkpoint_state(checkpoint_path)
        model = TfcTdfV3MLX(config=_config_to_mapping(config))
        load_converted_weights(model, convert_torch_to_mlx_weights(state))
        model.eval()
        return cls(model, config, device=device)

    # ----------------------------------------------------------------- protocol

    @property
    def resolved_device(self) -> str:
        return self._device

    @property
    def model(self):
        """The resident MLX model, for callers composing the advanced path directly."""
        return self._model

    def release(self) -> None:
        self._model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass

    def separate(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 44100,
        batch_size: Optional[int] = None,
        overlap: Optional[int] = None,
        progress: bool = True,
    ) -> dict:
        """Chunked overlap-average, mirroring MDX23CInference.separate()'s
        exact arithmetic (constant zero-pad, additive accumulation, divide
        by `overlap` -- no fade window; see module docstring)."""
        import mlx.core as mx

        config = self._config

        # Shape/resample normalization identical to MDX23CInference.separate().
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        elif audio.shape[0] == 2 and audio.shape[1] != 2:
            audio = audio.T
        if sample_rate != config.audio.sample_rate:
            import librosa

            audio = librosa.resample(
                audio.T, orig_sr=sample_rate, target_sr=config.audio.sample_rate
            ).T

        mix = np.ascontiguousarray(audio.T, dtype=np.float32)  # (channels, samples)

        if batch_size is None:
            batch_size = config.inference.batch_size
        if overlap is None:
            overlap = config.inference.num_overlap

        plan = ChunkingPlan.from_config(config, overlap=overlap)
        chunk_size, hop_size = plan.chunk_size, plan.hop_size

        mix_shape = mix.shape[1]
        pad_size = hop_size - (mix_shape - chunk_size) % hop_size
        mix = np.concatenate(
            [
                np.zeros((2, chunk_size - hop_size), dtype=np.float32),
                mix,
                np.zeros((2, pad_size + chunk_size - hop_size), dtype=np.float32),
            ],
            axis=1,
        )
        mix = mx.array(mix)

        total_length = mix.shape[1]
        num_chunks = (total_length - chunk_size) // hop_size + 1
        positions = [i * hop_size for i in range(num_chunks)]

        num_stems = self._model.num_target_instruments
        if num_stems > 1:
            accumulated = mx.zeros((num_stems, 2, total_length), dtype=mx.float32)
        else:
            accumulated = mx.zeros((2, total_length), dtype=mx.float32)

        iterator = range(0, len(positions), batch_size)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(list(iterator), desc="Separating")

        for batch_start in iterator:
            batch_positions = positions[batch_start:batch_start + batch_size]
            batch_chunks = mx.stack(
                [mix[:, p:p + chunk_size] for p in batch_positions], axis=0
            )
            batch_result = self._model(batch_chunks)
            mx.eval(batch_result)

            for j, p in enumerate(batch_positions):
                result = batch_result[j]
                span = slice(p, p + chunk_size)
                if num_stems > 1:
                    accumulated[:, :, span] = accumulated[:, :, span] + result
                else:
                    accumulated[:, span] = accumulated[:, span] + result

        end = pad_size + chunk_size - hop_size
        output = accumulated[..., chunk_size - hop_size:total_length - end] / overlap
        output_np = np.array(output, dtype=np.float32)

        stem_names = [self._config.training.target_instrument] if self._config.training.target_instrument \
            else list(self._config.training.instruments)
        if len(stem_names) != num_stems:
            raise RuntimeError(
                "MDX23C output heads do not match configured stem names; "
                "provide an explicit target-instrument output contract"
            )

        stems = {}
        for i, stem_name in enumerate(stem_names):
            stem_audio = output_np[i] if num_stems > 1 else output_np
            stems[stem_name] = stem_audio.T  # (samples, channels)
        return stems
