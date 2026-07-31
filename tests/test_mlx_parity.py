"""Torch-vs-MLX output parity on the real drumsep-6stem checkpoint, including
silence, exercised through the public `.separate()` API on a real audio file
on disk (not a bare module forward pass).

Marked ``realweights`` and deselected by default: needs the MLX extra, an
Apple Silicon Mac, and the default checkpoint already on disk (this test
never downloads it itself if missing -- it skips).

Run explicitly:  pytest -m realweights tests/test_mlx_parity.py -v

The silence cases are the point of this file, not signal alone: every track's
final chunk is zero-padded by `MDX23CInference.separate()`'s own chunking
arithmetic, and a fixture without a genuinely silent/near-silent region would
not exercise that path at all. See `mdxnet_infer/mlx/model.py::
exact_zero_safe_rfft` for the (measured-inert-here) mitigation that guards
against a much larger divergence in the sibling `bs-roformer-infer` package.

Recorded measurement (2026-07-30, Apple M-series, torch 2.13.0, mlx 0.31.2),
worst-case max-abs Torch-vs-MLX divergence:

    tail          with guard    without guard (fix removed, then restored)
    signal        1.371e-06     1.654e-06
    zeros         1.445e-06     1.952e-06
    near_silent   1.028e-06     1.490e-06

All six sit in the same ~1e-6 noise floor -- unlike the sibling package
(4.0e-07 clean vs 1.455e-02 zero-padded), removing the guard here does not
move any silent-tail case outside the noise its clean-signal case already
has. The guard is measured inert for this architecture; see
`mdxnet_infer/mlx/model.py`'s module docstring for why.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.realweights

MODEL = "drumsep-6stem"
SEED = 1
# Measured worst case (see module docstring) is ~2e-06; the gate is set from
# what the implementation actually achieves, with headroom for run-to-run
# noise, not widened to make an unrelated future regression pass quietly.
MAX_ABS_TOLERANCE = 1e-5


def _mlx_available():
    try:
        import mlx.core  # noqa: F401
        import mlx_spectro  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def loaded():
    """(config, torch engine, mlx backend) built from the real, already-cached
    default checkpoint. Skips (never fails) when a prerequisite is missing."""
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'mdxnet-infer[mlx]'")

    from mdxnet_infer.backends.mlx_backend import MLXBackend
    from mdxnet_infer.config import MDX23CConfig
    from mdxnet_infer.inference import MDX23CInference

    try:
        ckpt_path, yaml_path = MDX23CInference.download_model(MODEL, progress=False)
    except Exception as exc:  # pragma: no cover - network/env dependent
        pytest.skip(f"{MODEL} checkpoint unavailable: {exc}")

    config = MDX23CConfig.from_yaml(Path(yaml_path))
    torch_engine = MDX23CInference(
        model_path=ckpt_path, config=config, device="cpu", model_name=MODEL
    )
    mlx_backend = MLXBackend.from_checkpoint(config=config, checkpoint_path=ckpt_path)
    return config, torch_engine, mlx_backend


def _fixture_audio(config, tail: str) -> np.ndarray:
    """One real chunk's worth of stereo audio: real signal up front, `tail`
    behaviour after it -- (samples, channels), matching what `librosa.load`
    hands `MDX23CInference.separate()`."""
    from mdxnet_infer.backends.base import ChunkingPlan

    plan = ChunkingPlan.from_config(config)
    signal_samples = plan.chunk_size // 2

    rng = np.random.default_rng(SEED)
    chunk = (rng.standard_normal((plan.chunk_size, 2)) * 0.1).astype(np.float32)
    if tail == "zeros":
        chunk[signal_samples:, :] = 0.0
    elif tail == "near_silent":
        chunk[signal_samples:, :] *= 1e-6
    return chunk


@pytest.mark.parametrize("tail", ["signal", "zeros", "near_silent"])
def test_mlx_matches_torch_including_silence(tmp_path, loaded, tail):
    config, torch_engine, mlx_backend = loaded
    audio = _fixture_audio(config, tail)

    # A real file on disk, read back exactly as `separate_file()` would.
    wav_path = tmp_path / f"{tail}.wav"
    sf.write(str(wav_path), audio, config.audio.sample_rate, subtype="FLOAT")
    import librosa

    on_disk, sr = librosa.load(str(wav_path), sr=None, mono=False)
    on_disk = on_disk.T  # (samples, channels), matching separate_file()

    torch_stems = torch_engine.separate(on_disk, sample_rate=sr, progress=False)
    mlx_stems = mlx_backend.separate(on_disk, sample_rate=sr, progress=False)

    assert set(torch_stems) == set(mlx_stems)
    worst = 0.0
    for stem, reference in torch_stems.items():
        diff = float(np.abs(reference - mlx_stems[stem]).max())
        worst = max(worst, diff)
    print(f"\n[mlx parity] tail={tail} worst_max_abs={worst:.3e}")
    assert worst < MAX_ABS_TOLERANCE, (
        f"tail={tail}: Torch-vs-MLX max abs {worst:.3e} exceeds "
        f"{MAX_ABS_TOLERANCE:.0e}. If this fires only for a silent tail, "
        f"suspect exact_zero_safe_rfft in mdxnet_infer/mlx/model.py -- "
        f"investigate rather than widen the tolerance"
    )
