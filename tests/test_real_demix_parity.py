"""End-to-end parity with a pre-edit, pristine-upstream DrumSep capture."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from mdxnet_infer.inference import MDX23CInference


FIXTURE = Path(__file__).parent / "fixtures" / "real_demix"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.realweights
def test_drumsep_public_separate_matches_pristine_upstream():
    """All six final stems must match upstream demix, not just model forward."""
    checkpoint_text = os.getenv("MDXNET_REAL_CHECKPOINT")
    config_text = os.getenv("MDXNET_REAL_CONFIG")
    if not checkpoint_text or not config_text:
        pytest.skip("set MDXNET_REAL_CHECKPOINT and MDXNET_REAL_CONFIG")
    checkpoint, config = Path(checkpoint_text), Path(config_text)
    metadata = json.loads((FIXTURE / "metadata.json").read_text())
    assert _sha256(checkpoint) == metadata["checkpoint_sha256"]
    assert _sha256(config) == metadata["config_sha256"]
    assert _sha256(FIXTURE / "drum_mix.wav") == metadata["audio_sha256"]

    audio, sample_rate = sf.read(FIXTURE / "drum_mix.wav", dtype="float32", always_2d=True)
    assert sample_rate == metadata["sample_rate"]
    torch.set_num_threads(1)
    engine = MDX23CInference(model_path=checkpoint, config_path=config, device="cpu")
    actual = engine.separate(audio, sample_rate=sample_rate, progress=False)

    with np.load(FIXTURE / "upstream_stems.npz") as expected:
        assert list(actual) == metadata["stems"]
        for stem in metadata["stems"]:
            original = expected[stem].T
            assert actual[stem].shape == original.shape
            difference = np.abs(actual[stem] - original)
            assert float(difference.max()) < 1e-4, (
                f"{stem}: maximum absolute error {difference.max():.6g}"
            )
            assert float(np.sqrt(np.mean(difference**2))) < 1e-5, (
                f"{stem}: RMS error {np.sqrt(np.mean(difference**2)):.6g}"
            )
