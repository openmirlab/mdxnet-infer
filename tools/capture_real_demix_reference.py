"""Capture real DrumSep stems from pinned, unmodified MSST source.

Requires a read-only ZFTurbo/Music-Source-Separation-Training checkout,
`ml-collections` in the capture environment, and the official SHA-verified
DrumSep checkpoint/config. This script never imports `mdxnet_infer`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

SOURCE_REVISION = "84b1eac0887756b4f1a9d7a1ff49105939749ed2"
CHECKPOINT_SHA256 = "d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0"
CONFIG_SHA256 = "17d1649a227f841165bdb4c11a42082898192a1ea3ceab7e7e0b9293d6589dd6"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "real_demix"
SAMPLE_RATE = 44100
SAMPLES = 130560
STEMS = ("kick", "snare", "toms", "hh", "ride", "crash")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_audio(path: Path) -> None:
    """Generate a copyright-free stereo 120-BPM drum/bass mixture."""
    signal = np.zeros(SAMPLES, dtype=np.float64)
    rng = np.random.default_rng(20261003)
    for beat in range(6):
        start = int(beat * 0.5 * SAMPLE_RATE)
        size = min(int(0.18 * SAMPLE_RATE), SAMPLES - start)
        time = np.arange(size) / SAMPLE_RATE
        frequency = 72 - 30 * np.minimum(time / 0.18, 1)
        signal[start : start + size] += 0.55 * np.sin(2 * np.pi * frequency * time) * np.exp(-25 * time)
        if beat % 2:
            signal[start : start + size] += (
                0.19 * rng.standard_normal(size) + 0.12 * np.sin(2 * np.pi * 180 * time)
            ) * np.exp(-24 * time)
    for step in range(12):
        start = int(step * 0.25 * SAMPLE_RATE)
        if start >= SAMPLES:
            break
        size = min(int(0.07 * SAMPLE_RATE), SAMPLES - start)
        time = np.arange(size) / SAMPLE_RATE
        signal[start : start + size] += 0.048 * rng.standard_normal(size) * np.exp(-75 * time)
    for frequency, start_seconds in ((110, 0), (146.83, 1), (130.81, 2)):
        start = int(start_seconds * SAMPLE_RATE)
        size = min(SAMPLE_RATE, SAMPLES - start)
        time = np.arange(size) / SAMPLE_RATE
        envelope = (1 - np.exp(-45 * time)) * np.exp(-1.7 * time)
        signal[start : start + size] += 0.06 * np.sin(2 * np.pi * frequency * time) * envelope
    stereo = np.stack((signal, 0.94 * signal + 0.018 * np.roll(signal, 37)), axis=1)
    sf.write(path, np.clip(stereo, -0.95, 0.95), SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        parser.error(f"upstream must be {SOURCE_REVISION}, got {revision}")
    for path, expected in ((args.checkpoint, CHECKPOINT_SHA256), (args.config, CONFIG_SHA256)):
        if sha256(path) != expected:
            parser.error(f"SHA-256 mismatch: {path}")

    sys.path.insert(0, str(args.upstream))
    from ml_collections import ConfigDict
    from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
    from utils.model_utils import demix

    FIXTURES.mkdir(parents=True, exist_ok=True)
    audio_path = FIXTURES / "drum_mix.wav"
    make_audio(audio_path)
    signal, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    assert sample_rate == SAMPLE_RATE and signal.shape == (SAMPLES, 2)
    with args.config.open() as stream:
        config = ConfigDict(yaml.load(stream, Loader=yaml.FullLoader))
    assert list(config.training.instruments) == list(STEMS)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model = TFC_TDF_net(config).eval()
    model.load_state_dict(state)
    output = demix(config, model, signal.T, torch.device("cpu"), "mdx23c", pbar=False)
    assert list(output) == list(STEMS)
    np.savez_compressed(FIXTURES / "upstream_stems.npz", **output)
    cpu_model = next(
        (line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
         if line.startswith("model name")), platform.processor()
    )
    metadata = {
        "upstream_revision": revision,
        "upstream_model_blob": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD:models/mdx23c_tfc_tdf_v3.py"],
            text=True,
        ).strip(),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "audio_sha256": sha256(audio_path),
        "sample_rate": SAMPLE_RATE,
        "samples": SAMPLES,
        "stems": list(STEMS),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "cpu_model": cpu_model,
        "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "original_use_amp": bool(config.training.use_amp),
        "output": {
            stem: {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            for stem, array in output.items()
        },
    }
    (FIXTURES / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"audio_sha256": metadata["audio_sha256"], "output": metadata["output"]}, indent=2))


if __name__ == "__main__":
    main()
