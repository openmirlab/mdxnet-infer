# mdxnet-infer

**Inference-only MDX23C TFC-TDF drum source separation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

---

## Why This Exists

mdxnet-infer separates drum audio into individual component stems using the MDX23C TFC-TDF architecture. The framework this architecture comes from — [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) — is a large, general-purpose repo covering dozens of architectures and full training pipelines; mdxnet-infer reprovides just the inference-only subset needed to run one community-trained checkpoint, as a small, pip/uv-installable package with no training code and no unrelated architectures.

The checkpoint itself has its own, separate problem: it's the community-trained **DrumSep** model (6-stem) by aufr33 and jarredou, and its original release host, `github.com/jarredou/models`, is gone — the account/repository no longer exists and the release assets are unreachable. openmirlab located byte-identical third-party mirrors, verified them, and re-published the checkpoint under org control (GitHub Release, sha256-checked) so this package no longer depends on a single third-party account staying online. See Acknowledgments and [What This Project Will NEVER Bundle](#what-this-project-will-never-bundle) below for the full provenance story.

## Acknowledgments

mdxnet-infer is built on the MDX23C TFC-TDF architecture and the DrumSep model weights by aufr33 and jarredou.

- **TFC-TDF v3 architecture** originates from [kuielab/sdx23](https://github.com/kuielab/sdx23/) (KUIELab's entry to the Sound Demixing Challenge 2023, MIT licensed).
- **[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)** by Roman Solovyev (ZFTurbo) — the training/inference framework this package's `model.py` is a verbatim-ported, inference-only subset of (`models/mdx23c_tfc_tdf_v3.py`, MIT licensed). See Citation below for the associated benchmark paper.
- **DrumSep model weights** by aufr33 and jarredou — see [What This Project Will NEVER Bundle](#what-this-project-will-never-bundle) below; these are separate community-trained checkpoints, not part of the ZFTurbo/KUIELab code release, and their license terms are not formally documented by the original authors.

## Citation

```bibtex
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks},
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

(Citation as used by the upstream [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) repository this architecture is ported from. Verified against [arXiv:2305.07489](https://arxiv.org/abs/2305.07489) — title, authors, and eprint match exactly.)

## Features

- 6-stem separation: kick, snare, toms, hi-hat, ride, crash (with optional cymbal merging)
- High-level `separate()` function and `MDX23CInference` class API
- Time-Frequency Convolution with Time-Distributed Fully-connected (TFC-TDF) blocks
- 44.1 kHz stereo output

## Scope

**In scope:**
- Inference-only 6-stem DrumSep separation via CLI and Python API
- The TFC-TDF v3 architecture, ported verbatim for inference (no training loop)
- Automatic and manual weight management with sha256 verification

**Out of scope, forever:**
- Training code — use the upstream [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) repo for that
- Architectures/checkpoints other than MDX23C TFC-TDF DrumSep
- A GUI

**Currently unavailable (not "forever" — revisit if a mirror surfaces):**
- `drumsep-5stem` auto-download — see [Supported Models](#supported-models)

## Install

### With UV (Recommended)

```bash
uv add mdxnet-infer
```

### With pip

```bash
pip install mdxnet-infer
```

## Quick Start

```python
from mdxnet_infer import separate

# Separate a drum file (downloads model automatically)
output_paths = separate("drums.wav", output_dir="output/", model_name="drumsep-6stem")
```

## Usage

### CLI

```bash
# 6-stem separation (kick, snare, toms, hh, ride, crash)
mdxnet-infer drums.wav -o output/ --model drumsep-6stem

# Combine ride+crash into a single cymbals stem
mdxnet-infer drums.wav -o output/ --model drumsep-6stem --combine-cymbals

# Run on a specific device
mdxnet-infer drums.wav -o output/ --device cuda

# Use a custom weights folder (see "What This Project Will NEVER Bundle" below)
mdxnet-infer drums.wav -o output/ --cache-dir /path/to/weights
```

### Python API

```python
from mdxnet_infer import separate, MDX23CInference

# High-level convenience function
output_paths = separate("drums.wav", output_dir="output/", model_name="drumsep-6stem")

# Lower-level API with more control
engine = MDX23CInference.from_pretrained("drumsep-6stem", device="cuda")
import librosa
audio, sr = librosa.load("drums.wav", sr=None, mono=False)
stems = engine.separate(audio.T, sample_rate=sr)
# stems -> {"kick": array, "snare": array, "toms": array, "hh": array, "ride": array, "crash": array}
```

## Supported Models

| Model | Stems | Description |
|-------|-------|-------------|
| `drumsep-6stem` | kick, snare, toms, hh, ride, crash | aufr33-jarredou MDX23C DrumSep v0.1 |
| `drumsep-5stem` | kick, snare, toms, hh, cymbals | **Unavailable** — lost upstream, see [What This Project Will NEVER Bundle](#what-this-project-will-never-bundle) |

## What This Project Will NEVER Bundle

mdxnet-infer will **never** ship the DrumSep checkpoint files inside the pip/wheel package itself. Weights are downloaded on first use (or placed manually for offline use) into a local cache directory, and every file — fresh download or cached — is checked against recorded sha256 digests before it's loaded; a mismatch raises `ChecksumMismatchError` and deletes the bad file rather than silently loading it.

- **6-stem model** (`drumsep-6stem`): `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt`, originally from the "aufr33-jarredou MDX23C DrumSep v0.1" release on `github.com/jarredou/models`.
- **Original hosting is gone; re-hosted under org control.** As of the 2026-07 campaign, the `github.com/jarredou/models` account/repository no longer exists — the original release assets are unreachable. openmirlab located the checkpoint and its config on two independent third-party Hugging Face mirrors, confirmed the two mirrors were **byte-identical** (matching sha256 digests), and re-published both files as an openmirlab-controlled GitHub Release so `mdxnet-infer` no longer depends on a third-party account staying online (org constitution art.4: weights hosting must be under org control or a verified mirror, never a single third-party account).
- **Download URLs and checksums** (auto-downloaded by the package; see [Manual download](#manual-download-and-offline-use) for the manual recipe):

  | File | URL | sha256 |
  |------|-----|--------|
  | `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt` | `https://github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt` | `d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0` |
  | `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml` | `https://github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml` | `17d1649a227f841165bdb4c11a42082898192a1ea3ceab7e7e0b9293d6589dd6` |

- **5-stem model (`drumsep-5stem`) is lost upstream — no longer available.** It was also hosted only on `github.com/jarredou/models` (the "DrumSep" release, `drumsep_5stems_mdx23c_jarredou.ckpt`), which is gone. Unlike the 6-stem model, no intact original-quality mirror could be found anywhere on the web during this campaign — the only surviving copy is a derived OpenVINO conversion (`Intel/drumsep_mdx23c_jarredou_openvino`), which is not usable as a drop-in PyTorch checkpoint. `drumsep-5stem` has therefore been **removed from `KNOWN_MODELS`** (no auto-download path); the architecture config (`MDX23CConfig.drumsep_5stem()`) remains available for anyone who already has their own copy of the checkpoint. It may return to `KNOWN_MODELS` if a verified original-format mirror surfaces.
- **License is not formally documented by aufr33/jarredou.** No LICENSE file or explicit terms accompanied the original GitHub Release or the linked Colab notebooks. Third-party re-uploads disagree: a Hugging Face mirror of the same checkpoint (`Politrees/UVR_resources`) is tagged `license:mit` (uploader-asserted, not from the original authors), while the derived OpenVINO conversion (`Intel/drumsep_mdx23c_jarredou_openvino`) is tagged `license:cc-by-nc-sa-4.0`. Until the original authors confirm terms, treat these weights as **non-commercial-safe only** — do not assume MIT for the weights even though the code around them is MIT. openmirlab is a non-commercial project, so this is usable here, but downstream users with commercial intent should not assume the same.

### Weights folder: default location and override

Downloaded weight files are cached in a directory resolved in this order:

1. The `cache_dir` argument to `separate()` / `MDX23CInference.download_model()` / `MDX23CInference.from_pretrained()`, or the CLI's `--cache-dir` flag.
2. The `MDXNET_INFER_CACHE_DIR` environment variable.
3. The default: `~/.cache/mdxnet-infer/`.

```bash
# Override via env var
export MDXNET_INFER_CACHE_DIR=/path/to/weights
mdxnet-infer drums.wav -o output/

# Override via CLI flag
mdxnet-infer drums.wav -o output/ --cache-dir /path/to/weights
```

```python
# Override via API argument
from mdxnet_infer import MDX23CInference
engine = MDX23CInference.from_pretrained("drumsep-6stem", cache_dir="/path/to/weights")
```

### Manual download and offline use

For offline/air-gapped machines, download the two files from the table above yourself (e.g. with `curl` or a browser) and place them directly in the weights folder (default `~/.cache/mdxnet-infer/`, or your `MDXNET_INFER_CACHE_DIR`/`--cache-dir` override) — no subdirectory needed, filenames must match exactly:

```bash
mkdir -p ~/.cache/mdxnet-infer
curl -L -o ~/.cache/mdxnet-infer/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt \
  https://github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt
curl -L -o ~/.cache/mdxnet-infer/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml \
  https://github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml
```

Verify the sha256 digests match the table above before use:

```bash
sha256sum ~/.cache/mdxnet-infer/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.*
```

Once both files are in place, `mdxnet-infer`/`separate()`/`from_pretrained()` will find and use them without attempting a download.

## Development

```bash
# Set up environment
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run the test suite (40 tests, offline: synthetic tensors / mocked HTTP, no real checkpoint needed)
pytest tests/ -v

# Lint
ruff check .

# Packaging check
python -m build
```

CI (`.github/workflows/test.yml`) runs the same test suite on Python 3.10 and 3.12 on every push/PR; `publish.yml` gates PyPI publishing on that suite passing.

## License

- **Code**: MIT License. See [LICENSE](LICENSE) for details. `src/mdxnet_infer/model.py` is a verbatim-ported, inference-only subset of ZFTurbo's MIT-licensed Music-Source-Separation-Training, itself based on KUIELab's MIT-licensed TFC-TDF v3 architecture — both original copyrights are preserved in LICENSE.
- **Weights**: not covered by the above. See [What This Project Will NEVER Bundle](#what-this-project-will-never-bundle) — the aufr33/jarredou DrumSep checkpoints have no formally documented license; treat as non-commercial-safe only until clarified.

## Support

For bugs and feature requests, please open an issue on [GitHub](https://github.com/openmirlab/mdxnet-infer/issues).
