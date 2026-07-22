# mdxnet-infer

**Inference-only MDX23C TFC-TDF source separation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

---

## Why This Exists

mdxnet-infer separates audio into component stems using the MDX23C TFC-TDF architecture. The framework this architecture comes from — [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) — is a large, general-purpose repo covering dozens of architectures and full training pipelines; mdxnet-infer reprovides just the inference-only subset needed to run a curated MDX23C registry, as a small, pip/uv-installable package with no training code and no unrelated architectures.

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

- DrumSep, vocals/instrumental, dereverb, four-stem, and SFX recipes
- DrumSep-only `separate()` / `separate_drums()` plus generic `separate_file()`
  and `MDX23CInference` APIs
- Package-qualified `MDXNetSeparator` facade for reusable task-level calls
- Explicit `MDXNetSession` lifecycle with package-owned checkpoint metadata
- Time-Frequency Convolution with Time-Distributed Fully-connected (TFC-TDF) blocks
- 44.1 kHz stereo output

## Scope

**In scope:**
- Inference-only separation for the registry's MDX23C recipes via CLI and Python API
- The TFC-TDF v3 architecture, ported verbatim for inference (no training loop)
- Automatic and manual weight management with sha256 verification

**Out of scope, forever:**
- Training code — use the upstream [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) repo for that
- Architectures/checkpoints other than the explicitly listed MDX23C TFC-TDF recipes
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

# Generic model (cymbal merging is intentionally DrumSep-only)
mdxnet-infer song.wav -o output/ --model d1581

# Run on a specific device
mdxnet-infer drums.wav -o output/ --device cuda

# Use a custom weights folder (see "What This Project Will NEVER Bundle" below)
mdxnet-infer drums.wav -o output/ --cache-dir /path/to/weights
```

### Python API

```python
from mdxnet_infer import separate, separate_file, MDX23CInference

# High-level convenience function
output_paths = separate("drums.wav", output_dir="output/", model_name="drumsep-6stem")

# Generic MDX23C file API
output_paths = separate_file("song.wav", output_dir="output/", model_name="d1581")

# Lower-level API with more control
engine = MDX23CInference.from_pretrained("drumsep-6stem", device="cuda")
import librosa
audio, sr = librosa.load("drums.wav", sr=None, mono=False)
stems = engine.separate(audio.T, sample_rate=sr)
# stems -> {"kick": array, "snare": array, "toms": array, "hh": array, "ride": array, "crash": array}
```

### Clean task-level facade

`MDXNetSeparator` is an additive front door over the existing inference
engine. In-memory arrays or tensors require an explicit positive `sample_rate`
and are normalized to float32 before delegation; path inputs retain the
compatible file-in/files-out `separate()` behavior. A helper lazily creates
and reuses one engine, while one-shot functions create a fresh helper for each
call.

```python
from mdxnet_infer import MDXNetSeparator, separate_tensor

separator = MDXNetSeparator(model_name="drumsep-6stem", device="cpu")
stems = separator.separate(audio, sample_rate=sr)
stems = separate_tensor(audio, sample_rate=sr, device="cpu", progress=False)
```

The lower-level `MDX23CInference` API remains available for custom checkpoint,
configuration, chunking, and overlap composition.

### Explicit model lifecycle

`MDXNetSession` provides a strict lifecycle for services and repeated calls.
`load()` downloads (or verifies) the release-pinned checkpoint and config,
`infer()` requires a ready session, and `release()` frees the in-memory model
while leaving the disk cache intact. The package ships its release-pinned
metadata in `mdxnet_infer/config/checkpoints.toml` (including separate
checkpoint and YAML config artifacts), while custom `checkpoint_path`, `checkpoint_url`, and
`checkpoint_metadata` values remain supported.

```python
from mdxnet_infer import MDXNetSession

with MDXNetSession(model_name="drumsep-6stem", device="cpu") as session:
    stems = session.infer(audio, sample_rate=44100)
```

The session is package-owned and has no dependency on a central runtime or
catalog service; external applications can wrap it with their own policies.
`release()` permits a later `load()`; `close()` is terminal. Device requests
accept legacy `None`/`auto` selection (CUDA, then MPS, then CPU), or explicit
`cpu`, `cuda`, `cuda:N`, and `mps`; malformed or unavailable explicit choices
raise before model construction.

## Supported Models

| Stable name | Stems | API family / attribution |
|-------|-------|-------------|
| `drumsep-6stem` | kick, snare, toms, hh, ride, crash | DrumSep; aufr33/jarredou |
| `instvoc-hq1` | Vocals, Instrumental | Generic; InstVoc HQ 1 |
| `instvoc-hq2` | Vocals, Instrumental | Generic; InstVoc HQ 2 |
| `d1581` | Vocals, Instrumental | Generic; D1581 (12k FFT) |
| `4stem-zfturbo` | vocals, bass, drums, other | Generic; ZFTurbo |
| `dereverb-aufr33-jarredou` | dry, other | Generic; aufr33/jarredou |
| `sfx-jasper` | foreground, background | Generic; Jasper (1.34 GB, lazy download only) |

The underscore source names (for example `mdx23c_d1581`) remain compatibility
aliases, but new code should use these stable names. `instvoc-zfturbo` is
excluded because its checkpoint is byte-identical to HQ1; `drumsep-5stem`
remains provenance-pending; and `mid_side` / `orch` remain excluded because
their single-target output semantics lack a generic API contract.

## What This Project Will NEVER Bundle

mdxnet-infer will **never** ship checkpoint files inside the pip/wheel package itself. Weights are downloaded on first use (or placed manually for offline use) into a local cache directory, and every file — fresh download or cached — is checked against recorded sha256 digests before it's loaded; a mismatch raises `ChecksumMismatchError` and deletes the bad file rather than silently loading it. `src/mdxnet_infer/config/checkpoints.toml` is the sole authority for recipe URLs, full digests, source revisions, provenance, and license statements.

- **6-stem model** (`drumsep-6stem`): `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt`, originally from the "aufr33-jarredou MDX23C DrumSep v0.1" release on `github.com/jarredou/models`.
- **Original hosting is gone; re-hosted under org control.** As of the 2026-07 campaign, the `github.com/jarredou/models` account/repository no longer exists — the original release assets are unreachable. openmirlab located the checkpoint and its config on two independent third-party Hugging Face mirrors, confirmed the two mirrors were **byte-identical** (matching sha256 digests), and re-published both files as an openmirlab-controlled GitHub Release so `mdxnet-infer` no longer depends on a third-party account staying online (org constitution art.4: weights hosting must be under org control or a verified mirror, never a single third-party account).
- **Download URLs and checksums** (auto-downloaded by the package; see [Manual download](#manual-download-and-offline-use) for the manual recipe):

  | File | URL | sha256 |
  |------|-----|--------|
  | `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt` | `https://github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt` | `d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0` |
  | `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml` | `https://github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml` | `17d1649a227f841165bdb4c11a42082898192a1ea3ceab7e7e0b9293d6589dd6` |

- **5-stem model (`drumsep-5stem`) is lost upstream — no longer available.** It was also hosted only on `github.com/jarredou/models` (the "DrumSep" release, `drumsep_5stems_mdx23c_jarredou.ckpt`), which is gone. Unlike the 6-stem model, no intact original-quality mirror could be found anywhere on the web during this campaign — the only surviving copy is a derived OpenVINO conversion (`Intel/drumsep_mdx23c_jarredou_openvino`), which is not usable as a drop-in PyTorch checkpoint. `drumsep-5stem` has therefore been **removed from `KNOWN_MODELS`** (no auto-download path); the architecture config (`MDX23CConfig.drumsep_5stem()`) remains available for anyone who already has their own copy of the checkpoint. It may return to `KNOWN_MODELS` if a verified original-format mirror surfaces.
- **License is not formally documented by aufr33/jarredou.** No LICENSE file or explicit terms accompanied the original GitHub Release or the linked Colab notebooks. Third-party re-uploads disagree: a Hugging Face mirror of the same checkpoint (`Politrees/UVR_resources`) is tagged `license:mit` (uploader-asserted, not from the original authors), while the derived OpenVINO conversion (`Intel/drumsep_mdx23c_jarredou_openvino`) is tagged `license:cc-by-nc-sa-4.0`. Until the original authors confirm terms, treat these weights as **non-commercial-safe only** — do not assume MIT for the weights even though the code around them is MIT. openmirlab is a non-commercial project, so this is usable here, but downstream users with commercial intent should not assume the same.
- **Generic model mirrors are not license grants.** The six additional recipes
  are revision-pinned downloads from the community
  `noblebarkrr/mvsepless_resources` mirror. Their SHA-256 values were probed,
  but the mirror does not establish the original authors' license; treat them
  as non-commercial-safe only pending author confirmation.

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
