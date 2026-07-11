# mdxnet-infer

**Inference-only MDX23C TFC-TDF drum source separation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

---

## Overview

mdxnet-infer separates drum audio into individual component stems using the MDX23C TFC-TDF architecture. It ships two community-trained DrumSep models: a 5-stem and a 6-stem variant, both trained by aufr33 and jarredou. Model weights are downloaded automatically on first use (see [Weights provenance](#weights-provenance) below — the current hosting has a known issue).

## Features

- 5-stem separation: kick, snare, toms, hi-hat, cymbals
- 6-stem separation: kick, snare, toms, hi-hat, ride, crash (with optional cymbal merging)
- High-level `separate()` function and `MDX23CInference` class API
- Time-Frequency Convolution with Time-Distributed Fully-connected (TFC-TDF) blocks
- 44.1 kHz stereo output

## Acknowledgments

mdxnet-infer is built on the MDX23C TFC-TDF architecture and the DrumSep model weights by aufr33 and jarredou.

- **TFC-TDF v3 architecture** originates from [kuielab/sdx23](https://github.com/kuielab/sdx23/) (KUIELab's entry to the Sound Demixing Challenge 2023, MIT licensed).
- **[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)** by Roman Solovyev (ZFTurbo) — the training/inference framework this package's `model.py` is a verbatim-ported, inference-only subset of (`models/mdx23c_tfc_tdf_v3.py`, MIT licensed). See [Citation](#citation) for the associated benchmark paper.
- **DrumSep model weights** by aufr33 and jarredou — see [Weights provenance](#weights-provenance) below; these are separate community-trained checkpoints, not part of the ZFTurbo/KUIELab code release, and their license terms are not formally documented by the original authors.

## Weights provenance

- **6-stem model** (`drumsep-6stem`): `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt`, from the "aufr33-jarredou MDX23C DrumSep v0.1" release.
- **5-stem model** (`drumsep-5stem`): `drumsep_5stems_mdx23c_jarredou.ckpt`, from the "DrumSep" release.
- **Known issue (unresolved):** both checkpoints were hosted as GitHub Release assets under `github.com/jarredou/models`. As of this campaign (2026-07) that GitHub account/repository no longer exists — **the download URLs in `inference.py` currently 404**. `mdxnet-infer` cannot fetch these two models until they are re-hosted. Third-party mirrors exist (e.g. a checkpoint file on `Politrees/UVR_resources` on Hugging Face) but have not been verified byte-for-byte against the original release assets, so none is wired in as a replacement yet.
- **License is not formally documented by aufr33/jarredou.** No LICENSE file or explicit terms accompanied the original GitHub Release or the linked Colab notebooks. Third-party re-uploads disagree: a Hugging Face mirror of the same checkpoint (`Politrees/UVR_resources`) is tagged `license:mit` (uploader-asserted, not from the original authors), while a derived OpenVINO conversion (`Intel/drumsep_mdx23c_jarredou_openvino`) is tagged `license:cc-by-nc-sa-4.0`. Until the original authors confirm terms, treat these weights as **non-commercial-safe only** — do not assume MIT for the weights even though the code around them is MIT.
- Per org policy, weights are never bundled in this repo and must be hosted under org control (or a verified upstream mirror) before a release ships; this is unresolved and blocks the release gate.

## Quick Start

```python
from mdxnet_infer import separate

# Separate a drum file (downloads model automatically)
output_paths = separate("drums.wav", output_dir="output/", model_name="drumsep-6stem")
```

## Installation

### With UV (Recommended)

```bash
uv add mdxnet-infer
```

### With pip

```bash
pip install mdxnet-infer
```

## Usage

### CLI

```bash
# 6-stem separation (kick, snare, toms, hh, ride, crash)
mdxnet-infer drums.wav -o output/ --model drumsep-6stem

# 5-stem separation (kick, snare, toms, hh, cymbals)
mdxnet-infer drums.wav -o output/ --model drumsep-5stem

# Combine ride+crash into a single cymbals stem
mdxnet-infer drums.wav -o output/ --model drumsep-6stem --combine-cymbals

# Run on a specific device
mdxnet-infer drums.wav -o output/ --device cuda
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
| `drumsep-5stem` | kick, snare, toms, hh, cymbals | jarredou 5-stem DrumSep |

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

(Citation as used by the upstream [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) repository this architecture is ported from.)

## License

- **Code**: MIT License. See [LICENSE](LICENSE) for details. `src/mdxnet_infer/model.py` is a verbatim-ported, inference-only subset of ZFTurbo's MIT-licensed Music-Source-Separation-Training, itself based on KUIELab's MIT-licensed TFC-TDF v3 architecture — both original copyrights are preserved in LICENSE.
- **Weights**: not covered by the above. See [Weights provenance](#weights-provenance) — the aufr33/jarredou DrumSep checkpoints have no formally documented license; treat as non-commercial-safe only until clarified.

## Support

For bugs and feature requests, please open an issue on [GitHub](https://github.com/openmirlab/mdxnet-infer/issues).
