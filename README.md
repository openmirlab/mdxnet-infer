# mdxnet-infer

**Inference-only MDX23C TFC-TDF drum source separation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

---

## Overview

mdxnet-infer separates drum audio into individual component stems using the MDX23C TFC-TDF architecture. It ships two community-trained DrumSep models: a 5-stem and a 6-stem variant, both trained by aufr33 and jarredou. Model weights are downloaded automatically from Hugging Face on first use.

## Features

- 5-stem separation: kick, snare, toms, hi-hat, cymbals
- 6-stem separation: kick, snare, toms, hi-hat, ride, crash (with optional cymbal merging)
- High-level `separate()` function and `MDX23CInference` class API
- Time-Frequency Convolution with Time-Distributed Fully-connected (TFC-TDF) blocks
- 44.1 kHz stereo output

## Acknowledgments

mdxnet-infer is built on the MDX23C TFC-TDF architecture and the DrumSep model weights by aufr33 and jarredou. The architecture originates from the Music Demixing Challenge and the Music-Source-Separation-Training framework.

- **[MDX23C-8KFFT-InstVoc_HQ](https://arxiv.org/abs/2305.07489)** by Kimberley Jensen et al. — MDX23C architecture from ISMIR 2023 Music Demixing Challenge.
- **[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)** by ZFTurbo — TFC-TDF training framework.
- **[aufr33/drumsep](https://huggingface.co/aufr33/drumsep)** by aufr33 and jarredou — DrumSep pretrained weights.

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
@inproceedings{jensen2023mdx23c,
  title={Analysis of the ISMIR 2023 Melody Extraction Challenge},
  author={Kimberley Jensen and et al.},
  booktitle={Proceedings of the Music Demixing Challenge at ISMIR},
  year={2023}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Support

For bugs and feature requests, please open an issue on [GitHub](https://github.com/openmirlab/mdxnet-infer/issues).
