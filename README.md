# mdxnet-infer

Inference-only MDX23C TFC-TDF drum source separation package.

Separates drum audio into individual components (kick, snare, toms, hi-hat, cymbals)
using the aufr33/jarredou DrumSep models based on the MDX23C TFC-TDF architecture.

## Installation

```bash
pip install mdxnet-infer
```

## Usage

### Python API

```python
from mdxnet_infer import separate, MDX23CInference

# High-level convenience function (downloads model automatically)
output_paths = separate("drums.wav", output_dir="output/", model_name="drumsep-6stem")

# Lower-level API with more control
engine = MDX23CInference.from_pretrained("drumsep-6stem", device="cuda")
import librosa, numpy as np
audio, sr = librosa.load("drums.wav", sr=None, mono=False)
stems = engine.separate(audio.T, sample_rate=sr)
# stems -> {"kick": array, "snare": array, "toms": array, "hh": array, "ride": array, "crash": array}
```

### CLI

```bash
# 6-stem separation (kick, snare, toms, hh, ride, crash)
mdxnet-infer drums.wav -o output/ --model drumsep-6stem

# 5-stem separation (kick, snare, toms, hh, cymbals)
mdxnet-infer drums.wav -o output/ --model drumsep-5stem

# Combine ride+crash into single cymbals stem (6-stem model)
mdxnet-infer drums.wav -o output/ --model drumsep-6stem --combine-cymbals

# Run on specific device
mdxnet-infer drums.wav -o output/ --device cuda
```

## Supported Models

| Model | Stems | Description |
|-------|-------|-------------|
| `drumsep-6stem` | kick, snare, toms, hh, ride, crash | aufr33-jarredou MDX23C DrumSep v0.1 |
| `drumsep-5stem` | kick, snare, toms, hh, cymbals | jarredou 5-stem DrumSep |

## Architecture

Built on the Music-Source-Separation-Training TFC-TDF architecture:
- Time-Frequency Convolution with Time-Distributed Fully-connected (TFC-TDF) blocks
- Encoder-decoder with skip connections
- Multi-subband processing
- 44.1kHz stereo output
