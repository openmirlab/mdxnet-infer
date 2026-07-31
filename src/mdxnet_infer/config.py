"""
Configuration dataclass tree for MDX23C models (audio/model/training/inference).

Supports loading from YAML config files used by jarredou DrumSep models
(``from_yaml``/``from_mapping``), plus two hard-coded presets
(``drumsep_6stem``, ``drumsep_5stem``) that mirror the two known checkpoints'
actual training configs. ``from_mapping`` silently drops unknown fields
(training-only keys not in a given dataclass), matching the source YAML
files' own leniency; it raises ``ValueError`` only if ``data`` itself is not
a mapping.

Reads: yaml (external only; no internal package imports)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import yaml


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    # NOTE: not authoritative -- runtime never reads this field. `inference.py`
    # derives the actual chunk size at separation time from
    # `hop_length * (inference.dim_t - 1)` (see `inference.py`'s
    # `separate()`). For `drumsep_6stem` the two happen to agree
    # (512 * 255 = 130560); for `drumsep_5stem` they do NOT
    # (this field stores 523776, but runtime computes 512 * 511 = 261632).
    # Retained for config-shape compatibility with the source YAML files,
    # not deleted -- do not treat it as ground truth.
    chunk_size: int = 523776
    dim_f: int = 1024
    dim_t: int = 1024
    hop_length: int = 512
    n_fft: int = 2048
    num_channels: int = 2
    sample_rate: int = 44100
    min_mean_abs: float = 0.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    act: str = 'gelu'
    bottleneck_factor: int = 4
    growth: int = 128
    norm: str = 'InstanceNorm'
    num_blocks_per_scale: int = 2
    num_channels: int = 128
    num_scales: int = 5
    num_subbands: int = 4
    scale: Tuple[int, int] = (2, 2)


@dataclass
class TrainingConfig:
    """Training configuration (used for stem info)."""
    instruments: List[str] = field(default_factory=lambda: [
        'kick', 'snare', 'toms', 'hh', 'ride', 'crash'
    ])
    target_instrument: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference configuration."""
    batch_size: int = 1
    dim_t: int = 256
    num_overlap: int = 4
    normalize: bool = False


@dataclass
class MDX23CConfig:
    """Complete MDX23C model configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'MDX23CConfig':
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict) -> 'MDX23CConfig':
        """Build a configuration from a YAML- or registry-shaped mapping.

        The registry stores the architecture needed to construct a named
        model before its remote YAML artifact is materialized.  Unknown
        training-only fields are intentionally ignored, just as for the
        source YAML files.
        """
        if not isinstance(data, dict):
            raise ValueError("MDX23C config must be a mapping")

        audio_data = data.get('audio', {})
        model_data = data.get('model', {})
        training_data = data.get('training', {})
        inference_data = data.get('inference', {})

        # Handle scale as list -> tuple
        if 'scale' in model_data and isinstance(model_data['scale'], list):
            model_data['scale'] = tuple(model_data['scale'])

        return cls(
            audio=AudioConfig(**{k: v for k, v in audio_data.items()
                                if k in AudioConfig.__dataclass_fields__}),
            model=ModelConfig(**{k: v for k, v in model_data.items()
                               if k in ModelConfig.__dataclass_fields__}),
            training=TrainingConfig(**{k: v for k, v in training_data.items()
                                      if k in TrainingConfig.__dataclass_fields__}),
            inference=InferenceConfig(**{k: v for k, v in inference_data.items()
                                        if k in InferenceConfig.__dataclass_fields__}),
        )

    @classmethod
    def drumsep_6stem(cls) -> 'MDX23CConfig':
        """Default config for aufr33-jarredou 6-stem DrumSep model."""
        return cls(
            audio=AudioConfig(
                chunk_size=130560,
                dim_f=1024,
                dim_t=256,
                hop_length=512,
                n_fft=2048,
                num_channels=2,
                sample_rate=44100,
                min_mean_abs=0.001,
            ),
            model=ModelConfig(
                act='gelu',
                bottleneck_factor=4,
                growth=128,
                norm='InstanceNorm',
                num_blocks_per_scale=2,
                num_channels=128,
                num_scales=5,
                num_subbands=4,
                scale=(2, 2),
            ),
            training=TrainingConfig(
                instruments=['kick', 'snare', 'toms', 'hh', 'ride', 'crash'],
                target_instrument=None,
            ),
            inference=InferenceConfig(
                batch_size=1,
                dim_t=256,
                num_overlap=4,
            ),
        )

    @classmethod
    def drumsep_5stem(cls) -> 'MDX23CConfig':
        """Default config for jarredou 5-stem DrumSep model."""
        return cls(
            audio=AudioConfig(
                chunk_size=523776,
                dim_f=1024,
                dim_t=1024,
                hop_length=512,
                n_fft=2048,
                num_channels=2,
                sample_rate=44100,
                min_mean_abs=0.0,
            ),
            model=ModelConfig(
                act='gelu',
                bottleneck_factor=4,
                growth=128,
                norm='InstanceNorm',
                num_blocks_per_scale=2,
                num_channels=128,
                num_scales=5,
                num_subbands=4,
                scale=(2, 2),
            ),
            training=TrainingConfig(
                instruments=['kick', 'snare', 'toms', 'hh', 'cymbals'],
                target_instrument=None,
            ),
            inference=InferenceConfig(
                batch_size=2,
                dim_t=512,
                num_overlap=4,
                normalize=False,
            ),
        )
