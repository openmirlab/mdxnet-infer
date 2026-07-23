"""
mdxnet-infer: Inference-only MDX23C TFC-TDF drum source separation.

Provides a simple API for separating drum audio into individual stem components
using the aufr33/jarredou DrumSep models.

Example usage::

    from mdxnet_infer import separate, MDX23CInference

    # High-level convenience function
    output_paths = separate("drums.wav", output_dir="output/")

    # Lower-level API
    engine = MDX23CInference.from_pretrained("drumsep-6stem")
    stems = engine.separate(audio, sample_rate=44100)
"""

from .__about__ import __version__
from .model import TFC_TDF_net
from .inference import MDX23CInference
from .config import MDX23CConfig

# High-level convenience function
from .inference import separate_drums as separate
from .clean_api import MDXNetSeparator, MDXNetSession, separate_file, separate_tensor
from .checkpoint_catalog import CHECKPOINT_CATALOG, get_checkpoint_metadata

__all__ = [
    "__version__",
    "TFC_TDF_net",
    "MDX23CInference",
    "MDX23CConfig",
    "separate",
    "MDXNetSeparator",
    "MDXNetSession",
    "CHECKPOINT_CATALOG",
    "get_checkpoint_metadata",
    "separate_file",
    "separate_tensor",
]
