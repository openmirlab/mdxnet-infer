"""Thin barrel for the vendored MLX TFC_TDF_v3 model -- lazy on purpose.

`mdxnet_infer.mlx` is only ever imported on demand by the (caller-owned) MLX
compute backend, never by the package's default import path, so `import
mdxnet_infer` stays MLX-free even with this subpackage present. Attribute
access is deferred via `__getattr__` so that merely importing this package
doesn't eagerly import `mlx.core`/`mlx.nn` -- the cost of that import is
paid only when a name is actually used.

Reads: .model (TfcTdfV3MLX, lazily), .convert (convert_torch_to_mlx_weights,
load_converted_weights, lazily)
"""

from __future__ import annotations

__all__ = [
    "TfcTdfV3MLX",
    "convert_torch_to_mlx_weights",
    "load_converted_weights",
]


def __getattr__(name: str):
    if name == "TfcTdfV3MLX":
        from .model import TfcTdfV3MLX

        return TfcTdfV3MLX
    if name in ("convert_torch_to_mlx_weights", "load_converted_weights"):
        from . import convert

        return getattr(convert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
