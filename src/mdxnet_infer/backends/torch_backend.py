"""PyTorch backend -- the shipped path, unchanged, behind the seam.

Holds an already-constructed `MDX23CInference` and turns a mixture into
stems by calling its existing `separate()` unmodified; this is a wrapper,
never a second implementation, so the Torch path cannot drift from the
`MDX23CInference` advanced-composition API that callers still import
directly (the accuracy rule in this package's CLAUDE.md -- `model.py`'s
forward pass must stay byte-for-byte identical to upstream -- extends to
this file: it must never reimplement the loop `MDX23CInference.separate()`
already owns).

Reads: ..inference (MDX23CInference), .base (SeparationBackend), numpy
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TorchBackend:
    """Wraps a constructed `MDX23CInference` as a `SeparationBackend`."""

    name = "torch"

    def __init__(self, engine):
        self._engine = engine

    @classmethod
    def is_available(cls) -> bool:
        return True

    @property
    def resolved_device(self) -> str:
        return str(self._engine.device)

    @property
    def engine(self):
        """The resident `MDX23CInference`, for callers composing the advanced path directly."""
        return self._engine

    def separate(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 44100,
        batch_size: Optional[int] = None,
        overlap: Optional[int] = None,
        progress: bool = True,
    ) -> dict:
        return self._engine.separate(
            audio,
            sample_rate=sample_rate,
            batch_size=batch_size,
            overlap=overlap,
            progress=progress,
        )

    def release(self) -> None:
        model = getattr(self._engine, "model", None)
        if model is not None and hasattr(model, "cpu"):
            model.cpu()
        self._engine = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                torch.mps.empty_cache()
        except ImportError:  # pragma: no cover - torch is a hard dependency
            pass
