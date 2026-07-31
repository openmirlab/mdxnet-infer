"""The backend seam -- one narrow protocol every compute backend implements.

A backend owns everything framework-specific: model construction, checkpoint
weights, tensor layout, device placement, and the chunked overlap-average
itself. Chunking accumulates on-device for speed, so the seam deliberately
sits at one whole mixture rather than a single chunk -- a per-chunk seam
would drag every accumulator back to the host and hand the acceleration
straight back (same reasoning as the sibling `bs-roformer-infer` package's
seam; see its `brain/architecture.md`).

One deliberate divergence from that sibling package, worth stating plainly
because it is *not* a copy-paste of its seam: there, `SeparationBackend.
separate()` takes a bare `(channels, samples)` array because the model never
needs resampling or a batch-size knob at this layer. Here, `MDX23CInference.
separate()` -- the method this whole seam exists to keep behavior-identical
to -- already owns resampling, stereo-shape normalization, and an explicit
`batch_size`/`overlap` API as part of "separate one mixture," so duplicating
a narrower channels-first seam on top would just be a second shape
convention for the same concept. `SeparationBackend.separate()` here mirrors
`MDX23CInference.separate()`'s own signature and shape conventions instead.

Above the seam nothing knows a dtype, a tensor layout, or which chip is busy:
folder iteration, stem naming, and file writing are decided once in
`inference.py`.

Reads: numpy (boundary array type only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ChunkingPlan:
    """How one mixture is cut into overlapping chunks.

    Not actually a single owner for both backends today: `mlx_backend.py`
    builds this from config and uses it. `torch_backend.py` never touches
    it -- it wraps `MDX23CInference.separate()` unmodified (see that
    module's docstring), and that method still computes the identical
    arithmetic inline (`inference.py`'s `chunk_size = hop_length * (dim_t -
    1)`, `hop_size = chunk_size // overlap`) rather than constructing a
    `ChunkingPlan`. So the same design decision is currently encoded in two
    places -- here and in `inference.py` -- not one;
    `tests/test_backends.py::test_chunking_plan_matches_inference_module_arithmetic`
    cross-checks that they still agree for the registry's presets, so a
    future change to either side that breaks that agreement fails loudly
    instead of drifting silently. Consolidating
    `inference.py` onto this dataclass is deferred, not done.

    Unlike the sibling package's `ChunkingPlan`, `pad_size` is not part of
    this plan -- it depends on the mixture's own length (`hop_size - (length
    - chunk_size) % hop_size`), not on config alone, so it is computed
    per-call in each backend's `separate()`, exactly where
    `MDX23CInference.separate()` computes it today.
    """

    chunk_size: int
    hop_size: int
    overlap: int

    @classmethod
    def from_config(cls, config, *, overlap: Optional[int] = None) -> "ChunkingPlan":
        resolved_overlap = overlap if overlap is not None else config.inference.num_overlap
        dim_t = config.inference.dim_t
        chunk_size = config.audio.hop_length * (dim_t - 1)
        hop_size = chunk_size // resolved_overlap
        return cls(chunk_size=chunk_size, hop_size=hop_size, overlap=resolved_overlap)


@runtime_checkable
class SeparationBackend(Protocol):
    """Turns one mixture into stems, hiding how and where it computed them."""

    #: Stable identifier, matching the `backend=` argument that selects it.
    name: str

    @classmethod
    def is_available(cls) -> bool:
        """True when this backend can actually run on this machine right now."""

    @property
    def resolved_device(self) -> str:
        """The concrete target chosen, after any `auto`/`None` sentinel was resolved."""

    def separate(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 44100,
        batch_size: Optional[int] = None,
        overlap: Optional[int] = None,
        progress: bool = True,
    ) -> dict:
        """Separate one mixture into named stems.

        `audio` accepts the same shapes `MDX23CInference.separate()` does:
        `(samples,)` mono, `(samples, 2)`, or `(2, samples)`. Returns a dict
        mapping stem name to `(samples, channels)` float32, matching that
        method's return convention so callers above the seam (`inference.py`,
        `clean_api.py`) do not need to know which backend ran.
        """

    def release(self) -> None:
        """Drop resident model and device memory. Disk checkpoints stay."""


class BackendUnavailable(RuntimeError):
    """Raised when a backend is requested by name but cannot run here.

    Always raised, never swallowed into a fallback: silently substituting a
    different backend discards what the caller explicitly asked for, and it
    would only be discovered by noticing the wrong hardware was busy.
    """
