"""Task-level MDX23C source-separation facade.

This is an additive front door over :mod:`mdxnet_infer.inference`.  The
existing engine remains the owner of model loading and separation; this module
only defines the input boundary and helper lifecycle.

Reads: inference (MDX23CInference, separate_drums)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .checkpoint_catalog import get_checkpoint_metadata
from .utils.cache import get_cache_dir
from .utils.download import download_file, sha256sum


class MDXNetSession:
    """Explicit, package-owned lifecycle for an MDX23C model.

    ``load`` owns checkpoint materialization and model construction, while
    ``infer`` is deliberately strict and requires a ready session.  The
    session retains no global model state; disk weights survive ``release``.
    """

    def __init__(self, *, model_name: str = "drumsep-6stem", model=None,
                 checkpoint_path=None, checkpoint_url=None,
                 checkpoint_metadata: Optional[dict] = None,
                 config_path=None, config=None, cache_dir=None, device=None,
                 progress: bool = True, **engine_options):
        self.model_name = model_name
        self._model = model
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_url = checkpoint_url
        self.config_path = Path(config_path) if config_path else None
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.progress = progress
        self._engine_options = dict(engine_options)
        self._metadata = dict(checkpoint_metadata or
                              (get_checkpoint_metadata(model_name) or {}))
        self._customized = any(value is not None for value in (
            checkpoint_path, checkpoint_url, checkpoint_metadata,
            config_path, config,
        ))
        self._status = "ready" if model is not None else "new"

    @property
    def status(self) -> str:
        return self._status

    def _expected(self, key: str):
        return self._metadata.get(key) or self._metadata.get(
            {"checkpoint_sha256": "ckpt_sha256", "config_sha256": "yaml_sha256"}.get(key, key)
        )

    def _target_path(self, path, url) -> Optional[Path]:
        """Resolve where a checkpoint/config artifact lives on disk, without
        touching it. An explicit ``path`` wins outright; otherwise an
        artifact resolves under ``cache_dir`` from ``url``'s filename.

        Pure path arithmetic, no I/O -- shared by :meth:`_materialize`
        (which downloads there if missing) and :meth:`is_cached` (which
        only checks for the file), so the two can never disagree on where
        an artifact lives.
        """
        if path is not None:
            return Path(path)
        if url:
            root = self.cache_dir or get_cache_dir()
            return root / Path(url).name
        return None

    def _materialize(self, path, url, sha_key):
        explicit = path is not None
        path = self._target_path(path, url)
        if path is not None and not explicit and not path.exists():
            download_file(url, path, progress=self.progress,
                          expected_sha256=self._expected(sha_key))
        if path is None:
            return None
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        expected = self._expected(sha_key)
        if expected and sha256sum(path).lower() != expected.lower():
            raise ValueError(f"checkpoint SHA-256 mismatch for {path}")
        return path

    def load(self) -> "MDXNetSession":
        if self._status == "ready":
            return self
        if self._status == "released":
            raise RuntimeError("cannot load a released MDXNetSession")
        self._status = "loading"
        try:
            from .inference import MDX23CInference
            # The package-local TOML snapshot is the default source for
            # checkpoint/config URLs.  Keep the legacy downloader as a
            # fallback for model names that have no package-owned entry.
            if not self._customized and not self._engine_options and not self._metadata:
                self._model = MDX23CInference.from_pretrained(
                    self.model_name, cache_dir=self.cache_dir,
                    device=self.device, progress=self.progress)
                self._status = "ready"
                return self
            ckpt_url = self.checkpoint_url or self._metadata.get("checkpoint_url")
            ckpt = self._materialize(self.checkpoint_path, ckpt_url,
                                     "checkpoint_sha256")
            config_url = self._metadata.get("config_url")
            yaml = self._materialize(self.config_path, config_url, "config_sha256")
            # A catalog entry must provide both artifacts.  If a custom
            # configuration intentionally omits a checkpoint, retain the
            # engine's existing behavior and let it construct without one.
            if not self._customized and self._metadata and (ckpt is None or yaml is None):
                raise ValueError(
                    f"checkpoint catalog entry for {self.model_name!r} "
                    "must provide checkpoint and config artifacts"
                )
            options = dict(self._engine_options)
            options.update(model_path=ckpt, config_path=yaml or self.config_path,
                           config=self.config, device=self.device,
                           model_name=self.model_name)
            self._model = MDX23CInference(**options)
            self._status = "ready"
            return self
        except Exception:
            self._status = "failed"
            raise

    def infer(self, audio, *, sample_rate: int = 44100, **options):
        if self._status != "ready" or self._model is None:
            raise RuntimeError("MDXNetSession must be ready; call load() before infer()")
        return self._model.separate(audio, sample_rate=sample_rate, **options)

    def release(self) -> None:
        if self._model is not None:
            model = getattr(self._model, "model", None)
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
            self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
        self._status = "released"

    def close(self) -> None:
        self.release()

    def is_cached(self) -> bool:
        """True if this session's checkpoint (and config, if applicable)
        already exist on disk, without downloading.

        Mirrors :meth:`load`'s own branch condition exactly: an
        uncustomized default session resolves via the catalog/
        :meth:`_materialize` path today, since ``checkpoints.toml`` already
        carries a ``drumsep-6stem`` entry -- not via
        ``MDX23CInference.from_pretrained``, which is only reached for
        model names absent from the catalog. Getting this branch wrong
        would silently check the wrong location for the common case.

        A customized session whose config intentionally omits a checkpoint
        (``load()`` then constructs the engine without one) reports
        ``False`` here: there is nothing on disk to call "cached".
        """
        if not self._customized and not self._engine_options and not self._metadata:
            from .inference import MDX23CInference
            return MDX23CInference.is_cached(self.model_name, cache_dir=self.cache_dir)
        ckpt_url = self.checkpoint_url or self._metadata.get("checkpoint_url")
        config_url = self._metadata.get("config_url")
        ckpt_target = self._target_path(self.checkpoint_path, ckpt_url)
        config_target = self._target_path(self.config_path, config_url)
        if ckpt_target is None or not ckpt_target.is_file():
            return False
        if config_target is not None and not config_target.is_file():
            return False
        return True

    def cache_info(self) -> dict:
        metadata = dict(self._metadata)
        return {"model": self.model_name, "status": self._status,
                "model_loaded": self._model is not None,
                "cached": self.is_cached(),
                "cache_dir": str(self.cache_dir or get_cache_dir()),
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "checkpoint_url": self.checkpoint_url or metadata.get("checkpoint_url"),
                "checkpoint_sha256": self._expected("checkpoint_sha256")}

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class MDXNetSeparator:
    """Reusable MDX23C source-separation helper.

    In-memory calls lazily create and cache one :class:`MDX23CInference`
    engine.  Path calls delegate to the compatible ``separate_drums``
    file-writing workflow, preserving its output-path contract.  One-shot
    functions create a fresh helper and therefore never share model state.
    """

    def __init__(self, engine=None, *, model_name: str = "drumsep-6stem",
                 cache_dir=None, device=None, progress: bool = True,
                 **engine_options):
        self._engine = engine
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.progress = progress
        self._engine_options = dict(engine_options)

    @property
    def engine(self):
        """Return the lazily initialized compatible inference engine."""
        if self._engine is None:
            from .inference import MDX23CInference

            if self._engine_options:
                self._engine = MDX23CInference(**self._engine_options)
            else:
                self._engine = MDX23CInference.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device=self.device,
                    progress=self.progress,
                )
        return self._engine

    def __call__(self, source, *, sample_rate: Optional[int] = None,
                 output_dir=None, combine_cymbals: bool = False):
        """Separate a path or waveform, delegating to existing APIs.

        Path inputs use the existing file-in/files-out ``separate_drums``
        function.  Array-like inputs require an explicit positive
        ``sample_rate`` and return the engine's stem dictionary.
        """
        if isinstance(source, (str, Path)):
            if sample_rate is not None:
                raise ValueError("sample_rate is not accepted for path inputs")
            from .inference import separate_drums

            return separate_drums(
                source,
                output_dir=output_dir,
                model_name=self.model_name,
                combine_cymbals=combine_cymbals,
                device=self.device,
                cache_dir=self.cache_dir,
                progress=self.progress,
            )
        if sample_rate is None:
            raise ValueError("sample_rate is required for in-memory audio")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        try:
            waveform = np.asarray(source)
        except (TypeError, ValueError):
            # Keep torch optional at this boundary; CPU/GPU tensors are
            # converted only when a caller actually supplies one.
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - package requires torch
                raise TypeError("audio must be a NumPy array or torch tensor") from exc
            if not isinstance(source, torch.Tensor):
                raise TypeError("audio must be a NumPy array or torch tensor")
            waveform = source.detach().cpu().numpy()
        if waveform.ndim not in (1, 2):
            raise ValueError("audio must have shape (samples,) or (samples, channels)")
        return self.engine.separate(
            waveform.astype(np.float32, copy=False), sample_rate=sample_rate,
        )

    def separate(self, source, *, sample_rate: Optional[int] = None,
                 output_dir=None, combine_cymbals: bool = False):
        """Named alias for :meth:`__call__`."""
        return self(source, sample_rate=sample_rate, output_dir=output_dir,
                    combine_cymbals=combine_cymbals)


def separate(source, *, model_name: str = "drumsep-6stem",
             sample_rate: Optional[int] = None, output_dir=None,
             combine_cymbals: bool = False, device=None, cache_dir=None,
             progress: bool = True):
    """One-shot clean separation; a fresh engine is used for each call."""
    return MDXNetSeparator(
        model_name=model_name, device=device, cache_dir=cache_dir,
        progress=progress,
    ).separate(source, sample_rate=sample_rate, output_dir=output_dir,
               combine_cymbals=combine_cymbals)


def separate_file(path: Union[str, Path], **kwargs):
    """One-shot separation for an audio path."""
    return separate(path, **kwargs)


def separate_tensor(audio, *, sample_rate: int, **kwargs):
    """One-shot separation for an in-memory waveform."""
    return separate(audio, sample_rate=sample_rate, **kwargs)


__all__ = ["MDXNetSeparator", "separate", "separate_file", "separate_tensor"]
