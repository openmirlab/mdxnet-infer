"""Cache directory management for mdxnet-infer model weights."""

import os
import shutil
from pathlib import Path
from typing import Optional


def get_cache_dir(subdir: Optional[str] = None) -> Path:
    """Return the cache directory for mdxnet-infer model weights.

    Resolution order:

    1. ``MDXNET_INFER_CACHE_DIR`` environment variable
    2. ``~/.cache/mdxnet-infer/``

    Args:
        subdir: Optional sub-directory name appended to the base path.

    Returns:
        Resolved :class:`~pathlib.Path` to the cache directory.
    """
    custom = os.environ.get("MDXNET_INFER_CACHE_DIR")
    if custom:
        base = Path(custom).expanduser().resolve()
    else:
        base = Path.home() / ".cache" / "mdxnet-infer"

    return base / subdir if subdir else base


def clear_cache(verbose: bool = True) -> bool:
    """Delete all cached model weights.

    Args:
        verbose: Print status messages.

    Returns:
        ``True`` on success, ``False`` on error.
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        if verbose:
            print(f"Cache directory does not exist: {cache_dir}")
        return True
    try:
        shutil.rmtree(cache_dir)
        if verbose:
            print(f"Cleared cache: {cache_dir}")
        return True
    except Exception as exc:
        if verbose:
            print(f"Error clearing cache: {exc}")
        return False
