"""Model weight download utilities for mdxnet-infer.

`download_file()` fetches a URL to disk with an optional sha256 check;
`sha256sum()` is the standalone digest helper `inference.py` also uses to
re-verify already-cached weight files (Weights UX contract, org
constitution art.4: auto-download must be sha256-verified).

Reads: (none — leaf utility)
"""

import hashlib
from pathlib import Path

from tqdm import tqdm


class ChecksumMismatchError(ValueError):
    """Raised when a downloaded file's sha256 doesn't match the expected digest."""


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the sha256 hex digest of a file on disk.

    Args:
        path: File to hash.
        chunk_size: Bytes read per iteration (default 1 MB).

    Returns:
        Lowercase hex digest string.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    progress: bool = True,
    expected_sha256: str = None,
) -> None:
    """Download a file from ``url`` to ``destination``.

    Args:
        url: HTTP/HTTPS URL to download.
        destination: Local path to write the file.
        progress: Show a tqdm progress bar.
        expected_sha256: If given, the downloaded file's sha256 digest is
            checked against this value. On mismatch, the (corrupted or
            tampered) file is deleted and :class:`ChecksumMismatchError`
            is raised.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status.
        ChecksumMismatchError: If ``expected_sha256`` is given and doesn't
            match the downloaded file.
    """
    import requests

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB

    if progress and total_size > 0:
        iterator = tqdm(
            response.iter_content(chunk_size=block_size),
            total=total_size // block_size + 1,
            unit="MB",
            desc=destination.name,
        )
    else:
        iterator = response.iter_content(chunk_size=block_size)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in iterator:
            if chunk:
                f.write(chunk)

    if expected_sha256:
        actual = sha256sum(destination)
        if actual.lower() != expected_sha256.lower():
            destination.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"sha256 mismatch for {destination.name}: "
                f"expected {expected_sha256}, got {actual}. "
                "The downloaded file has been removed; the release asset "
                "may be corrupted, or the URL may no longer point at the "
                "expected file."
            )
