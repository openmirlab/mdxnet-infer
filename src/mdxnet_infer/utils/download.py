"""Model weight download utilities for mdxnet-infer."""

from pathlib import Path

from tqdm import tqdm


def download_file(url: str, destination: Path, progress: bool = True) -> None:
    """Download a file from ``url`` to ``destination``.

    Args:
        url: HTTP/HTTPS URL to download.
        destination: Local path to write the file.
        progress: Show a tqdm progress bar.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status.
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
