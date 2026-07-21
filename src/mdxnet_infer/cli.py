"""
CLI entry point for mdxnet-infer.

Usage::

    mdxnet-infer input.wav -o output/ --model drumsep-6stem
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Entry point for the ``mdxnet-infer`` CLI command."""
    parser = argparse.ArgumentParser(
        prog="mdxnet-infer",
        description="MDX23C TFC-TDF drum source separation",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input audio file (WAV, FLAC, MP3, etc.)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        dest="output_dir",
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--model",
        default="drumsep-6stem",
        choices=["drumsep-6stem"],
        help=(
            "Pretrained model to use (default: drumsep-6stem). "
            "drumsep-5stem is currently unavailable -- see README."
        ),
    )
    parser.add_argument(
        "--combine-cymbals",
        action="store_true",
        default=False,
        help=(
            "Merge ride and crash into a single 'cymbals' stem "
            "(only relevant for drumsep-6stem)"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Inference device: 'cuda', 'cuda:N', 'cpu', 'mps', or 'auto' "
            "(auto-detected if omitted or 'auto')"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        dest="cache_dir",
        help=(
            "Directory for cached model weights "
            "(default: ~/.cache/mdxnet-infer/, or $MDXNET_INFER_CACHE_DIR)"
        ),
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    from .inference import separate_drums

    try:
        output_paths = separate_drums(
            audio_path=args.input,
            output_dir=args.output_dir,
            model_name=args.model,
            combine_cymbals=args.combine_cymbals,
            device=args.device,
            cache_dir=args.cache_dir,
            progress=not args.quiet,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"\nSeparation complete. {len(output_paths)} stems written.")


if __name__ == "__main__":
    main()
