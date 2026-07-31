"""PyTorch -> MLX weight conversion for TfcTdfV3MLX, plus a strict load gate.

`convert_torch_to_mlx_weights` (vendored, see below) renames a PyTorch
`state_dict`'s keys to match the MLX module tree built by `model.py` --
`encoder_blocks.N.tfc_tdf.blocks.M.tfc1.0.*` -> `encoder_blocks_N.tfc_tdf.
blocks_M.tfc1_norm.*`, `Conv2d` OIHW -> MLX OHWI, `ConvTranspose2d` IOHW ->
OHWI. Upstream then loads the result with `model.load_weights(list(weights.
items()), strict=False)`, which **silently discards** any key that doesn't
match the module tree: a checkpoint can "load" with whole layers left at
random initialization and produce plausible-looking garbage with no error.
This module adds `load_converted_weights()`, which diffs the model's own
parameter keys (via `mlx.utils.tree_flatten`) against the converted weight
keys and raises a `ValueError` naming the mismatch *before* calling
`load_weights` -- callers should use this instead of calling `load_weights`
directly.

Vendored from:
    Project:  mlx-audio-separator (MIT License)
    Author:   ssmall256 (as named in upstream LICENSE)
    Repo:     https://github.com/ssmall256/mlx-audio-separator
    File:     mlx_audio_separator/separator/models/mdxc/loader.py
    Revision: 0ddc8cf5507906b52ac45a9cd9e6d26e881a93f8
    Copyright (c) 2024-2026 ssmall256. Permission is hereby granted, free of
    charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without
    restriction, subject to the MIT License terms in upstream's LICENSE file.
    (`convert_torch_to_mlx_weights`, `_translate_torch_key`, `_strip_prefix`,
    `_to_numpy`, `_is_conv_weight`, and `_is_conv_transpose_weight` are the
    vendored functions, renamed only where needed to avoid a name clash with
    this module's own `load_converted_weights`, which is new, not from
    upstream.)

Reads: mlx.core, mlx.nn, mlx.utils (tree_flatten), numpy
"""

import logging
import re
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)


def _strip_prefix(key: str) -> str:
    for prefix in ("state_dict.", "model.", "module."):
        if key.startswith(prefix):
            return _strip_prefix(key[len(prefix):])
    return key


def _translate_torch_key(key: str) -> Optional[str]:
    key = _strip_prefix(key)
    if key.startswith("stft."):
        return None

    if key == "first_conv.weight":
        return "first_conv.weight"

    # Encoder TFC/TDF blocks
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc1\.0\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc1_norm.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc1\.2\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc1_conv.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.0\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_norm1.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.2\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_linear1.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.3\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_norm2.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.5\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_linear2.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc2\.0\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc2_norm.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc2\.2\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc2_conv.{m.group(3)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.shortcut\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.shortcut.{m.group(3)}"

    # Encoder downscale
    m = re.match(r"^encoder_blocks\.(\d+)\.downscale\.conv\.0\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.downscale.norm.{m.group(2)}"
    m = re.match(r"^encoder_blocks\.(\d+)\.downscale\.conv\.2\.(.+)$", key)
    if m:
        return f"encoder_blocks_{m.group(1)}.downscale.conv.{m.group(2)}"

    # Bottleneck
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tfc1\.0\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tfc1_norm.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tfc1\.2\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tfc1_conv.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tdf\.0\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tdf_norm1.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tdf\.2\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tdf_linear1.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tdf\.3\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tdf_norm2.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tdf\.5\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tdf_linear2.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tfc2\.0\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tfc2_norm.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.tfc2\.2\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.tfc2_conv.{m.group(2)}"
    m = re.match(r"^bottleneck_block\.blocks\.(\d+)\.shortcut\.(.+)$", key)
    if m:
        return f"bottleneck_block.blocks_{m.group(1)}.shortcut.{m.group(2)}"

    # Decoder upscale
    m = re.match(r"^decoder_blocks\.(\d+)\.upscale\.conv\.0\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.upscale.norm.{m.group(2)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.upscale\.conv\.2\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.upscale.conv.{m.group(2)}"

    # Decoder TFC/TDF blocks
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc1\.0\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc1_norm.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc1\.2\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc1_conv.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.0\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_norm1.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.2\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_linear1.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.3\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_norm2.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tdf\.5\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tdf_linear2.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc2\.0\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc2_norm.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.tfc2\.2\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.tfc2_conv.{m.group(3)}"
    m = re.match(r"^decoder_blocks\.(\d+)\.tfc_tdf\.blocks\.(\d+)\.shortcut\.(.+)$", key)
    if m:
        return f"decoder_blocks_{m.group(1)}.tfc_tdf.blocks_{m.group(2)}.shortcut.{m.group(3)}"

    if key == "final_conv.0.weight":
        return "final_conv1.weight"
    if key == "final_conv.2.weight":
        return "final_conv2.weight"

    # Ignore non-parameter artifacts and unsupported paths explicitly.
    if key.endswith(".num_batches_tracked"):
        return None
    logger.debug("Unhandled MDX23C checkpoint key: %s", key)
    return None


def _is_conv_weight(mlx_key: str, value: np.ndarray) -> bool:
    if value.ndim != 4:
        return False
    if mlx_key.endswith("upscale.conv.weight"):
        return False
    return mlx_key.endswith(".weight") and any(
        token in mlx_key
        for token in (
            "first_conv.weight",
            "final_conv1.weight",
            "final_conv2.weight",
            "tfc1_conv.weight",
            "tfc2_conv.weight",
            "shortcut.weight",
            "downscale.conv.weight",
        )
    )


def _is_conv_transpose_weight(mlx_key: str, value: np.ndarray) -> bool:
    return value.ndim == 4 and mlx_key.endswith("upscale.conv.weight")


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a weight value to a numpy array, handling torch tensors."""
    try:
        return value.detach().cpu().numpy()
    except AttributeError:
        return np.array(value)


def convert_torch_to_mlx_weights(state_dict: Dict[str, Any]) -> Dict[str, mx.array]:
    """Convert a PyTorch TFC_TDF_net state dict to MLX weights + layout.

    Handles: key renaming to the MLX module tree (`_translate_torch_key`),
    `Conv2d` OIHW -> MLX OHWI, and `ConvTranspose2d` IOHW -> MLX OHWI (the
    two layouts are distinguished by which module they belong to, not by
    shape alone, since both are rank-4).
    """
    mlx_weights: Dict[str, mx.array] = {}
    for key, value in state_dict.items():
        mlx_key = _translate_torch_key(key)
        if mlx_key is None:
            continue
        weight = _to_numpy(value)
        if _is_conv_weight(mlx_key, weight):
            weight = np.transpose(weight, (0, 2, 3, 1))
        elif _is_conv_transpose_weight(mlx_key, weight):
            weight = np.transpose(weight, (1, 2, 3, 0))
        mlx_weights[mlx_key] = mx.array(weight)
    return mlx_weights


def load_converted_weights(model: nn.Module, mlx_weights: Dict[str, mx.array]) -> None:
    """Load `mlx_weights` into `model`, refusing a silent partial load.

    `model.load_weights(..., strict=False)` on its own accepts any degree of
    mismatch between the checkpoint and the module tree, dropping whatever
    doesn't line up without a warning. This checks first: every one of the
    model's own parameter keys (from `mlx.utils.tree_flatten(model.
    parameters())`) must be present in `mlx_weights`, and every key in
    `mlx_weights` must be consumed by the model -- otherwise a `ValueError`
    is raised naming counts and up to 5 example keys on each side, so a
    conversion bug or a mismatched checkpoint fails loudly instead of
    loading a partially-random model.
    """
    model_keys = {key for key, _ in tree_flatten(model.parameters())}
    weight_keys = set(mlx_weights.keys())

    unmatched_model = sorted(model_keys - weight_keys)
    dropped_weights = sorted(weight_keys - model_keys)

    if unmatched_model or dropped_weights:
        parts = []
        if unmatched_model:
            example = ", ".join(unmatched_model[:5])
            parts.append(
                f"{len(unmatched_model)} model parameters unmatched (e.g. {example})"
            )
        if dropped_weights:
            example = ", ".join(dropped_weights[:5])
            parts.append(
                f"{len(dropped_weights)} converted tensors dropped (e.g. {example})"
            )
        raise ValueError("MLX weight conversion incomplete: " + ", ".join(parts))

    model.load_weights(list(mlx_weights.items()), strict=False)
