"""Shared `.ckpt` state-dict unwrapping -- one owner for a format both
backends read.

Both `MDX23CInference._load_weights` (Torch) and `mdxnet_infer.backends.
mlx_backend` (MLX weight conversion) need the same raw bytes: a PyTorch
checkpoint file that may be a bare state dict or one wrapped under a
``state_dict``/``model_state_dict`` key. Keeping that unwrap logic in two
places would be the same design decision encoded twice; this is the one
place it lives.

Reads: torch (leaf utility)
"""

from pathlib import Path
from typing import Dict, Union


def load_checkpoint_state(model_path: Union[str, Path]) -> Dict:
    """Load a `.ckpt` file's state dict, unwrapping known checkpoint shapes.

    Raises:
        FileNotFoundError: ``model_path`` does not exist.
    """
    import torch

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    return state_dict
