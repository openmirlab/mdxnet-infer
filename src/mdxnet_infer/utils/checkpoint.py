"""`.ckpt` state-dict unwrapping -- one small, named place for this format's
quirk to live.

`MDX23CInference._load_weights` needs the same raw bytes every time: a
PyTorch checkpoint file that may be a bare state dict or one wrapped under a
``state_dict``/``model_state_dict`` key. This used to be shared with an MLX
weight converter (removed along with MLX/MPS support); it stays its own
function rather than moving back inline, since "unwrap a checkpoint" is a
distinct, independently testable step from "load it into a model."

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
