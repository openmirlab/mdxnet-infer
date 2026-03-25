"""Stem post-processing utilities."""

from typing import Dict, List, Union
import numpy as np


def combine_cymbal_stems(
    stems: Dict[str, Union[np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Combine ride and crash into a single cymbals stem.

    Removes ``'ride'`` and ``'crash'`` from *stems* (in place) and adds
    ``'cymbals'`` = ride + crash.  If either key is absent the dict is
    returned unchanged.

    Args:
        stems: Mapping from stem name to audio array.  Modified in place.

    Returns:
        The same dict with ride/crash replaced by cymbals.
    """
    if "ride" not in stems or "crash" not in stems:
        return stems

    ride = stems.pop("ride")
    crash = stems.pop("crash")
    min_len = min(len(ride), len(crash))
    stems["cymbals"] = ride[:min_len] + crash[:min_len]
    return stems


def combine_others_stems(
    stems: Dict[str, Union[np.ndarray]],
    component_names: List[str] = None,
) -> Dict[str, np.ndarray]:
    """Combine hi-hat, cymbals, and toms into an ``'others'`` stem.

    Skips if ``'others'`` already exists.  Only combines components that are
    present in *stems*.

    Args:
        stems: Mapping from stem name to audio array.  Modified in place.
        component_names: List of stem names to combine.
            Defaults to ``['hihat', 'cymbals', 'toms']``.

    Returns:
        The same dict with ``'others'`` added.
    """
    if "others" in stems:
        return stems

    if component_names is None:
        component_names = ["hihat", "cymbals", "toms"]

    components = [stems[name] for name in component_names if name in stems]
    if not components:
        return stems

    # Handle both numpy and torch tensors
    try:
        import torch
        if isinstance(components[0], torch.Tensor):
            stems["others"] = torch.stack(components).sum(dim=0)
            return stems
    except ImportError:
        pass

    stems["others"] = sum(components)
    return stems
