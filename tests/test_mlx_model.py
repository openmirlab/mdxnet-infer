"""Offline MLX model tests: weight-conversion audit, wiring, and the
zero-safe rfft guard's own mechanics.

Skipped whole-file when the ``[mlx]`` extra is not installed. These do not
need a real checkpoint -- they build a tiny synthetic config so the model
graph is small and fast, and audit the conversion/load-gate logic against a
*real* (if randomly initialized) Torch state dict, which is the strongest
check obtainable without downloading weights. Real-checkpoint Torch-vs-MLX
numeric parity, including the silence fixtures, lives in
``test_mlx_parity.py`` (``realweights``, deselected by default).
"""

import pytest

mlx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_spectro")


def _tiny_config():
    """A tiny synthetic MDX23CConfig -- small enough to construct and
    forward-pass in milliseconds, but with every field TfcTdfV3MLX reads."""
    from mdxnet_infer.config import MDX23CConfig

    return MDX23CConfig.from_mapping({
        "audio": {
            "chunk_size": 2 * (8 - 1) * 4,
            "dim_f": 16,
            "dim_t": 8,
            "hop_length": 4,
            "n_fft": 32,
            "num_channels": 2,
            "sample_rate": 44100,
        },
        "model": {
            "act": "gelu",
            "bottleneck_factor": 2,
            "growth": 4,
            "norm": "InstanceNorm",
            "num_blocks_per_scale": 1,
            "num_channels": 4,
            "num_scales": 2,
            "num_subbands": 2,
            "scale": [2, 2],
        },
        "training": {
            "instruments": ["kick", "snare"],
            "target_instrument": None,
        },
        "inference": {
            "batch_size": 1,
            "dim_t": 8,
            "num_overlap": 2,
        },
    })


def _build_torch_model(config):
    import torch

    from mdxnet_infer.model import TFC_TDF_net

    torch.manual_seed(0)
    return TFC_TDF_net(config, device=torch.device("cpu"))


def test_convert_and_load_accepts_a_real_random_torch_state_dict():
    """The conversion + strict-load audit must succeed on an unmodified,
    randomly initialized state dict from the real torch model -- this is the
    structural check that catches a key-mapping bug without needing weights."""
    from mdxnet_infer.mlx import TfcTdfV3MLX, convert_torch_to_mlx_weights, load_converted_weights
    from mdxnet_infer.backends.mlx_backend import _config_to_mapping

    config = _tiny_config()
    torch_model = _build_torch_model(config)
    mlx_model = TfcTdfV3MLX(config=_config_to_mapping(config))

    weights = convert_torch_to_mlx_weights(torch_model.state_dict())
    load_converted_weights(mlx_model, weights)  # must not raise


def test_load_converted_weights_raises_on_a_dropped_model_parameter():
    """Deleting a converted tensor must surface as a raised error naming the
    shortfall, not a silent partial load (this is the bug `strict=False`
    alone would hide)."""
    from mdxnet_infer.mlx import TfcTdfV3MLX, convert_torch_to_mlx_weights, load_converted_weights
    from mdxnet_infer.backends.mlx_backend import _config_to_mapping

    config = _tiny_config()
    torch_model = _build_torch_model(config)
    mlx_model = TfcTdfV3MLX(config=_config_to_mapping(config))
    weights = convert_torch_to_mlx_weights(torch_model.state_dict())

    victim_key = next(iter(weights))
    del weights[victim_key]

    with pytest.raises(ValueError, match="unmatched"):
        load_converted_weights(mlx_model, weights)


def test_load_converted_weights_raises_on_an_unconsumed_converted_tensor():
    from mdxnet_infer.mlx import TfcTdfV3MLX, convert_torch_to_mlx_weights, load_converted_weights
    from mdxnet_infer.backends.mlx_backend import _config_to_mapping

    config = _tiny_config()
    torch_model = _build_torch_model(config)
    mlx_model = TfcTdfV3MLX(config=_config_to_mapping(config))
    weights = convert_torch_to_mlx_weights(torch_model.state_dict())

    weights["bogus.unmapped.key"] = mlx.array([1.0])

    with pytest.raises(ValueError, match="dropped"):
        load_converted_weights(mlx_model, weights)


def test_mlx_forward_shape_matches_torch_forward_shape():
    """Wiring sanity: with the *same* converted weights, the MLX model's
    output shape (stems, channels, samples) matches the Torch model's for
    the same input length. This does not check numeric parity (that needs
    real, trained weights -- see test_mlx_parity.py); it catches a shape or
    axis-order bug in the port itself."""
    import numpy as np
    import torch

    from mdxnet_infer.mlx import TfcTdfV3MLX, convert_torch_to_mlx_weights, load_converted_weights
    from mdxnet_infer.backends.mlx_backend import _config_to_mapping

    config = _tiny_config()
    torch_model = _build_torch_model(config)
    torch_model.eval()
    mlx_model = TfcTdfV3MLX(config=_config_to_mapping(config))
    load_converted_weights(mlx_model, convert_torch_to_mlx_weights(torch_model.state_dict()))
    mlx_model.eval()

    chunk_size = config.audio.hop_length * (config.inference.dim_t - 1)
    rng = np.random.default_rng(0)
    audio_np = (rng.standard_normal((1, 2, chunk_size)) * 0.1).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(audio_np)).numpy()
    mlx_out = np.array(mlx_model(mlx.array(audio_np)))

    assert torch_out.shape == mlx_out.shape


def test_exact_zero_safe_rfft_swaps_and_restores_mx_fft_rfft():
    """The guard must actually change which implementation runs while active,
    and put the original back afterwards -- a no-op wrapper protects nothing,
    and a wrapper that fails to restore would corrupt every later call."""
    from mdxnet_infer.mlx.model import exact_zero_safe_rfft

    original = mlx.fft.rfft
    with exact_zero_safe_rfft():
        assert mlx.fft.rfft is not original
        frame = mlx.zeros((4, 8))
        result = mlx.fft.rfft(frame, axis=-1)
        mlx.eval(result)
    assert mlx.fft.rfft is original
