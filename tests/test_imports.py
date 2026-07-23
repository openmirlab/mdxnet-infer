"""Tests verifying that all public API symbols are importable."""


def test_package_imports():
    """Top-level package exposes all expected names."""
    import mdxnet_infer

    assert hasattr(mdxnet_infer, "TFC_TDF_net")
    assert hasattr(mdxnet_infer, "MDX23CInference")
    assert hasattr(mdxnet_infer, "MDX23CConfig")
    assert hasattr(mdxnet_infer, "separate")
    assert hasattr(mdxnet_infer, "__version__")


def test_model_importable():
    """model.py exports are importable."""
    from mdxnet_infer.model import TFC_TDF_net, STFT, TFC_TDF, Upscale, Downscale

    assert all(obj is not None for obj in (TFC_TDF_net, STFT, TFC_TDF, Upscale, Downscale))


def test_config_importable():
    """config.py exports are importable."""
    from mdxnet_infer.config import (
        MDX23CConfig,
        AudioConfig,
        ModelConfig,
        TrainingConfig,
        InferenceConfig,
    )

    assert all(
        obj is not None
        for obj in (MDX23CConfig, AudioConfig, ModelConfig, TrainingConfig, InferenceConfig)
    )


def test_inference_importable():
    """inference.py exports are importable."""
    from mdxnet_infer.inference import MDX23CInference, separate_drums

    assert MDX23CInference is not None
    assert callable(separate_drums)


def test_cli_importable():
    """cli.py main function is importable."""
    from mdxnet_infer.cli import main
    assert callable(main)


def test_utils_importable():
    """utils subpackage is importable."""
    from mdxnet_infer.utils import get_cache_dir as package_get_cache_dir
    from mdxnet_infer.utils import download_file, combine_cymbal_stems as package_combine_cymbal_stems
    from mdxnet_infer.utils.cache import get_cache_dir as cache_get_cache_dir, clear_cache
    from mdxnet_infer.utils.stems import (
        combine_cymbal_stems as stems_combine_cymbal_stems,
        combine_others_stems,
    )

    assert callable(package_get_cache_dir)
    assert callable(download_file)
    assert callable(package_combine_cymbal_stems)
    assert callable(cache_get_cache_dir)
    assert callable(clear_cache)
    assert callable(stems_combine_cymbal_stems)
    assert callable(combine_others_stems)


def test_separate_alias():
    """'separate' in top-level is the same as separate_drums."""
    from mdxnet_infer import separate
    from mdxnet_infer.inference import separate_drums

    assert separate is separate_drums
