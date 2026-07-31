"""Backend seam contract: resolution, import purity, and honest failure.

These are offline and hardware-independent. They guard the two properties the
seam exists to protect -- that a requested backend is honoured or refused,
never silently swapped, and that the default import path stays free of
optional frameworks.
"""

import pytest

from mdxnet_infer.backends import (
    BACKEND_NAMES,
    DEFAULT_BACKEND,
    BackendUnavailable,
    ChunkingPlan,
    get_backend,
    resolve_backend_name,
)


def test_default_and_none_resolve_to_torch():
    assert resolve_backend_name(None) == DEFAULT_BACKEND == "torch"
    assert resolve_backend_name("torch") == "torch"


def test_auto_resolves_to_a_registered_backend():
    """`auto` is the one place a fallback is what the caller asked for."""
    assert resolve_backend_name("auto") in BACKEND_NAMES


def test_unknown_backend_name_raises_value_error():
    with pytest.raises(ValueError):
        resolve_backend_name("onnx")


def test_unavailable_backend_raises_rather_than_substituting(monkeypatch):
    """An explicit request is honoured or fails loudly -- never downgraded.

    A silent substitution is only ever discovered by noticing the wrong
    hardware was busy.
    """
    from mdxnet_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: False))
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx")


def test_auto_falls_back_to_torch_when_mlx_is_unavailable(monkeypatch):
    from mdxnet_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: False))
    assert resolve_backend_name("auto") == "torch"


def test_auto_prefers_mlx_when_available(monkeypatch):
    from mdxnet_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    assert resolve_backend_name("auto") == "mlx"


def test_torch_backend_satisfies_the_protocol_surface():
    backend = get_backend("torch")
    assert backend.name == "torch"
    assert backend.is_available() is True
    for method in ("separate", "release"):
        assert hasattr(backend, method), f"TorchBackend is missing {method}"


def test_mlx_backend_declares_the_protocol_surface():
    backend = get_backend("mlx")
    assert backend.name == "mlx"
    for method in ("separate", "release", "from_checkpoint", "is_available"):
        assert hasattr(backend, method), f"MLXBackend is missing {method}"


def test_importing_the_package_does_not_pull_in_an_optional_framework():
    """`pip install mdxnet-infer` must stay MLX-free and import-clean.

    Run in a fresh subprocess rather than this test process: on a machine
    that actually has the ``[mlx]`` extra installed, an *earlier* test in
    this same file legitimately imports real ``mlx`` as a side effect of
    calling the real (unmonkeypatched) ``MLXBackend.is_available()`` -- that
    is correct behaviour for `resolve_backend_name("auto")`, not a leak, and
    asserting against this process's already-polluted `sys.modules` would
    make the test's pass/fail depend on execution order instead of on what
    `import mdxnet_infer` itself does.
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import mdxnet_infer\n"
        "optional = {'mlx', 'mlx_spectro', 'mlx_audio_io', 'mlx_audio_separator'}\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules} & optional)\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import mdxnet_infer pulled in optional frameworks: {leaked}"


def test_importing_backends_package_does_not_pull_in_mlx():
    """Even importing the seam itself must not import mlx -- only requesting
    the mlx backend by name should. Subprocess-isolated for the same reason
    as the test above."""
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import mdxnet_infer.backends\n"
        "optional = {'mlx', 'mlx_spectro'}\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules} & optional)\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import mdxnet_infer.backends pulled in: {leaked}"


def test_chunking_plan_matches_inference_module_arithmetic():
    """`ChunkingPlan` must not silently drift from `MDX23CInference.separate()`'s
    own inline chunk-size/hop-size formula -- this test ties the two together so
    a future edit to one is caught by the other."""
    from mdxnet_infer.config import MDX23CConfig

    config = MDX23CConfig.drumsep_6stem()
    plan = ChunkingPlan.from_config(config)

    mdx_segment_size = config.inference.dim_t
    expected_chunk_size = config.audio.hop_length * (mdx_segment_size - 1)
    expected_hop_size = expected_chunk_size // config.inference.num_overlap

    assert plan.chunk_size == expected_chunk_size
    assert plan.hop_size == expected_hop_size
    assert plan.overlap == config.inference.num_overlap


def test_chunking_plan_honours_explicit_overlap_override():
    from mdxnet_infer.config import MDX23CConfig

    config = MDX23CConfig.drumsep_6stem()
    plan = ChunkingPlan.from_config(config, overlap=2)
    assert plan.overlap == 2
    assert plan.hop_size == plan.chunk_size // 2


@pytest.mark.parametrize("device", [None, "auto", "mps"])
def test_mlx_accepts_only_its_own_execution_target(device):
    from mdxnet_infer.backends.mlx_backend import MLXBackend

    assert MLXBackend._select_device(device) == "mps"


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cpu"])
def test_mlx_refuses_a_torch_device_rather_than_ignoring_it(device):
    """Silently ignoring a caller's device discards their stated intent.

    They would only discover it by noticing the wrong hardware was busy -- the
    same failure article 4b forbids for devices generally.
    """
    from mdxnet_infer.backends.base import BackendUnavailable
    from mdxnet_infer.backends.mlx_backend import MLXBackend

    with pytest.raises(BackendUnavailable):
        MLXBackend._select_device(device)
