"""Focused parity and boundary checks for the additive clean facade."""

from pathlib import Path

import numpy as np
import pytest

from mdxnet_infer.clean_api import MDXNetSeparator, separate


class FakeEngine:
    def __init__(self):
        self.calls = []

    def separate(self, audio, sample_rate=44100, **kwargs):
        self.calls.append((audio, sample_rate, kwargs))
        return {"kick": audio}


def test_facade_delegates_and_preserves_existing_imports():
    import mdxnet_infer
    from mdxnet_infer.inference import MDX23CInference, separate_drums

    assert mdxnet_infer.MDX23CInference is MDX23CInference
    assert mdxnet_infer.separate is separate_drums
    engine = FakeEngine()
    result = MDXNetSeparator(engine=engine).separate(
        np.ones(8, dtype=np.float64), sample_rate=44100
    )
    assert np.array_equal(result["kick"], np.ones(8, dtype=np.float32))
    assert engine.calls[0][0].dtype == np.float32


def test_in_memory_audio_requires_explicit_positive_rate():
    helper = MDXNetSeparator(engine=FakeEngine())
    with pytest.raises(ValueError, match="sample_rate is required"):
        helper(np.zeros(4))
    with pytest.raises(ValueError, match="must be positive"):
        helper(np.zeros(4), sample_rate=0)


def test_path_rejects_conflicting_rate_and_delegates(monkeypatch, tmp_path):
    calls = []

    def fake_separate_drums(*args, **kwargs):
        calls.append((args, kwargs))
        return {"kick": Path("out.wav")}

    monkeypatch.setattr("mdxnet_infer.inference.separate_drums", fake_separate_drums)
    helper = MDXNetSeparator(model_name="drumsep-6stem", progress=False)
    path = tmp_path / "drums.wav"
    path.touch()
    assert helper(path) == {"kick": Path("out.wav")}
    assert calls[0][0] == (path,)
    with pytest.raises(ValueError, match="not accepted"):
        helper(path, sample_rate=44100)


def test_helper_reuses_engine_and_one_shot_builds_fresh(monkeypatch):
    engines = []

    def build(*args, **kwargs):
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr("mdxnet_infer.clean_api.MDX23CInference", build, raising=False)
    # Inject a fake lazy-compatible module path instead of loading weights.
    import mdxnet_infer.inference as inference
    monkeypatch.setattr(inference.MDX23CInference, "from_pretrained", classmethod(lambda cls, *a, **k: build()))
    helper = MDXNetSeparator()
    helper(np.zeros(4), sample_rate=44100)
    helper(np.zeros(4), sample_rate=44100)
    assert len(engines) == 1
    separate(np.zeros(4), sample_rate=44100, progress=False)
    separate(np.zeros(4), sample_rate=44100, progress=False)
    assert len(engines) == 3


def test_advanced_engine_surface_remains_available():
    from mdxnet_infer.inference import MDX23CInference

    assert callable(MDX23CInference.separate)

