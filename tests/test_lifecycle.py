import numpy as np
import pytest

from mdxnet_infer.clean_api import MDXNetSession


class FakeEngine:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.model = None

    def separate(self, audio, sample_rate=44100, **kwargs):
        self.calls.append((audio, sample_rate, kwargs))
        return {"kick": audio}


def test_session_requires_explicit_load_and_releases(monkeypatch):
    import mdxnet_infer.inference as inference
    monkeypatch.setattr(inference.MDX23CInference, "from_pretrained",
                        classmethod(lambda cls, *a, **k: FakeEngine()))
    session = MDXNetSession(progress=False)
    assert session.status == "new"
    with pytest.raises(RuntimeError, match="call load"):
        session.infer(np.zeros(4, dtype=np.float32))
    assert session.load().status == "ready"
    result = session.infer(np.zeros(4, dtype=np.float32))
    assert "kick" in result
    session.release()
    assert session.status == "released"
    assert session.cache_info()["model_loaded"] is False
    with pytest.raises(RuntimeError):
        session.infer(np.zeros(4, dtype=np.float32))


def test_session_context_manager(monkeypatch):
    import mdxnet_infer.inference as inference
    monkeypatch.setattr(inference.MDX23CInference, "from_pretrained",
                        classmethod(lambda cls, *a, **k: FakeEngine()))
    with MDXNetSession(progress=False) as session:
        assert session.status == "ready"
    assert session.status == "released"


def test_custom_checkpoint_path_is_verified(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    import mdxnet_infer.inference as inference
    monkeypatch.setattr(inference.MDX23CInference, "__init__",
                        lambda self, **kwargs: setattr(self, "model", None))
    session = MDXNetSession(checkpoint_path=checkpoint,
                            checkpoint_metadata={
                                "checkpoint_sha256": __import__("hashlib").sha256(b"weights").hexdigest(),
                            }, progress=False)
    session.load()
    assert session.status == "ready"


def test_custom_checkpoint_hash_mismatch(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    session = MDXNetSession(checkpoint_path=checkpoint,
                            checkpoint_metadata={"checkpoint_sha256": "0" * 64})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        session.load()


def test_default_session_uses_package_checkpoint_catalog(tmp_path, monkeypatch):
    import mdxnet_infer.clean_api as api
    import mdxnet_infer.inference as inference

    metadata = api.get_checkpoint_metadata("drumsep-6stem")
    seen = []

    def fake_download(url, destination, *, progress=True, expected_sha256=None):
        seen.append((url, expected_sha256))
        destination.write_bytes(b"artifact")
        return destination

    monkeypatch.setattr(api, "download_file", fake_download)
    monkeypatch.setattr(
        api,
        "sha256sum",
        lambda path: metadata["checkpoint_sha256"]
        if path.suffix == ".ckpt" else metadata["config_sha256"],
    )

    def fake_init(self, **kwargs):
        self.model = None
        self.kwargs = kwargs

    monkeypatch.setattr(inference.MDX23CInference, "__init__", fake_init)

    session = api.MDXNetSession(cache_dir=tmp_path, progress=False)
    session.load()

    assert seen == [
        (metadata["checkpoint_url"], metadata["checkpoint_sha256"]),
        (metadata["config_url"], metadata["config_sha256"]),
    ]
    assert session._model.kwargs["model_path"].name.endswith(".ckpt")
    assert session._model.kwargs["config_path"].name.endswith(".yaml")
