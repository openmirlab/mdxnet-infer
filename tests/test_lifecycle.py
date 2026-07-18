from pathlib import Path

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


# ---------------------------------------------------------------------------
# MDXNetSession.is_cached() -- cheap, non-downloading cache-status checks
# ---------------------------------------------------------------------------
#
# is_cached() must mirror load()'s own three-part branch condition exactly
# (`not self._customized and not self._engine_options and not self._metadata`).
# checkpoints.toml already carries a drumsep-6stem entry, so the DEFAULT
# session's _metadata is non-empty and the default case actually resolves
# via the catalog/_materialize path below -- NOT via
# MDX23CInference.is_cached()/KNOWN_MODELS, which is only reached for a
# model name absent from the catalog. Getting this branch wrong would
# silently check the wrong location for the common case.


def test_is_cached_false_before_and_true_after_custom_checkpoint_present(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    session = MDXNetSession(checkpoint_path=checkpoint, progress=False)

    assert session.is_cached() is False

    checkpoint.write_bytes(b"weights")
    assert session.is_cached() is True


def test_is_cached_never_downloads(tmp_path, monkeypatch):
    import mdxnet_infer.clean_api as api

    monkeypatch.setattr(api, "download_file",
                        lambda *a, **k: pytest.fail("network download attempted"))
    metadata = api.get_checkpoint_metadata("drumsep-6stem")
    session = api.MDXNetSession(cache_dir=tmp_path, progress=False)

    assert session.is_cached() is False

    ckpt_path = tmp_path / Path(metadata["checkpoint_url"]).name
    config_path = tmp_path / Path(metadata["config_url"]).name
    ckpt_path.write_bytes(b"fake checkpoint")
    config_path.write_bytes(b"fake config")

    assert session.is_cached() is True


def test_default_session_is_cached_routes_through_catalog_not_known_models(tmp_path, monkeypatch):
    """Proves the default (uncustomized) session's is_cached() takes the
    _target_path/catalog branch, not MDX23CInference.is_cached()/
    KNOWN_MODELS -- both catalogs point at the same URLs today, so a
    plain True/False assertion alone wouldn't distinguish the two branches.
    """
    from pathlib import Path
    import mdxnet_infer.clean_api as api
    import mdxnet_infer.inference as inference

    def fail_is_cached(*_args, **_kwargs):
        pytest.fail(
            "MDX23CInference.is_cached() should not be used for the "
            "default model -- checkpoints.toml already has a "
            "drumsep-6stem entry, so load()'s branch condition routes "
            "the default case through _materialize/_target_path instead."
        )

    monkeypatch.setattr(inference.MDX23CInference, "is_cached",
                        classmethod(fail_is_cached))
    monkeypatch.setattr(api, "download_file",
                        lambda *a, **k: pytest.fail("network download attempted"))

    metadata = api.get_checkpoint_metadata("drumsep-6stem")
    session = api.MDXNetSession(cache_dir=tmp_path, progress=False)

    # False, then True after placing fake artifacts -- and neither call
    # trips the monkeypatched MDX23CInference.is_cached() above.
    assert session.is_cached() is False

    (tmp_path / Path(metadata["checkpoint_url"]).name).write_bytes(b"fake checkpoint")
    (tmp_path / Path(metadata["config_url"]).name).write_bytes(b"fake config")

    assert session.is_cached() is True


def test_cache_info_reports_cached_key(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    session = MDXNetSession(checkpoint_path=checkpoint, progress=False)

    assert session.cache_info()["cached"] is False

    checkpoint.write_bytes(b"weights")
    assert session.cache_info()["cached"] is True
