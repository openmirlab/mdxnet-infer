"""Offline public device and session lifecycle contract tests."""

import numpy as np
import pytest
import torch

from mdxnet_infer.clean_api import MDXNetSession


class _Engine:
    def __init__(self):
        self.model = _Runtime()

    def separate(self, audio, sample_rate=44100, **kwargs):
        return {"kick": audio}


class _Runtime:
    def __init__(self):
        self.released = False

    def cpu(self):
        self.released = True
        return self


def test_session_load_is_idempotent_then_release_rebuilds_and_close_is_terminal(monkeypatch):
    import mdxnet_infer.inference as inference

    built = []

    def build(cls, *args, **kwargs):
        engine = _Engine()
        built.append((kwargs, engine))
        return engine

    monkeypatch.setattr(inference.MDX23CInference, "from_pretrained", classmethod(build))
    session = MDXNetSession(model_name="offline-double", device="cuda:1", progress=False)

    with pytest.raises(RuntimeError, match="call load"):
        session.infer(np.zeros(4, dtype=np.float32))
    assert session.load() is session
    assert session.load() is session
    session.infer(np.zeros(4, dtype=np.float32))
    assert len(built) == 1
    assert built[0][0]["device"] == "cuda:1"

    session.release()
    assert session.status == "released"
    assert built[0][1].model.released
    session.load()
    assert len(built) == 2

    session.close()
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    with pytest.raises(RuntimeError, match="must be ready"):
        session.infer(np.zeros(4, dtype=np.float32))


def test_resolve_device_validates_explicit_requests_and_keeps_cuda_index(monkeypatch):
    import mdxnet_infer.inference as inference

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert inference._resolve_device("cuda:1") == torch.device("cuda:1")
    assert inference._resolve_device("cpu") == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        inference._resolve_device("cuda")
    with pytest.raises(ValueError):
        inference._resolve_device("gpu")
    with pytest.raises(ValueError):
        inference._resolve_device("cuda:-1")


def test_engine_receives_resolved_cuda_index(monkeypatch):
    import mdxnet_infer.inference as inference
    from mdxnet_infer.config import MDX23CConfig

    received = []

    class Model:
        def __init__(self, config, device):
            received.append(device)

        def to(self, device):
            received.append(device)
            return self

        def eval(self):
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(inference, "TFC_TDF_net", Model)
    engine = inference.MDX23CInference(config=MDX23CConfig.drumsep_6stem(), device="cuda:1")

    assert engine.device == torch.device("cuda:1")
    assert received == [torch.device("cuda:1"), torch.device("cuda:1")]


def test_auto_falls_back_to_cpu_without_accelerators(monkeypatch):
    import mdxnet_infer.inference as inference

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert inference._resolve_device("auto") == torch.device("cpu")


def test_known_models_download_metadata_is_derived_from_toml_catalog():
    from mdxnet_infer.checkpoint_catalog import get_checkpoint_metadata
    from mdxnet_infer.inference import MDX23CInference

    entry = get_checkpoint_metadata("drumsep-6stem")
    known = MDX23CInference.KNOWN_MODELS["drumsep-6stem"]
    assert known["ckpt_url"] == entry["checkpoint_url"]
    assert known["yaml_url"] == entry["config_url"]
    assert known["ckpt_sha256"] == entry["checkpoint_sha256"]
    assert known["yaml_sha256"] == entry["config_sha256"]
