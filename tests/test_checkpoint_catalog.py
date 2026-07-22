"""Offline contracts for the MDX23C checkpoint registry.

The registry describes downloadable artifacts, but reading it must remain a
pure local operation: listing a large model such as SFX must not materialize
or fetch its checkpoint.
"""

import pytest


STABLE_MODELS = {
    "drumsep-6stem",
    "instvoc-hq1",
    "instvoc-hq2",
    "d1581",
    "4stem-zfturbo",
    "dereverb-aufr33-jarredou",
    "sfx-jasper",
}


def test_registry_exposes_the_supported_stable_model_names():
    from mdxnet_infer.checkpoint_catalog import checkpoint_catalog

    entries = checkpoint_catalog()
    assert {entry["model"] for entry in entries.values()} == STABLE_MODELS


def test_registry_entries_have_complete_offline_provenance():
    from mdxnet_infer.checkpoint_catalog import checkpoint_catalog

    for entry in checkpoint_catalog().values():
        assert entry["model"] in STABLE_MODELS
        assert isinstance(entry["aliases"], list)
        assert entry["stems"]
        assert entry["target_instrument"] is None
        assert entry["api_family"] in {"drumsep", "generic"}
        assert entry["license"]
        assert entry["provenance"]
        assert entry["source_revision"]
        assert entry["updated_at"]
        assert len(entry["checkpoint_sha256"]) == 64
        assert len(entry["config_sha256"]) == 64

        artifacts = entry["artifacts"]
        assert {artifact["kind"] for artifact in artifacts} == {"checkpoint", "config"}
        for artifact in artifacts:
            assert artifact["url"].startswith("https://")
            assert len(artifact["sha256"]) == 64
            assert artifact["license"]
            assert artifact["provenance"]
            assert artifact["source_revision"]
            assert artifact["updated_at"]


def test_d1581_preserves_recipe_stem_case_and_stft_parameters():
    from mdxnet_infer.checkpoint_catalog import get_checkpoint_metadata

    d1581 = get_checkpoint_metadata("d1581")
    assert d1581["stems"] == ["Vocals", "Instrumental"]
    assert d1581["audio"]["n_fft"] == 12288
    assert d1581["audio"]["hop_length"] == 2048


def test_sfx_registry_access_is_offline_and_does_not_create_a_cache(tmp_path, monkeypatch):
    """SFX is catalogued, yet its 1.34 GB checkpoint stays fully lazy."""
    import mdxnet_infer.inference as inference
    from mdxnet_infer.checkpoint_catalog import (
        checkpoint_catalog,
        get_checkpoint_metadata,
        list_model_names,
    )
    from mdxnet_infer.inference import MDX23CInference

    def no_download(*args, **kwargs):
        pytest.fail("registry inspection attempted a download")

    monkeypatch.setattr(inference, "download_file", no_download)
    assert get_checkpoint_metadata("sfx-jasper")["model"] == "sfx-jasper"
    assert "sfx-jasper" in list_model_names(api_family="generic")
    assert "sfx-jasper" in {entry["model"] for entry in checkpoint_catalog().values()}
    assert MDX23CInference.is_cached("sfx-jasper", cache_dir=tmp_path) is False
    assert list(tmp_path.iterdir()) == []


def test_flattened_compatibility_artifacts_match_their_registry_records():
    from mdxnet_infer.checkpoint_catalog import checkpoint_catalog

    for entry in checkpoint_catalog().values():
        by_kind = {artifact["kind"]: artifact for artifact in entry["artifacts"]}
        assert entry["checkpoint_url"] == by_kind["checkpoint"]["url"]
        assert entry["checkpoint_sha256"] == by_kind["checkpoint"]["sha256"]
        assert entry["config_url"] == by_kind["config"]["url"]
        assert entry["config_sha256"] == by_kind["config"]["sha256"]
