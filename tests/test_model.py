"""
Unit tests for MDX23C model architecture and configuration.

These tests run without downloading any model weights — they only verify
that the model can be instantiated from config and forward-passes synthetic
tensors correctly.
"""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestMDX23CConfig:
    def test_default_instantiation(self):
        from mdxnet_infer.config import MDX23CConfig
        cfg = MDX23CConfig()
        assert cfg.audio.sample_rate == 44100
        assert cfg.audio.n_fft == 2048
        assert cfg.model.num_subbands == 4

    def test_drumsep_6stem(self):
        from mdxnet_infer.config import MDX23CConfig
        cfg = MDX23CConfig.drumsep_6stem()
        assert cfg.training.instruments == ["kick", "snare", "toms", "hh", "ride", "crash"]
        assert cfg.training.target_instrument is None
        assert cfg.inference.dim_t == 256

    def test_drumsep_5stem(self):
        from mdxnet_infer.config import MDX23CConfig
        cfg = MDX23CConfig.drumsep_5stem()
        assert cfg.training.instruments == ["kick", "snare", "toms", "hh", "cymbals"]
        assert cfg.inference.batch_size == 2

    def test_from_yaml(self, tmp_path):
        """Config can be loaded from a minimal YAML file."""
        import yaml
        from mdxnet_infer.config import MDX23CConfig

        yaml_content = {
            "audio": {"sample_rate": 44100, "n_fft": 2048, "hop_length": 512,
                      "dim_f": 1024, "dim_t": 256, "num_channels": 2},
            "model": {"num_subbands": 4, "num_channels": 128, "num_scales": 5,
                      "growth": 128, "bottleneck_factor": 4, "num_blocks_per_scale": 2,
                      "scale": [2, 2], "act": "gelu", "norm": "InstanceNorm"},
            "training": {"instruments": ["kick", "snare"], "target_instrument": None},
            "inference": {"batch_size": 1, "dim_t": 256, "num_overlap": 4},
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        cfg = MDX23CConfig.from_yaml(yaml_path)
        assert cfg.model.scale == (2, 2)
        assert cfg.training.instruments == ["kick", "snare"]

    def test_audio_config_fields(self):
        from mdxnet_infer.config import AudioConfig
        a = AudioConfig(sample_rate=48000, n_fft=4096)
        assert a.sample_rate == 48000
        assert a.n_fft == 4096

    def test_model_config_fields(self):
        from mdxnet_infer.config import ModelConfig
        m = ModelConfig(num_scales=3, growth=64)
        assert m.num_scales == 3
        assert m.growth == 64


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestTFC_TDF_net:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def config_6stem(self):
        from mdxnet_infer.config import MDX23CConfig
        return MDX23CConfig.drumsep_6stem()

    @pytest.fixture
    def config_5stem(self):
        from mdxnet_infer.config import MDX23CConfig
        return MDX23CConfig.drumsep_5stem()

    def test_instantiation_6stem(self, config_6stem, device):
        from mdxnet_infer.model import TFC_TDF_net
        model = TFC_TDF_net(config_6stem, device)
        assert model.num_target_instruments == 6
        assert model.num_subbands == 4

    def test_instantiation_5stem(self, config_5stem, device):
        from mdxnet_infer.model import TFC_TDF_net
        model = TFC_TDF_net(config_5stem, device)
        assert model.num_target_instruments == 5

    def test_has_encoder_decoder(self, config_6stem, device):
        from mdxnet_infer.model import TFC_TDF_net
        model = TFC_TDF_net(config_6stem, device)
        assert len(model.encoder_blocks) == config_6stem.model.num_scales
        assert len(model.decoder_blocks) == config_6stem.model.num_scales
        assert model.bottleneck_block is not None

    def test_forward_6stem(self, config_6stem, device):
        """Model forward pass produces correct output shape for 6 stems."""
        from mdxnet_infer.model import TFC_TDF_net

        model = TFC_TDF_net(config_6stem, device)
        model.eval()

        # Build a minimal batch: (batch=1, channels=2, samples)
        chunk_size = config_6stem.audio.hop_length * (config_6stem.inference.dim_t - 1)
        x = torch.randn(1, 2, chunk_size)

        with torch.no_grad():
            out = model(x)

        # Expected: (batch=1, num_stems=6, channels=2, samples)
        assert out.shape[0] == 1
        assert out.shape[1] == 6
        assert out.shape[2] == 2

    def test_forward_5stem(self, config_5stem, device):
        """Model forward pass produces correct output shape for 5 stems."""
        from mdxnet_infer.model import TFC_TDF_net

        model = TFC_TDF_net(config_5stem, device)
        model.eval()

        chunk_size = config_5stem.audio.hop_length * (config_5stem.inference.dim_t - 1)
        x = torch.randn(1, 2, chunk_size)

        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 1
        assert out.shape[1] == 5
        assert out.shape[2] == 2


# ---------------------------------------------------------------------------
# Inference engine tests (no weights)
# ---------------------------------------------------------------------------

class TestMDX23CInference:
    def test_instantiation_no_weights(self):
        """MDX23CInference can be created without a model path (random weights)."""
        from mdxnet_infer.inference import MDX23CInference
        from mdxnet_infer.config import MDX23CConfig

        engine = MDX23CInference(
            config=MDX23CConfig.drumsep_6stem(),
            device="cpu",
        )
        assert engine.device == torch.device("cpu")
        assert engine.stem_names == ["kick", "snare", "toms", "hh", "ride", "crash"]

    def test_known_models_registry(self):
        from mdxnet_infer.inference import MDX23CInference
        assert "drumsep-6stem" in MDX23CInference.KNOWN_MODELS
        # drumsep-5stem has no surviving original source anywhere on the
        # web (see README) and was removed from the downloadable registry.
        assert "drumsep-5stem" not in MDX23CInference.KNOWN_MODELS

    def test_known_models_have_sha256(self):
        """Every registered model's weight URLs carry a sha256 digest for
        the Weights UX contract's auto-download verification leg."""
        from mdxnet_infer.inference import MDX23CInference

        for name, info in MDX23CInference.KNOWN_MODELS.items():
            assert info.get("ckpt_sha256"), f"{name} missing ckpt_sha256"
            assert info.get("yaml_sha256"), f"{name} missing yaml_sha256"
            assert len(info["ckpt_sha256"]) == 64
            assert len(info["yaml_sha256"]) == 64
            assert info["ckpt_url"].startswith(
                "https://github.com/openmirlab/mdxnet-infer/releases/download/"
            )
            assert info["yaml_url"].startswith(
                "https://github.com/openmirlab/mdxnet-infer/releases/download/"
            )

    def test_stem_names_6stem(self):
        from mdxnet_infer.inference import MDX23CInference
        from mdxnet_infer.config import MDX23CConfig

        engine = MDX23CInference(config=MDX23CConfig.drumsep_6stem(), device="cpu")
        assert engine.stem_names == ["kick", "snare", "toms", "hh", "ride", "crash"]

    def test_stem_names_5stem(self):
        from mdxnet_infer.inference import MDX23CInference
        from mdxnet_infer.config import MDX23CConfig

        engine = MDX23CInference(config=MDX23CConfig.drumsep_5stem(), device="cpu")
        assert engine.stem_names == ["kick", "snare", "toms", "hh", "cymbals"]

    def test_separate_returns_correct_stems(self):
        """separate() returns dict with all expected stem keys and correct shapes."""
        from mdxnet_infer.inference import MDX23CInference
        from mdxnet_infer.config import MDX23CConfig

        config = MDX23CConfig.drumsep_6stem()
        engine = MDX23CInference(config=config, device="cpu")

        # 1 second of stereo silence at 44100 Hz
        audio = np.zeros((44100, 2), dtype=np.float32)

        stems = engine.separate(audio, sample_rate=44100, progress=False)

        assert set(stems.keys()) == {"kick", "snare", "toms", "hh", "ride", "crash"}
        for name, arr in stems.items():
            assert isinstance(arr, np.ndarray), f"Stem {name!r} is not ndarray"
            assert arr.ndim == 2, f"Stem {name!r} should be (samples, 2)"
            assert arr.shape[1] == 2, f"Stem {name!r} should have 2 channels"

    def test_separate_mono_input(self):
        """Mono input is accepted and converted to stereo internally."""
        from mdxnet_infer.inference import MDX23CInference
        from mdxnet_infer.config import MDX23CConfig

        config = MDX23CConfig.drumsep_6stem()
        engine = MDX23CInference(config=config, device="cpu")

        audio = np.zeros(44100, dtype=np.float32)
        stems = engine.separate(audio, sample_rate=44100, progress=False)
        assert len(stems) == 6

    def test_model_name_auto_config(self):
        """Passing model_name selects the correct built-in config."""
        from mdxnet_infer.inference import MDX23CInference

        engine = MDX23CInference(device="cpu", model_name="drumsep-6stem")
        assert engine.stem_names == ["kick", "snare", "toms", "hh", "ride", "crash"]

    def test_unknown_model_name_raises(self):
        """An unregistered model_name (with no explicit config) raises
        clearly instead of silently falling back to the default config."""
        from mdxnet_infer.inference import MDX23CInference

        with pytest.raises(ValueError, match="Unknown model_name"):
            MDX23CInference(device="cpu", model_name="drumsep-5stem")

    def test_unknown_model_name_with_explicit_config_ok(self):
        """An unregistered model_name is fine as long as an explicit config
        is also given (e.g. for a user-supplied 5-stem checkpoint)."""
        from mdxnet_infer.inference import MDX23CInference
        from mdxnet_infer.config import MDX23CConfig

        engine = MDX23CInference(
            device="cpu",
            model_name="drumsep-5stem",
            config=MDX23CConfig.drumsep_5stem(),
        )
        assert engine.stem_names == ["kick", "snare", "toms", "hh", "cymbals"]

    def test_missing_model_file_raises(self, tmp_path):
        from mdxnet_infer.inference import MDX23CInference
        from mdxnet_infer.config import MDX23CConfig

        with pytest.raises(FileNotFoundError):
            MDX23CInference(
                model_path=tmp_path / "nonexistent.ckpt",
                config=MDX23CConfig.drumsep_6stem(),
                device="cpu",
            )

    def test_device_auto_sentinel_resolves(self):
        """The literal string 'auto' triggers the same auto-detect branch
        as device=None, instead of reaching torch.device('auto') and
        raising."""
        from mdxnet_infer.inference import MDX23CInference
        from mdxnet_infer.config import MDX23CConfig

        engine = MDX23CInference(config=MDX23CConfig.drumsep_6stem(), device="auto")
        assert isinstance(engine.device, torch.device)
        # 'auto' must resolve to exactly what device=None auto-detects on
        # this box (cuda/mps/cpu, whichever the machine actually has).
        auto_engine = MDX23CInference(config=MDX23CConfig.drumsep_6stem(), device=None)
        assert engine.device == auto_engine.device


# ---------------------------------------------------------------------------
# MDX23CInference.is_cached() -- cheap, non-downloading cache-status checks
# ---------------------------------------------------------------------------

class TestMDX23CInferenceIsCached:
    """is_cached() must resolve target paths identically to _fetch_verified
    (via the shared _target_path helper) and must never touch the network --
    it exists so callers can ask "is it cached?" for status reporting
    without risking a side-effecting download."""

    def test_false_when_absent(self, tmp_path, monkeypatch):
        from mdxnet_infer.inference import MDX23CInference
        import mdxnet_infer.inference as inference

        monkeypatch.setattr(inference, "download_file",
                            lambda *a, **k: pytest.fail("network download attempted"))
        assert MDX23CInference.is_cached("drumsep-6stem", cache_dir=tmp_path) is False

    def test_true_when_both_files_present_at_fetch_verified_target_paths(self, tmp_path, monkeypatch):
        from pathlib import Path
        from mdxnet_infer.inference import MDX23CInference
        import mdxnet_infer.inference as inference

        monkeypatch.setattr(inference, "download_file",
                            lambda *a, **k: pytest.fail("network download attempted"))
        info = MDX23CInference.KNOWN_MODELS["drumsep-6stem"]
        # Same target-path computation _fetch_verified uses (url.split('/')[-1]).
        (tmp_path / Path(info["ckpt_url"]).name).write_bytes(b"fake checkpoint")
        (tmp_path / Path(info["yaml_url"]).name).write_bytes(b"fake config")

        assert MDX23CInference.is_cached("drumsep-6stem", cache_dir=tmp_path) is True

    def test_false_when_only_checkpoint_present(self, tmp_path):
        from pathlib import Path
        from mdxnet_infer.inference import MDX23CInference

        info = MDX23CInference.KNOWN_MODELS["drumsep-6stem"]
        (tmp_path / Path(info["ckpt_url"]).name).write_bytes(b"fake checkpoint")

        assert MDX23CInference.is_cached("drumsep-6stem", cache_dir=tmp_path) is False

    def test_unknown_model_name_raises(self):
        from mdxnet_infer.inference import MDX23CInference

        with pytest.raises(ValueError, match="Unknown model"):
            MDX23CInference.is_cached("not-a-real-model")

    def test_default_cache_dir_uses_get_cache_dir(self, tmp_path, monkeypatch):
        """No explicit cache_dir falls back to get_cache_dir(), same as
        download_model()."""
        from mdxnet_infer.inference import MDX23CInference

        monkeypatch.setenv("MDXNET_INFER_CACHE_DIR", str(tmp_path))
        assert MDX23CInference.is_cached() is False


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_get_cache_dir_default(self):
        from mdxnet_infer.utils.cache import get_cache_dir
        from pathlib import Path

        d = get_cache_dir()
        assert str(d).endswith("mdxnet-infer")

    def test_get_cache_dir_env(self, monkeypatch, tmp_path):
        from mdxnet_infer.utils.cache import get_cache_dir

        monkeypatch.setenv("MDXNET_INFER_CACHE_DIR", str(tmp_path))
        d = get_cache_dir()
        assert d == tmp_path

    def test_get_cache_dir_subdir(self):
        from mdxnet_infer.utils.cache import get_cache_dir

        d = get_cache_dir("drumsep-6stem")
        assert d.name == "drumsep-6stem"

    def test_sha256sum(self, tmp_path):
        import hashlib
        from mdxnet_infer.utils.download import sha256sum

        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert sha256sum(f) == expected

    @staticmethod
    def _fake_requests_module(content: bytes):
        """Build a stand-in `requests` module whose `.get()` returns a
        fixed byte payload, for exercising download_file()'s sha256 path
        without a real network call."""
        import types

        class _FakeResponse:
            status_code = 200
            headers = {"content-length": str(len(content))}

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield content

        fake_requests = types.ModuleType("requests")
        fake_requests.get = lambda url, stream=True, timeout=300: _FakeResponse()
        return fake_requests

    def test_download_file_sha256_mismatch_raises_and_cleans_up(self, tmp_path, monkeypatch):
        """A checksum mismatch after download raises and removes the bad file."""
        import sys
        from mdxnet_infer.utils import download as download_mod

        monkeypatch.setitem(sys.modules, "requests", self._fake_requests_module(b"wrong"))

        dest = tmp_path / "file.bin"
        with pytest.raises(download_mod.ChecksumMismatchError):
            download_mod.download_file(
                "http://example.invalid/file.bin",
                dest,
                progress=False,
                expected_sha256="0" * 64,
            )
        assert not dest.exists()

    def test_download_file_sha256_match_ok(self, tmp_path, monkeypatch):
        import sys
        import hashlib
        from mdxnet_infer.utils import download as download_mod

        content = b"correct bytes"
        digest = hashlib.sha256(content).hexdigest()

        monkeypatch.setitem(sys.modules, "requests", self._fake_requests_module(content))

        dest = tmp_path / "file.bin"
        download_mod.download_file(
            "http://example.invalid/file.bin",
            dest,
            progress=False,
            expected_sha256=digest,
        )
        assert dest.read_bytes() == content

    def test_combine_cymbal_stems(self):
        from mdxnet_infer.utils.stems import combine_cymbal_stems

        ride = np.ones((100, 2))
        crash = np.ones((100, 2)) * 2
        stems = {"kick": np.zeros((100, 2)), "ride": ride, "crash": crash}

        combine_cymbal_stems(stems)

        assert "ride" not in stems
        assert "crash" not in stems
        assert "cymbals" in stems
        np.testing.assert_array_equal(stems["cymbals"], ride + crash)

    def test_combine_cymbal_stems_missing_key(self):
        from mdxnet_infer.utils.stems import combine_cymbal_stems

        stems = {"kick": np.zeros((100, 2)), "ride": np.ones((100, 2))}
        result = combine_cymbal_stems(stems)
        # Should be unchanged since 'crash' is missing
        assert "cymbals" not in result
        assert "ride" in result

    def test_combine_others_stems(self):
        from mdxnet_infer.utils.stems import combine_others_stems

        stems = {
            "kick": np.zeros((100, 2)),
            "hihat": np.ones((100, 2)),
            "cymbals": np.ones((100, 2)),
            "toms": np.ones((100, 2)),
        }
        combine_others_stems(stems)
        assert "others" in stems
        np.testing.assert_array_almost_equal(stems["others"], np.ones((100, 2)) * 3)


# ---------------------------------------------------------------------------
# CLI registration test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_entry_point_callable(self):
        from mdxnet_infer.cli import main
        assert callable(main)

    def test_cli_help(self):
        """CLI --help exits with code 0."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "mdxnet_infer.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "MDX23C" in result.stdout or "drum" in result.stdout.lower()
