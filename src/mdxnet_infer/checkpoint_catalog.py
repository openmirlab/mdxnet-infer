"""Package-owned, release-pinned checkpoint metadata.

``config/checkpoints.toml`` is the source of truth.  The mapping exports are
kept as compatibility views for callers that used the original Python table.
"""

from pathlib import Path
from types import MappingProxyType
import tomllib


def checkpoint_config_path() -> Path:
    return Path(__file__).with_name("config") / "checkpoints.toml"


def _load(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid checkpoint config: {path}") from exc


def _parse(path=None) -> dict:
    path = Path(path) if path else checkpoint_config_path()
    config = _load(path)
    if config.get("schema", {}).get("version") != 1:
        raise ValueError("unsupported checkpoint config schema version")
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("checkpoint config must define models")
    defaults = config.get("metadata", {})
    entries = {}
    for key, model in models.items():
        if not isinstance(model, dict):
            raise ValueError(f"invalid metadata for {key}")
        files = model.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"model {key!r} must define files")
        artifacts = []
        by_kind = {}
        for artifact in files:
            if not isinstance(artifact, dict):
                raise ValueError(f"invalid artifact for {key}")
            kind, url, digest = artifact.get("kind"), artifact.get("url"), artifact.get("sha256")
            if kind not in {"checkpoint", "config"}:
                raise ValueError(f"invalid artifact kind for {key}")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise ValueError(f"invalid {kind} URL for {key}")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                raise ValueError(f"invalid {kind} SHA-256 for {key}")
            name = artifact.get("name") or Path(url).name
            if not isinstance(name, str) or not name:
                raise ValueError(f"invalid {kind} name for {key}")
            item = {"kind": kind, "name": name, "url": url, "sha256": digest.lower(),
                    "license": artifact.get("license", model.get("license", defaults.get("license", ""))),
                    "provenance": artifact.get("provenance", model.get("provenance", defaults.get("provenance", ""))),
                    "source_revision": artifact.get("source_revision", model.get("source_revision", defaults.get("source_revision", ""))),
                    "updated_at": artifact.get("updated_at", model.get("updated_at", defaults.get("updated_at", "")))}
            artifacts.append(item)
            by_kind[kind] = item
        if "checkpoint" not in by_kind or "config" not in by_kind:
            raise ValueError(f"model {key!r} requires checkpoint and config artifacts")
        entry = {
            "model": model.get("model", key),
            "checkpoint_url": by_kind["checkpoint"]["url"],
            "checkpoint_sha256": by_kind["checkpoint"]["sha256"],
            "config_url": by_kind["config"]["url"],
            "config_sha256": by_kind["config"]["sha256"],
            "artifacts": artifacts,
            "license": model.get("license", defaults.get("license", "")),
            "provenance": model.get("provenance", defaults.get("provenance", "")),
            "source_revision": model.get("source_revision", defaults.get("source_revision", "")),
            "updated_at": model.get("updated_at", defaults.get("updated_at", "")),
        }
        entries[key] = entry
    return {"raw": config, "entries": entries}


_CONFIG = _parse()
_ENTRIES = _CONFIG["entries"]
CHECKPOINT_CATALOG = MappingProxyType({k: MappingProxyType(v) for k, v in _ENTRIES.items()})


def get_checkpoint_metadata(model_name: str) -> dict | None:
    for entry in _ENTRIES.values():
        if model_name in (entry["model"],):
            return dict(entry)
    return None


def checkpoint_catalog() -> dict:
    return {key: dict(value) for key, value in _ENTRIES.items()}


def validate_checkpoint_config(path=None) -> dict:
    return _parse(path)["raw"]


__all__ = ["CHECKPOINT_CATALOG", "checkpoint_catalog", "checkpoint_config_path", "get_checkpoint_metadata", "validate_checkpoint_config"]
