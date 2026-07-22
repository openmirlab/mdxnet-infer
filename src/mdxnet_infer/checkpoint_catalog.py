"""Package-owned, release-pinned MDX23C recipe metadata.

``config/checkpoints.toml`` is the sole source of truth for stable model
names, architecture recipes, stems, and remote artifacts.  The flattened
checkpoint/config fields remain compatibility views for callers from the
single-DrumSep registry era.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


_ARTIFACT_KINDS = {"checkpoint", "config"}
_API_FAMILIES = {"drumsep", "generic"}


def checkpoint_config_path() -> Path:
    return Path(__file__).with_name("config") / "checkpoints.toml"


def _load(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid checkpoint config: {path}") from exc


def _required_text(value, field: str, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid {field} for {key}")
    return value


def _sha256(value, kind: str, key: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid {kind} SHA-256 for {key}")
    if any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"invalid {kind} SHA-256 for {key}")
    return value.lower()


def _provenance(record: dict, model: dict, defaults: dict, key: str) -> dict:
    """Resolve per-artifact provenance without giving artifacts a second home."""
    return {
        field: _required_text(
            record.get(field, model.get(field, defaults.get(field, ""))),
            field.replace("_", " "),
            key,
        )
        for field in ("license", "provenance", "source_revision", "updated_at")
    }


def _parse(path=None) -> dict:
    path = Path(path) if path else checkpoint_config_path()
    config = _load(path)
    if config.get("schema", {}).get("version") != 2:
        raise ValueError("unsupported checkpoint config schema version")
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("checkpoint config must define models")

    defaults = config.get("metadata", {})
    entries = {}
    known_names = set()
    for key, model in models.items():
        if not isinstance(model, dict):
            raise ValueError(f"invalid metadata for {key}")
        name = _required_text(model.get("model", key), "model name", key)
        aliases = model.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias for alias in aliases):
            raise ValueError(f"invalid aliases for {key}")
        all_names = [name, *aliases]
        if len(set(all_names)) != len(all_names) or any(item in known_names for item in all_names):
            raise ValueError(f"duplicate model name or alias for {key}")
        known_names.update(all_names)

        stems = model.get("stems")
        if not isinstance(stems, list) or not stems or not all(isinstance(stem, str) and stem for stem in stems):
            raise ValueError(f"invalid stems for {key}")
        target_instrument = model.get("target_instrument") or None
        if target_instrument is not None and target_instrument not in stems:
            raise ValueError(f"target instrument must be one of stems for {key}")
        api_family = model.get("api_family")
        if api_family not in _API_FAMILIES:
            raise ValueError(f"invalid API family for {key}")
        recipe = model.get("recipe")
        if not isinstance(recipe, dict) or not all(section in recipe for section in ("audio", "model", "inference")):
            raise ValueError(f"model {key!r} must define an architecture recipe")
        if not all(isinstance(recipe[section], dict) for section in ("audio", "model", "inference")):
            raise ValueError(f"invalid architecture recipe for {key}")

        files = model.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"model {key!r} must define files")
        artifacts = []
        by_kind = {}
        for artifact in files:
            if not isinstance(artifact, dict):
                raise ValueError(f"invalid artifact for {key}")
            kind = artifact.get("kind")
            if kind not in _ARTIFACT_KINDS or kind in by_kind:
                raise ValueError(f"invalid artifact kind for {key}")
            url = artifact.get("url")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise ValueError(f"invalid {kind} URL for {key}")
            digest = _sha256(artifact.get("sha256"), kind, key)
            name_for_file = artifact.get("name") or Path(url).name
            _required_text(name_for_file, f"{kind} name", key)
            item = {
                "kind": kind,
                "name": name_for_file,
                "url": url,
                "sha256": digest,
                **_provenance(artifact, model, defaults, key),
            }
            artifacts.append(item)
            by_kind[kind] = item
        if set(by_kind) != _ARTIFACT_KINDS:
            raise ValueError(f"model {key!r} requires checkpoint and config artifacts")

        entry = {
            "model": name,
            "aliases": list(aliases),
            "stems": list(stems),
            "target_instrument": target_instrument,
            "api_family": api_family,
            "recipe": recipe,
            # A small offline view for callers choosing buffers/devices.  The
            # full constructible recipe remains under ``recipe``.
            "audio": dict(recipe["audio"]),
            "checkpoint_url": by_kind["checkpoint"]["url"],
            "checkpoint_sha256": by_kind["checkpoint"]["sha256"],
            "config_url": by_kind["config"]["url"],
            "config_sha256": by_kind["config"]["sha256"],
            "artifacts": artifacts,
            **_provenance({}, model, defaults, key),
        }
        entries[key] = entry
    return {"raw": config, "entries": entries}


_CONFIG = _parse()
_ENTRIES = _CONFIG["entries"]
CHECKPOINT_CATALOG = MappingProxyType({key: MappingProxyType(value) for key, value in _ENTRIES.items()})


def get_checkpoint_metadata(model_name: str) -> dict | None:
    """Return a copy of metadata for a stable name or compatibility alias."""
    for entry in _ENTRIES.values():
        if model_name == entry["model"] or model_name in entry["aliases"]:
            return dict(entry)
    return None


def checkpoint_catalog() -> dict:
    """Return compatibility metadata keyed by the registry's internal key."""
    return {key: dict(value) for key, value in _ENTRIES.items()}


def list_model_names(*, api_family: str | None = None) -> tuple[str, ...]:
    """List stable public names without downloading any artifact."""
    if api_family is not None and api_family not in _API_FAMILIES:
        raise ValueError(f"unknown API family: {api_family!r}")
    return tuple(
        entry["model"] for entry in _ENTRIES.values()
        if api_family is None or entry["api_family"] == api_family
    )


def validate_checkpoint_config(path=None) -> dict:
    return _parse(path)["raw"]


__all__ = [
    "CHECKPOINT_CATALOG",
    "checkpoint_catalog",
    "checkpoint_config_path",
    "get_checkpoint_metadata",
    "list_model_names",
    "validate_checkpoint_config",
]
