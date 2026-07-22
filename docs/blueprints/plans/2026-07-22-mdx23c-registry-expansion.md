# MDX23C registry expansion — plan

> Generated: 2026-07-22 · Spec source: maintainer implementation brief · Stage 1: fresh

## Context

`src/mdxnet_infer/config/checkpoints.toml` currently has schema version 1 and
describes one downloadable model, `drumsep-6stem`.  Its Python reader,
`checkpoint_catalog.py`, deliberately exposes flattened compatibility fields
(`checkpoint_url`, `config_url`, and their digests), and the inference engine
duplicates that one model in `MDX23CInference.KNOWN_MODELS`.

The model code already has the needed architectural seam: `MDX23CConfig.from_yaml`
filters a recipe YAML down to architecture/inference fields, and `TFC_TDF_net`
uses `target_instrument` to choose either one or all output heads.  The six
accepted recipes all have `target_instrument: null`; their YAML instrument
lists can therefore remain the exact public output names.  `mid_side` and
`orch` do use a target instrument and are excluded until their one-output API
semantics have a dedicated contract.

`MDXNetSeparator` currently routes every file input through `separate_drums`,
and the CLI accepts only DrumSep.  The expansion must add generic MDX23C file
separation without claiming that vocals, dereverb, 4-stem, or SFX models are
part of the DrumSep API.

The intent is a registry-owned, SHA-verified set of technically-probed MDX23C
recipes that keeps the existing DrumSep compatibility surface intact.

## Resolved questions (from the implementation brief)

| Question | User's answer |
|---|---|
| Which recipes are downloadable? | `drumsep-6stem`, `instvoc-hq1`, `instvoc-hq2`, `d1581`, `4stem-zfturbo`, `dereverb-aufr33-jarredou`, and `sfx-jasper`. |
| Which candidates remain out of registry? | `instvoc-zfturbo` (same full checkpoint SHA as HQ1), `drumsep-5stem` (provenance pending), and `mid_side` / `orch` (target semantics pending). |
| What is the public separation boundary? | Generic MDX names use generic APIs; only `drumsep-6stem` belongs to the DrumSep file convenience API. |
| How should SFX behave? | It remains a normal registry entry but downloads lazily into the cache; its 1.34 GB checkpoint is never packaged or preloaded. |

## Approach

1. Upgrade the TOML reader to schema version 2.  Make each model recipe own
   its stable public name, exact stems, target-instrument metadata, API family,
   and a list of checkpoint/config artifacts.  Preserve the existing flattened
   metadata and `CHECKPOINT_CATALOG` mapping as compatibility views.
2. Record the seven accepted recipes using revision-pinned mirror URLs, full
   SHA-256 values, source revision, provenance, and the unresolved model
   license truthfully.  Keep the old DrumSep release artifact as-is; do not add
   model bytes to the repository.
3. Make the inference engine derive `KNOWN_MODELS`, config loading, download,
   cache checks, and stem names from registry metadata.  Reject a named recipe
   with target-instrument semantics unless its public output contract is added
   deliberately; existing generic recipes are all multi-stem.
4. Split generic file separation from `separate_drums`; route facade and CLI
   calls through the generic workflow, retaining cymbal merging exclusively for
   `drumsep-6stem`.  Keep legacy `separate` compatibility while exporting the
   generic one-shot functions.
5. Add unit coverage for registry validation, recipe config/stem resolution,
   D1581's 12k FFT / 2048 hop, SFX lazy cache behavior, and strict checkpoint
   loading plus short synthetic inference using the opt-in probe cache.
6. Refresh user and maintainer documentation, changelog, MIR routing guidance,
   and provenance/liveness tooling references so the TOML registry is the sole
   artifact authority.  State why duplicate, provenance-pending, and
   target-semantics-pending models are excluded.

## Critical files

| File | Why it matters | Touched in step |
|---|---|---|
| `src/mdxnet_infer/config/checkpoints.toml` | Canonical recipe and artifact records | 1, 2 |
| `src/mdxnet_infer/checkpoint_catalog.py` | Validates TOML and exports compatibility views | 1 |
| `src/mdxnet_infer/inference.py` | Resolves named recipes, strict loads, in-memory/file inference | 3, 4 |
| `src/mdxnet_infer/clean_api.py` | Lifecycle and one-shot public facade | 4 |
| `src/mdxnet_infer/cli.py` | Stable-name selection and DrumSep-only cymbal flag | 4 |
| `tests/` | Existing lifecycle/model contracts plus new registry and real-checkpoint tests | 5 |
| `README.md`, `CLAUDE.md`, `CHANGELOG.md` | User and maintainer truth about supported recipes | 6 |

## Single-source-of-truth owners

| Decision (the thing that changes as a unit) | Owner (where it lives) |
|---|---|
| Stable model names, stems, API family, and artifact provenance | `src/mdxnet_infer/config/checkpoints.toml` |
| Backward-compatible flattened metadata views | `src/mdxnet_infer/checkpoint_catalog.py` |

## Verification

1. Registry/parser work → unit test a valid catalog, invalid digests, exact recipe
   stems, and offline catalog resolution.
2. Inference/facade work → run existing lifecycle and clean-API tests plus new
   D1581 and SFX lazy-cache tests.
3. Checkpoint confidence → run strict state-dict load and short synthetic
   inference for every accepted recipe with
   `DEMUCS_MDX23C_PROBE_CACHE=/tmp/mdx23c-probe.Wev81w`.
4. Release gates → `uv run pytest -q`, `uv run ruff check` for touched Python,
   `uv build`, clean wheel/sdist offline import + registry resolution,
   `git diff --check`, and scans for weights, files over 5 MB, and training
   dependencies.

## Out of scope (deferred to other sessions)

- `instvoc-zfturbo`: duplicate checkpoint bytes of HQ1.
- `drumsep-5stem`: checkpoint provenance remains unresolved.
- `mid_side` and `orch`: their single-target output semantics need an explicit
  generic API decision before exposure.
- Re-hosting third-party checkpoint bytes under an OpenMIRLab release.
