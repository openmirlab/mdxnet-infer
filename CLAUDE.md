# mdxnet-infer

Inference-only MDX23C TFC-TDF source separation. Ships a package-owned
registry of DrumSep and generic community recipes; no training code and no
bundled weights. `drumsep-5stem` remains a local-weights-only architecture.

## Scope

- `src/mdxnet_infer/model.py` — `TFC_TDF_net`, a verbatim inference-only
  port of ZFTurbo's Music-Source-Separation-Training `mdx23c_tfc_tdf_v3.py`
  (itself based on KUIELab's sdx23 TFC-TDF v3 architecture). Deep module:
  its internal layout is tied to how the pretrained `.ckpt` state dicts are
  keyed, so don't restructure it without re-verifying `load_state_dict`
  against a real checkpoint.
- `src/mdxnet_infer/config/checkpoints.toml` — the sole registry authority:
  stable public names, exact output stems, API family, constructible recipe,
  and full provenance for checkpoint/config artifacts. Do not copy URLs,
  SHA-256 values, or model lists into Python.
- `src/mdxnet_infer/checkpoint_catalog.py` — validates the TOML and exposes
  both rich recipe metadata and old flattened checkpoint/config views.
- `src/mdxnet_infer/config.py` — `MDX23CConfig` dataclass tree (audio/model/
  training/inference), loadable from the YAML shipped alongside each
  checkpoint, plus two hard-coded presets (`drumsep_6stem`, `drumsep_5stem`)
  matching the two known checkpoints' actual training configs. `drumsep_5stem`
  is kept even though its checkpoint is unavailable (see below), in case a
  user supplies their own weights or a mirror surfaces.
- `src/mdxnet_infer/inference.py` — `MDX23CInference` (load + chunked
  overlap-add separation), generic `separate_file()`, and DrumSep-only
  `separate_drums()`. `KNOWN_MODELS` is a compatibility view derived from
  the TOML; it must never become a second registry. Generic recipes must not
  acquire cymbal or other DrumSep post-processing semantics.
- `src/mdxnet_infer/cli.py` — argparse entry point (`mdxnet-infer` console
  script), dynamic stable-name selection, and generic file routing.
- `src/mdxnet_infer/utils/` — `download.py` (streamed HTTP download),
  `cache.py` (cache dir resolution, env-overridable), `stems.py` (post-hoc
  stem combination: ride+crash -> cymbals, etc.), `checkpoint.py`
  (`load_checkpoint_state()` — the one place a `.ckpt`'s
  `state_dict`/`model_state_dict` wrapper gets unwrapped; both the Torch
  path and the MLX weight converter call it, so that convention lives in
  exactly one module).
- `src/mdxnet_infer/backends/` — the compute seam. `base.py` holds the
  `SeparationBackend` protocol (one mixture in, named stems out, via a
  signature that mirrors `MDX23CInference.separate()`'s own — see its
  module docstring for why this deliberately isn't the narrower
  `(channels, samples)` seam a sibling OpenMIRLab package uses) and
  `ChunkingPlan`; `torch_backend.py` wraps the existing
  `MDX23CInference.separate()` without reforking it; `mlx_backend.py` is the
  new MLX implementation, reimplementing (not reusing) that same chunking
  arithmetic in MLX. `__init__.py` resolves a backend by name. Every registry
  checkpoint shares one architecture, so unlike a sibling package's backend
  seam there is no per-checkpoint "variation" concept here to support or
  refuse. Backend modules import lazily, so `import mdxnet_infer` never
  pulls in `mlx` — `tests/test_backends.py` asserts that (via a subprocess,
  since an *earlier* test in the same process legitimately importing real
  mlx as a side effect of a real `is_available()` check is not the same
  thing as `import mdxnet_infer` doing it).
- `src/mdxnet_infer/mlx/` — the vendored MLX TFC_TDF_v3 (MIT, from
  `ssmall256/mlx-audio-separator`, source revision recorded in the file
  header), imported only by `backends/mlx_backend.py`. `convert.py`'s
  `load_converted_weights()` raises rather than loading partially: the
  naive `model.load_weights(strict=False)` path it replaces silently drops
  unmatched keys, which leaves layers at random initialisation and produces
  confident garbage. `model.py` carries one deliberate deviation from
  upstream, `exact_zero_safe_rfft()` — read its docstring before touching
  it. Unlike the sibling package this pattern comes from, it is **measured
  inert** here (no operation in this architecture discards-and-renormalizes
  a frame's own magnitude, and there is no attention to spread a corrupted
  frame across time positions) — kept for consistency and cheap insurance,
  not because removing it changes measured parity.
- `tests/` — import smoke tests + model/config/inference unit tests, all
  offline (no network, no real checkpoint needed — instantiates
  `TFC_TDF_net` with random weights and forward-passes synthetic tensors).
  `test_backends.py` and `test_mlx_model.py` add the same offline guarantee
  for the backend seam and the vendored MLX model (the latter skips cleanly
  without the `[mlx]` extra). `test_mlx_parity.py` is real-checkpoint,
  `realweights`-marked, deselected by default.

## Accuracy rule

`model.py`'s forward pass must stay byte-for-byte identical to upstream's
`mdx23c_tfc_tdf_v3.py` — it is a verbatim port, not a reimplementation.
Any change to `model.py`'s math requires a before/after golden-fixture
comparison (record fixture on current code first, then prove bit-identical
after). No such fixture exists yet in this repo — none of the changes to
date have touched model.py's numerics. The same discipline applies to
`mlx/model.py`'s port of it: `tests/test_mlx_parity.py` is the golden
fixture, real-checkpoint and `realweights`-marked.

## MLX backend (Apple Silicon)

Optional, additive `backend="torch"` (default, unchanged) / `"mlx"` /
`"auto"` axis; `device` keeps its Torch meaning; the MLX backend accepts only `None`/`auto`/`mps` for it and raises for anything else rather than ignoring it. See
README's "Backends and devices" section for the public contract and
`src/mdxnet_infer/backends/` above for the seam. Measured Torch-vs-MLX
parity on the real `drumsep-6stem` checkpoint, through the public
`.separate()` API, worst-case max-abs divergence (2026-07-30, Apple
M-series, torch 2.13.0, mlx 0.31.2): clean signal 1.371e-06, zero-padded
tail 1.445e-06, near-silent tail 1.028e-06 — all in the same ~1e-6 noise
floor, with and without `exact_zero_safe_rfft()` (see above). `MLX_ENABLE_AMP`
is not read anywhere in this port (there is no mixed-precision path to
disable, unlike the sibling package this pattern is adapted from).

## Verification commands

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -v
python -m build   # packaging check

# MLX backend, on an Apple Silicon Mac with the [mlx] extra installed:
uv pip install -e ".[dev,mlx]"
pytest -m realweights tests/test_mlx_parity.py -v
```

## File-top header convention

Load-bearing files (roughly >150 lines) carry a file-top header: title line,
2-3 sentences of what/why, then a `Reads:` line naming the internal modules
imported. Files verbatim-ported from upstream MIT code additionally carry
`# Copyright (c) ... / SPDX-License-Identifier: MIT` lines above the
docstring (see `model.py`).

## Known, deliberately unfixed issues

- **`drumsep-6stem` weight hosting was dead, now resolved.** The original
  `github.com/jarredou/models` release assets are gone (404, verified via
  `gh api` and direct HTTP HEAD). Two independent third-party HF mirrors of
  the checkpoint were cross-verified byte-identical (matching sha256), and
  both files were re-published as an openmirlab-controlled GitHub Release
  (`weights-drumsep-v1`). `KNOWN_MODELS['drumsep-6stem']` now points there
  and records sha256 digests that `download_model()` verifies.
- **`drumsep-5stem` is lost upstream — not resolved, not coming back
  automatically.** Same dead `jarredou/models` hosting, but no intact
  original-format mirror could be found anywhere on the web (only a
  non-drop-in OpenVINO conversion, `Intel/drumsep_mdx23c_jarredou_openvino`).
  Removed from `KNOWN_MODELS`; `config.py`'s `drumsep_5stem` preset stays for
  anyone with their own checkpoint. Revisit if a verified mirror surfaces.
- **Weights license is undocumented upstream.** No LICENSE or explicit terms
  were ever published by aufr33/jarredou for the DrumSep checkpoints.
  Third-party re-uploads disagree (MIT-tagged HF mirror vs. a
  CC-BY-NC-SA-4.0-tagged derived OpenVINO conversion). Treat as
  non-commercial-safe only until the original authors confirm terms — see
  README's "Weights provenance" section. Not resolved here; this is a
  release-gate blocker, not a doc-fixable gap.
- **Excluded MDX23C candidates are deliberate.** `instvoc-zfturbo` duplicates
  HQ1's full checkpoint SHA; `drumsep-5stem` remains provenance-pending; and
  `mid_side` / `orch` use `target_instrument`, so exposing their one-output
  semantics requires a dedicated public contract. Do not add them merely
  because their architecture loads.
- README's original citation ("Kimberley Jensen et al." / an invented paper
  title) was fabricated and has been corrected to the real arXiv:2305.07489
  citation (Solovyev, Stempkovskiy, Habruseva) as used by upstream MSST.
  Flagging here so a future contributor doesn't wonder why the citation
  changed without a corresponding code change.
