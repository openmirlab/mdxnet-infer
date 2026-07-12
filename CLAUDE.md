# mdxnet-infer

Inference-only MDX23C TFC-TDF drum stem separation. Ships one community
DrumSep checkpoint (6-stem) by aufr33/jarredou, org-hosted. No training code.
The 5-stem checkpoint is lost upstream (see below).

## Scope

- `src/mdxnet_infer/model.py` — `TFC_TDF_net`, a verbatim inference-only
  port of ZFTurbo's Music-Source-Separation-Training `mdx23c_tfc_tdf_v3.py`
  (itself based on KUIELab's sdx23 TFC-TDF v3 architecture). Deep module:
  its internal layout is tied to how the pretrained `.ckpt` state dicts are
  keyed, so don't restructure it without re-verifying `load_state_dict`
  against a real checkpoint.
- `src/mdxnet_infer/config.py` — `MDX23CConfig` dataclass tree (audio/model/
  training/inference), loadable from the YAML shipped alongside each
  checkpoint, plus two hard-coded presets (`drumsep_6stem`, `drumsep_5stem`)
  matching the two known checkpoints' actual training configs. `drumsep_5stem`
  is kept even though its checkpoint is unavailable (see below), in case a
  user supplies their own weights or a mirror surfaces.
- `src/mdxnet_infer/inference.py` — `MDX23CInference` (load + chunked
  overlap-add separation) and `separate_drums()` (file-in/files-out
  convenience wrapper used by the CLI and the top-level `separate` alias).
  `KNOWN_MODELS` hard-codes `drumsep-6stem`'s org-hosted GitHub Release
  download URLs and sha256 digests (`download_model()` verifies both fresh
  downloads and cached files against them). `drumsep-5stem` is not in
  `KNOWN_MODELS` — see "Known, deliberately unfixed issues" below.
- `src/mdxnet_infer/cli.py` — argparse entry point (`mdxnet-infer` console
  script), thin pass-through to `separate_drums()`.
- `src/mdxnet_infer/utils/` — `download.py` (streamed HTTP download),
  `cache.py` (cache dir resolution, env-overridable), `stems.py` (post-hoc
  stem combination: ride+crash -> cymbals, etc.).
- `tests/` — import smoke tests + model/config/inference unit tests, all
  offline (no network, no real checkpoint needed — instantiates
  `TFC_TDF_net` with random weights and forward-passes synthetic tensors).

## Accuracy rule

`model.py`'s forward pass must stay byte-for-byte identical to upstream's
`mdx23c_tfc_tdf_v3.py` — it is a verbatim port, not a reimplementation.
Any change to `model.py`'s math requires a before/after golden-fixture
comparison (record fixture on current code first, then prove bit-identical
after). No such fixture exists yet in this repo — none of the changes to
date have touched model.py's numerics.

## Verification commands

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -v
python -m build   # packaging check
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
- README's original citation ("Kimberley Jensen et al." / an invented paper
  title) was fabricated and has been corrected to the real arXiv:2305.07489
  citation (Solovyev, Stempkovskiy, Habruseva) as used by upstream MSST.
  Flagging here so a future contributor doesn't wonder why the citation
  changed without a corresponding code change.
