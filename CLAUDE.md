# mdxnet-infer

Inference-only MDX23C TFC-TDF drum stem separation. Ships two community
DrumSep checkpoints (5-stem, 6-stem) by aufr33/jarredou. No training code.

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
  matching the two known checkpoints' actual training configs.
- `src/mdxnet_infer/inference.py` — `MDX23CInference` (load + chunked
  overlap-add separation) and `separate_drums()` (file-in/files-out
  convenience wrapper used by the CLI and the top-level `separate` alias).
  `KNOWN_MODELS` hard-codes the two checkpoints' download URLs — see
  "Known, deliberately unfixed issues" below.
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

- **Weight hosting is dead.** `KNOWN_MODELS` in `inference.py` points at
  `github.com/jarredou/models` release assets; that GitHub account/repo no
  longer exists (verified via `gh api` and direct HTTP HEAD — both 404).
  Both `drumsep-6stem` and `drumsep-5stem` currently fail to download. Not
  fixed here: no verified byte-identical replacement source was found
  (third-party HF mirrors exist but weren't checksummed against the
  original release assets), and re-hosting weights under org control is an
  outward-facing action requiring explicit sign-off per org policy.
  Recommendation: either get sign-off to mirror a verified checkpoint under
  openmirlab's HF account, or contact aufr33/jarredou for a current source.
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
