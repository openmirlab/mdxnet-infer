# Changelog

All notable changes to mdxnet-infer are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Expanded the package-owned MDX23C registry from DrumSep alone to seven
  SHA-verified recipes: InstVoc HQ1/HQ2, D1581, ZFTurbo 4-stem, aufr33/jarredou
  dereverb, and Jasper SFX alongside DrumSep. The TOML registry is now the
  single authority for model recipes, exact stems, pinned artifact URLs,
  full SHA-256 digests, provenance, source revisions, and license caveats.
- Added generic `separate_file()` routing for non-DrumSep models. DrumSep-only
  cymbal combining remains isolated in `separate_drums()`.

### Changed
- `MDX23CInference`, `MDXNetSession`, `MDXNetSeparator`, cache inspection,
  and CLI choices now resolve stable model names from the packaged registry;
  flattened checkpoint/config metadata remains as a compatibility view.
- Added explicit exclusions: `instvoc-zfturbo` duplicates HQ1 bytes,
  `drumsep-5stem` has unresolved provenance, and `mid_side` / `orch` need a
  target-instrument output contract.

### Changed
- Validate explicit inference devices early (`cpu`, `cuda`, `cuda:N`, and
  `mps`) while preserving legacy `None`/`auto` accelerator selection and CUDA
  index forwarding.
- `MDXNetSession.release()` now permits reload; idempotent `close()` is
  terminal. The legacy `KNOWN_MODELS` download compatibility view now derives
  URL and hash metadata from packaged `config/checkpoints.toml`.
- Add the conditional `tomli` dependency used to parse checkpoint TOML on
  Python 3.10.

### Fixed
- **Weights re-hosted under org control (weights-drumsep-v1 release).** The
  `drumsep-6stem` checkpoint/config's original hosting
  (`github.com/jarredou/models`) had vanished; two independent third-party
  Hugging Face mirrors were cross-verified byte-identical (matching
  sha256), and both files were re-published as an openmirlab-controlled
  GitHub Release. `KNOWN_MODELS['drumsep-6stem']` now points at
  `github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/`
  and records sha256 digests for both files.
- **Auto-download is now sha256-verified** (org constitution art.4 Weights
  UX contract): `utils/download.py`'s `download_file()` accepts an
  `expected_sha256` and raises `ChecksumMismatchError` (deleting the bad
  file) on mismatch; `MDX23CInference.download_model()` verifies both
  fresh downloads and already-cached files before reuse.
- **CLI `--cache-dir` was silently ignored.** `cli.py` parsed the flag but
  never passed it to `separate_drums()`; `separate_drums()` now accepts
  `cache_dir` and forwards it through to `from_pretrained()`/
  `download_model()`, so the CLI flag, the `MDXNET_INFER_CACHE_DIR` env
  var, and the `~/.cache/mdxnet-infer/` default now all work as documented
  (Weights UX contract leg 3: configurable weights folder).
- Removed a fabricated citation in README's Acknowledgments/Citation sections
  (misattributed to "Kimberley Jensen et al." under an invented paper title).
  Replaced with the real arXiv:2305.07489 citation ("Benchmarks and
  leaderboards for sound demixing tasks" by Solovyev, Stempkovskiy, Habruseva)
  as actually used by upstream ZFTurbo/Music-Source-Separation-Training.
- Corrected architecture attribution: the TFC-TDF v3 architecture traces to
  KUIELab's `sdx23` repository via ZFTurbo's Music-Source-Separation-Training,
  not to an unrelated "MDX23C-8KFFT-InstVoc_HQ" checkpoint.

### Removed
- **`drumsep-5stem` removed from `KNOWN_MODELS`.** No surviving original
  checkpoint could be found anywhere on the web (the only remaining copy
  is a non-drop-in OpenVINO conversion); CLI `--model` no longer offers
  it. `MDX23CConfig.drumsep_5stem()` (the architecture config) remains
  available for anyone with their own copy of the checkpoint. See
  README's Weights provenance section; it may return if a verified mirror
  surfaces. `MDX23CInference(model_name=...)` now raises `ValueError` for
  any unrecognized name unless an explicit `config`/`config_path` is also
  given (previously it silently fell back to the 6-stem config).

### Added
- `.github/workflows/publish.yml`: trusted-publishing (OIDC, no tokens)
  release workflow, gated on the full test suite passing.
- `.github/workflows/test.yml`: CI test workflow on push/PR (Python 3.10
  and 3.12).
- 6 new tests covering sha256 verification (match, mismatch + cleanup),
  the `KNOWN_MODELS` sha256 fields, and the unknown-`model_name` error
  path (40/40 passing, up from 34/34).

### Documented
- Rewrote README's "Weights provenance" section: the org-hosted
  `weights-drumsep-v1` release replacing the dead `jarredou/models`
  hosting, the dual-mirror byte-identical cross-verification story, a
  manual-download recipe with exact URLs/sha256/placement, the
  configurable weights-folder story (API argument, `MDXNET_INFER_CACHE_DIR`,
  `--cache-dir`, and the `~/.cache/mdxnet-infer/` default), and the
  5-stem model's lost-upstream status. The non-commercial-safe-only
  license caveat is unchanged.
- Documented that the DrumSep weights' license was never formally stated by
  the original authors (aufr33/jarredou); third-party mirrors disagree
  (MIT vs CC-BY-NC-SA-4.0 tags), so the weights are documented as
  non-commercial-safe only pending clarification.
- LICENSE now carries dual copyright (KUIELab, ZFTurbo/Roman Solovyev,
  OpenMIRLab) reflecting that `model.py` is a verbatim port of MIT-licensed
  upstream code, not an independent reimplementation.

### Changed
- Removed committed `__pycache__` bytecode artifacts from git; added
  `.gitignore`.
- Added file-top nav headers to load-bearing modules (`model.py`,
  `inference.py`) and a repo `CLAUDE.md` capability map.

## [0.1.0] - initial release

- Inference-only MDX23C TFC-TDF drum source separation (5-stem and 6-stem
  DrumSep models by aufr33/jarredou).
- `separate()` convenience function and `MDX23CInference` class API.
- CLI (`mdxnet-infer`).
