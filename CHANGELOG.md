# Changelog

All notable changes to mdxnet-infer are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- Removed a fabricated citation in README's Acknowledgments/Citation sections
  (misattributed to "Kimberley Jensen et al." under an invented paper title).
  Replaced with the real arXiv:2305.07489 citation ("Benchmarks and
  leaderboards for sound demixing tasks" by Solovyev, Stempkovskiy, Habruseva)
  as actually used by upstream ZFTurbo/Music-Source-Separation-Training.
- Corrected architecture attribution: the TFC-TDF v3 architecture traces to
  KUIELab's `sdx23` repository via ZFTurbo's Music-Source-Separation-Training,
  not to an unrelated "MDX23C-8KFFT-InstVoc_HQ" checkpoint.

### Documented
- Added a "Weights provenance" section to README recording that the DrumSep
  checkpoint download URLs (`github.com/jarredou/models` release assets)
  currently 404 — the hosting GitHub account no longer exists. This is an
  unresolved, release-blocking issue; no fix is applied here because no
  verified replacement source exists yet.
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
