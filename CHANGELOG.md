# Changelog

All notable changes are recorded here. This project follows
[Semantic Versioning](https://semver.org/).

## [1.2.0] - 2026-11-04

### Changed
- Repository restructured for public release: package moved to `src/cbi/`,
  long-form documents moved to `docs/`, deployment notes folded into
  `docs/deployment.md`, freeze manifest archived under `docs/archive/`.
- `cbi.config.find_root` is public and is used by the stage runner, so stages
  work from a `src/` layout.
- `pyproject.toml` now declares the `src/` layout, package discovery, license,
  classifiers and the `test` extra; `requirements.txt` was dropped in favour of
  `pip install -e ".[test]"`.
- Added `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`,
  `.env.example`, `data/README.md`, and a GitHub Actions test workflow.

## [1.1.0] - 2026-11-02

### Added
- Section-aware monitor documents (claim, abstract, methods, data availability,
  claim-relevant results, gene rows) and an eight-key red-flag rubric.
- Evidence threshold calibrated at 5% false-positive rate; the rubric/holistic
  variant is selected on validation only.
- Fingerprinted P4 score cache with per-request persistence, preflight probe
  and offline training (`p4 --offline`, `p4 --prepare-only`).
- M2 acceptance controls, corpus audit and an M2 support-formula comparison.

### Fixed
- Batch judge input no longer truncated to 6,000 characters.
- Cache merge, variant persistence and test-threshold reporting defects.

## [1.0.0] - 2026-11-01

### Added
- P0–P8 pipeline: real cohort pool, public PMC claim extraction, ground-truth
  engine, context features, LLM monitor, red-team harness, grader, blue-team
  loop and downstream claim weights.
- Isolation acceptance tests and the offline test suite.
