# Changelog

All notable changes to PyGX are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]


## [2.0.0] - 2026-11-20

### Added
- Test suite (pytest) covering models, discretizers, data splits and training steps.
  Runs on CPU in seconds, without a GPU or W&B.
- `build_model`: a single function that builds a model from a grid-search config,
  used by both the trainer and the explanation generator.
- `requirements-test.txt`: minimal CPU-only dependencies for running the tests.
- Continuous integration with GitHub Actions: Ruff and pytest run on every push and pull request.
- README section "Supported options per model".

### Changed
- `TopPercK` now keeps `ceil(n × topK)` nodes instead of `floor(n × topK)`, so a 
  non-zero fraction always selects at least one node. Explanation metrics that use
  `TopPercK` (e.g. fidelity) can differ from v1.0.0.
- `GraphConvCheb` and `GraphConvARMA` now raise a `ValueError` for `dropout > 0` or
  `graphNorm=True`, because their iterations are not separate layers. Previously these
  options were silently ignored.
- Unknown activation functions (`aF`) now raise a `ValueError`. Previously any value
  other than `Tanh` silently fell back to `ReLU`.
- Initial weights of `GraphConvCheb` and `GraphConvARMA` differ from v1.0.0 for the
  same random seed.

### Fixed
- Dropout was never applied: setting `dropout > 0` created no dropout layers.
- `GraphConvCheb` and `GraphConvARMA` ignored the `pooling` option and always used sum pooling.
- The explanation generator rebuilt models without the `pooling` option, so models
  trained with mean pooling were explained as sum-pooling models.
- Validation and test losses were logged at twice their real value.
- Discretizers (`TopPercK`, `PerfectK`, `LargestExplanationK`) selected every node
  when asked to select none, e.g. for an empty ground truth.

### Known issues
- Cached data splits do not depend on the dataset's contents: if a dataset changes
  but keeps the same root and split settings, the old splits are loaded silently.
- Requires PyTorch Geometric 2.3.x; newer versions reject the `train_data` argument
  that the trainer passes to every model.

## [1.0.0] - 2026-11-17

Final version of PyGX at the end of the Ph.D., as used for the thesis.

[Unreleased]: https://github.com/Fonta1/PyGX/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Fonta1/PyGX/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/Fonta1/PyGX/releases/tag/v1.0.0