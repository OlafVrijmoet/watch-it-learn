# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-26

First public release.

### Added
- Build a transformer as a canvas: attention and feed-forward sections, each with per-section config
- Deterministic, bit-exact training replay with a stage scrubber
- On-the-fly gradient introspection (per block, head, and neuron)
- Honest held-out accuracy on a deterministic 20% split
- Comparative experiments across multiple model versions
- From-scratch SVG model renderer
- CI plus Dependabot, pip-audit, and Bandit security scanning

[0.1.0]: https://github.com/OlafVrijmoet/watch-it-learn/releases/tag/v0.1.0
