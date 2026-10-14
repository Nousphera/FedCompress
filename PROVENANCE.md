# Provenance

This repository is an independent Nousphera copy of [FedCompress](https://github.com/FederatedML/FedCompress). The original FederatedML repository remains in place.

- Source commit: [`86102531097155cdc0bd94f2eac186bc4cdde197`](https://github.com/FederatedML/FedCompress/commit/86102531097155cdc0bd94f2eac186bc4cdde197)
- Source branch: `main`
- Baseline tag in this repository: `federatedml-baseline`
- Original Git history and author attribution are retained.

Maintenance changes after this baseline improve documentation, setup, and validation. They do not constitute a new reproduction of the published experiments. For the original code and configuration, inspect the baseline tag.

GitHub issues, pull requests, stars, releases, and Pages configuration are not part of Git history and have not been transferred. Existing discussions remain at the source repository.

## Maintained execution release (2026-10-15)

`train.py` adds sequential CPU orchestration and lossless stock-Flower payloads
around the original research components. Per-client effective-rank aggregation,
score plateau scheduling, compression intervals and evaluation losses are
repaired; changes and remaining paper/source boundaries are documented in
`docs/REPRODUCIBILITY.md`. No private dataset/history or benchmark result was
added. No license is inferred for the original unlicensed source.
