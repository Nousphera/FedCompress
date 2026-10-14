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

## Restored dataset adapters (2026-10-15)

Author source `fed_model_compress/latest`, inspected checkout HEAD
`fb6cf7344f63e1d3e235627fe86d9d2d2113a145`; working-tree changes preserved
in the read-only audit snapshot. Imported functions add CIFAR-100, PathMNIST
and VoxForge; preparation/runtime repairs are disclosed in docs/RECOVERED_DATA.md.
Source-file SHA-256:

- `latest/src/data.py`: `7ef02fbb430546e0ca0319c11f517666557812c2211ba7e19f27961694e81d4f`
- `latest/src/data_utils.py`: `d5456c21c487b4b58f1bc87562533afc991c25849db6bd36c7486de4f1248df7`

Only reviewed source functions were imported; no private history, dataset,
checkpoint or credentials were copied. Original licensing status is retained.

YAMNet attribution restored from TensorFlow models, inspected upstream
`c651073789719ff39dcf51406733bbcaaafb43f2` (`research/audioset/yamnet/yamnet.py`), Apache-2.0.
This is an attribution reference, not proof of the original copied commit.


## Reproducible setup follow-up

Added dataset import/download/validation commands and deterministic file manifests.
Prepared NPZ files feed the maintained trainer directly. Image labels and shapes
are validated against fixed dataset class spaces; explicit dataset imports do not
certify historical order, partitions or benchmark equivalence. Existing source,
teacher identity and cache qualifications are retained in the linked guides.

Added real Flower ClientApp/ServerApp entry points reusing the maintained core.
Logical partition IDs determine sampling and persistent local state; evaluated
global weights and round results are saved. Flower 1.13/Ray 2.10 remain pinned for
compatibility. See docs/FLOWER_APP.md for single-host simulation limits.
