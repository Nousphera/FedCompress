# FedCompress

Research code for **Communication-Efficient Federated Learning Through Adaptive Weight Clustering And Server-Side Distillation**, ICASSP 2024, by Vasileios Tsouvalas, Aaqib Saeed, Tanir Ozcelebi, and Nirvana Meratnia.

FedCompress combines adaptive client-side weight clustering with server-side knowledge distillation on out-of-distribution data to reduce federated communication costs.

[Paper](https://doi.org/10.1109/ICASSP48485.2024.10447174) · [arXiv](https://arxiv.org/abs/2401.14211) · [Original repository](https://github.com/FederatedML/FedCompress)

Repository history and the scope of this independent copy are recorded in [PROVENANCE.md](PROVENANCE.md).

## Quick start

```bash
git clone https://github.com/Nousphera/FedCompress.git
cd FedCompress
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --dataset synthetic --clients 2 --rounds 2 \
  --epochs 2 --warmup-epochs 1 --server-epochs 1 --batch-size 8 \
  --min-clusters 2 --max-clusters 3 --output runs/smoke
python -m pytest -q
```

The maintained CPU runner uses the original models, clients, TFMOT clustering
and server distillation with stock Flower message serialization. It records
actual encoded payload bytes and saves model parameters. The synthetic fixture
checks execution; it is not a paper accuracy or compression result.

| Method | Client clustering | Server distillation |
| --- | --- | --- |
| `--method fedavg` | No | No |
| `--method client` | Yes | No |
| `--method fedcompress` | Yes | Yes |

Real training supports `cifar10`, `cifar100`, `pathmnist`, `spcm` and `voxforge`.
FedCompress additionally requires `--ood-dir` with prepared images or mono
16 kHz WAVs. [Recovered data recipes](docs/RECOVERED_DATA.md) describe the
inputs, optional TFDS dependencies and source provenance; corpora are not bundled.
[Reproduction notes](docs/REPRODUCIBILITY.md) explain the method mapping,
numerical repairs, data requirements, paper settings and transport measurements.
Full benchmark tables and device measurements have not been reproduced.

Original `run.py`/`configs.py` orchestration remains for historical comparison;
its prior pins are in `requirements-legacy.txt`. Model initialization assets,
Git history and project website are preserved.

## Citation

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{tsouvalas2024communicationefficient,
  author = {Tsouvalas, Vasileios and Saeed, Aaqib and Ozcelebi, Tanir and Meratnia, Nirvana},
  title = {Communication-Efficient Federated Learning Through Adaptive Weight Clustering And Server-Side Distillation},
  booktitle = {ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year = {2024},
  pages = {5805--5809},
  doi = {10.1109/ICASSP48485.2024.10447174}
}
```

## License

The original repository contains no license file. This copy does not add a project-wide license or grant additional permissions.
Third-party YAMNet attribution and Apache-2.0 terms are retained in [NOTICE](NOTICE).
<!-- nousphera-project-resources -->
## Project resources

[Project page](https://nousphera.github.io/FedCompress/) · [Research overview](https://nousphera.github.io/projects/fedcompress/) · [A0 poster](https://nousphera.github.io/FedCompress/poster.pdf) · [Paper](https://arxiv.org/abs/2401.14211)

This repository contains the research implementation. The project page and poster summarize the published method and reported experiments; this documentation update does not claim an independent reproduction of those experiments.
