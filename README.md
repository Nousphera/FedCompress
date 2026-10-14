# FedCompress

Research code for **Communication-Efficient Federated Learning Through Adaptive Weight Clustering And Server-Side Distillation**, ICASSP 2024, by Vasileios Tsouvalas, Aaqib Saeed, Tanir Ozcelebi, and Nirvana Meratnia.

FedCompress combines adaptive client-side weight clustering with server-side knowledge distillation on out-of-distribution data to reduce federated communication costs.

[Paper](https://doi.org/10.1109/ICASSP48485.2024.10447174) · [arXiv](https://arxiv.org/abs/2401.14211) · [Original repository](https://github.com/FederatedML/FedCompress)

Repository history and the scope of this independent copy are recorded in [PROVENANCE.md](PROVENANCE.md).

## Quickstart

The launcher and utility tests require only Python's standard library:

```bash
git clone https://github.com/Nousphera/FedCompress.git
cd FedCompress
python run.py --help
python -B -S -m unittest discover -s tests -v
```

Training uses the historical pinned stack in `requirements.txt` (TensorFlow GPU 2.10.1, Keras 2.10.0, Flower 1.1.0, and Ray 2.1.0). A Linux x86-64 environment with Python 3.10 is a candidate starting point based on published wheel availability; it has **not** been validated as an installable environment for the complete requirements file. Read [environment and reproduction notes](docs/REPRODUCIBILITY.md) before installation.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

After preparing the environment and required data:

```bash
# FedAvg
python run.py --datasets cifar10 --methods fedavg
# Client-side weight clustering
python run.py --datasets cifar10 --methods client
# FedCompress: client clustering and server distillation
python run.py --datasets cifar10 --methods fedcompress
```

Omitting `--methods` selects `fedcompress`. These commands start full experiments, not smoke tests. The launcher currently supports CIFAR-10 and sets 100 rounds; other parameters are in `configs.py`. It resolves model/output paths relative to the repository and uses the active Python interpreter.

## Methods and data

| Method | Client clustering | Server distillation | Required data |
| --- | --- | --- | --- |
| `fedavg` | No | No | CIFAR-10 |
| `client` | Yes | No | CIFAR-10 |
| `fedcompress` | Yes | Yes | CIFAR-10 and external StyleGAN images |

CIFAR-10 is loaded through Keras. FedCompress expects a prepared image directory at `$TFDS_DATA_DIR/raw/stylegan_oriented/`; these images and their generation recipe are not included. Set `TFDS_DATA_DIR` to your dataset root before running FedCompress. Speech Commands/LibriSpeech code is also present, but is not exposed by the launcher and requires additional external files. See [data prerequisites](docs/REPRODUCIBILITY.md#data-and-assets).

The repository includes initialization weights in `assets/init_models/`. Trained checkpoints and experiment results are not included. New outputs go to `assets/saved_models/` and `assets/results/`.

## Reproduction status

The maintenance tests cover launcher and utility behavior. They do not validate dependency installation, data preparation, model loading, GPU/Ray execution, or reported paper results.

Historical baseline-epoch settings and compression scheduling are preserved, including known caveats. Consult [the reproduction notes](docs/REPRODUCIBILITY.md#preserved-research-behavior) before comparing approaches or reproducing the paper. The model architectures, training objectives, clustering logic, and initialization assets are unchanged.

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

The original repository contains no license file. This copy does not add a license or grant additional permissions.
<!-- nousphera-project-resources -->
## Project resources

[Project page](https://nousphera.github.io/FedCompress/) · [Research overview](https://nousphera.github.io/projects/fedcompress/) · [A0 poster](https://nousphera.github.io/FedCompress/poster.pdf) · [Paper](https://arxiv.org/abs/2401.14211)

This repository contains the research implementation. The project page and poster summarize the published method and reported experiments; this documentation update does not claim an independent reproduction of those experiments.
