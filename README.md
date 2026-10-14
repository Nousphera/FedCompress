# FedCompress

**Communication-Efficient Federated Learning through Adaptive Weight Clustering and Server-Side Distillation**
[Paper](https://doi.org/10.1109/ICASSP48485.2024.10447174) · [Project page](https://nousphera.github.io/FedCompress/) · [Code](https://github.com/Nousphera/FedCompress) · [Source provenance](PROVENANCE.md)

## Abstract

Federated Learning (FL) is a promising technique for the collaborative training of deep neural networks across multiple devices while preserving data privacy. Despite its potential benefits, FL is hindered by excessive communication costs due to repeated server-client communication during training. To address this challenge, model compression techniques, such as sparsification and weight clustering are applied, which often require modifying the underlying model aggregation schemes or involve cumbersome hyperparameter tuning, with the latter not only adjusts the model's compression rate but also limits model's potential for continuous improvement over growing data. In this paper, we propose FedCompress, a novel approach that combines dynamic weight clustering and server-side knowledge distillation to reduce communication costs while learning highly generalizable models. Through a comprehensive evaluation on diverse public datasets, we demonstrate the efficacy of our approach compared to baselines in terms of communication costs and inference speed.

*The abstract describes the published work. The maintained release has not reproduced its benchmark tables.*

## Setup

Use Python 3.11. The commands below create a dedicated virtual environment.
Flower 1.13.1 and Ray 2.10.0 are pinned; TensorFlow projects require the
compatible protobuf dependency range. Keep the checkout and virtual environment
on paths without spaces for Ray subprocesses. Do not upgrade Flower independently.

```bash
mkdir -p "$HOME/research-code" "$HOME/.venvs"
cd "$HOME/research-code"
git clone https://github.com/Nousphera/FedCompress.git
cd FedCompress
python3.11 -m venv "$HOME/.venvs/fedcompress"
source "$HOME/.venvs/fedcompress/bin/activate"
python -m pip install -r requirements.txt -r requirements-simulation.txt
python -m pytest -q
```

## Datasets

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --download
python scripts/prepare_data.py --dataset cifar10 --root datasets --validate
```

CIFAR-100 and PathMNIST use the same entry point; Speech Commands and VoxForge
use prepared TFDS corpora. FedCompress also needs auxiliary images or mono
16 kHz WAVs. [Dataset setup](docs/DATA_SETUP.md) gives preparation and validation
commands for all five datasets and both auxiliary modalities. Original auxiliary
subsets and experiment partitions were not recovered.

## Run

```bash
flwr run .
```

The default executes the real client clustering and server distillation for
two rounds on a generated fixture. For prepared CIFAR-10 and auxiliary images:

```bash
flwr run . paper --run-config 'options="--dataset cifar10 --data-root datasets --ood-dir data/stylegan --clients 20 --rounds 20 --output runs/cifar10"'
```

`--method fedcompress` is the default; `client` disables server distillation and
`fedavg` disables both clustering and distillation. These baselines do not need
auxiliary data. The sequential CPU runner remains available:

```bash
python train.py --dataset synthetic --clients 2 --rounds 2 --epochs 2 \
  --warmup-epochs 1 --server-epochs 1 --batch-size 8 --min-clusters 2 \
  --max-clusters 3 --output runs/sequential
```

## Outputs and reproducibility

Every run requires a new output directory. `results.json` records configuration,
versions, sampled logical clients, example counts and round metrics;
`final-parameters.npz` saves the evaluated global weights. Native simulations
preserve run-owned caches under the output directory across worker lifetimes.
They are local simulations using a shared filesystem, not multi-host deployments.
They do not measure network latency. Final weights are not complete resumable
optimizer/client-state checkpoints. Synthetic metrics are execution checks only.

The `options` run-config value is the complete runner argument string. Set the selected federation's
`options.num-supernodes` to exactly the configured client count; mismatches fail
explicitly. [Flower setup details](docs/FLOWER_APP.md) document identity, state,
sampling and the tested runtime boundary.

Client clustering, adaptive cluster scheduling and server self-distillation use
the recovered core. [Reproduction notes](docs/REPRODUCIBILITY.md) and
[recovered data](docs/RECOVERED_DATA.md) distinguish source correspondence from
missing historical OOD selections, partitions and device evaluation. Audio label
caches remain bound to prepared TFDS source identity and reader order.

## Citation

Use [CITATION.cff](CITATION.cff) or [citation.bib](citation.bib).

## License

The original repository has no project-wide license grant; this release does not add one. Third-party YAMNet Apache-2.0 attribution is retained in [NOTICE](NOTICE).
