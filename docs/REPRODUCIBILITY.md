# Reproducibility and method correspondence

The native `flwr run .` path is documented in [FLOWER_APP.md](FLOWER_APP.md).
The sequential commands below remain useful for CPU debugging; both paths reuse
the same maintained algorithm core. [DATA_SETUP.md](DATA_SETUP.md) provides the
current preparation/import/validation entry points.

## Maintained CPU environment

Use Python 3.11 and `pip install -r requirements.txt`. The maintained path is
`train.py`, with TensorFlow 2.15.1, TensorFlow Model Optimization 0.7.5 and
Flower 1.13.1. It executes original clients, clustering, distillation and server
strategy sequentially, passing fit instructions and responses through stock
Flower protobuf serialization. No patched Flower or Ray cluster is needed.
`run.py`, `configs.py` and [the historical dependency transcript](HISTORICAL_REQUIREMENTS.md) retain the older orchestration
and environment for comparison; that Ray/GPU path is not newly validated.

```bash
python train.py --dataset synthetic --clients 2 --rounds 2 \
  --epochs 2 --warmup-epochs 1 --server-epochs 1 --batch-size 8 \
  --min-clusters 2 --max-clusters 3 --output runs/smoke
python -m pytest -q
```

The fixture uses 32 client examples (two per client reserved for validation),
eight test examples and eight independently generated unlabeled server images.
It only checks execution. Use a new output directory. Outputs include
`results.json`, `final-parameters.npz`, and final aggregated/compressed Keras H5
models. Reports record configuration, versions, example counts, effective rank,
cluster counts and measured serialized message bytes.

## CIFAR-10 experiment

```bash
python train.py --dataset cifar10 --method fedcompress \
  --clients 20 --rounds 20 --epochs 10 --warmup-epochs 5 \
  --server-epochs 10 --ood-dir /path/to/prepared/stylegan/images \
  --output runs/cifar10-fedcompress-seed42 --seed 42
```

CIFAR-10 downloads through Keras. Server self-compression requires actual
unlabeled images; missing `--ood-dir` is an error. The original StyleGAN
(Oriented) image corpus and generation recipe are not bundled. Images are
resized to model input size and use the original CIFAR normalization/augmentation.
`--method fedavg` runs the same total local epochs without clustering or server
training. `--method client` clusters locally without server self-compression.
`--epochs` always counts warmup plus clustered local training; the old launcher
had different baseline epoch accounting. `--init-model` accepts an explicitly
chosen Keras H5 model; by default initialization is seeded and random.

Recovered adapters extend the maintained runner to CIFAR-100, PathMNIST,
SpeechCommands and VoxForge, including actual auxiliary image/WAV loading.
See [RECOVERED_DATA.md](RECOVERED_DATA.md) for preparation, dependencies,
source provenance and generated-data execution checks. The exact historical
auxiliary corpus, device deployment/quantization benchmarks and full quantitative
tables remain unavailable. The random nonoverlapping vision partition does not
implement the paper's separate 25% quantity-variance control.

## Method correspondence and numerical repairs

- Original TFMOT clustering wraps convolution kernels only; dense and batch
  normalization weights remain unclustered. Local warmup uses cross-entropy,
  then the original learnable-centroid clustering path. Default five warmup plus
  five clustered epochs preserves the supplied configuration's division.
- The newly aggregated model is the fixed teacher; a separate clustered student
  uses temperature-scaled KL on unlabeled server images. The teacher parameters
  remain unchanged during self-compression. This follows the paper prose; the
  appendix pseudocode has inconsistent teacher/round assignments.
- Each client now computes an effective-rank scalar locally, and the server
  weights those scalars by client training examples, as the paper specifies.
  Previously it concatenated raw client embeddings and computed a different
  global score. Original row normalization is retained. Zero singular values
  no longer produce NaNs; the all-zero matrix has score zero.
- Cluster capacity increases after `--patience` consecutive changes in a
  `--window`-round moving mean are at or below `--score-tolerance`. It is bounded
  by `--max-clusters`. This implements the paper's score plateau criterion;
  the old joint accuracy/score trigger is retained in Git history. The published
  defaults are window 3 and patience 3. The precise tolerance remains a run
  parameter and must be recorded.
- Server compression includes every configured interval, initialization and
  the final round. Loss is now compiled for post-clustering/server evaluation.
- Native TensorFlow operations preserve the original fixed-origin cutout,
  removing the unused Addons installation requirement for vision.

These fixes can change numerical results. `federatedml-baseline` and
[HISTORICAL_AUDIT.md](HISTORICAL_AUDIT.md) retain the earlier behavior/audit.
No old accuracy, compression or inference measurement was regenerated.

## Actual transport versus model-size estimates

Both directions use a lossless NPZ/DEFLATE payload inside a standard Flower
uint8 ndarray. This preserves shapes/dtypes and lets repeated clustered weights
compress without modifying Flower. `upstream_protobuf_bytes` and
`downstream_protobuf_bytes` count the actual serialized fit messages, including
metadata. They are local serialization measurements, not network latency.
The original zipped H5 size metric is retained separately; it is an artifact-size
estimate, not the measured communication cost. Neither metric is automatically
the paper's reported CCR/MCR or a hardware inference speedup.

## Licensing and provenance

The original source has no LICENSE file. This release does not invent a new
license or confer third-party permissions. Original Git history and attribution
are preserved; external libraries and datasets retain their own terms.
