# Methods and configuration

The [Flower App](FLOWER_APP.md) and sequential `train.py` use the same local
training, clustering, server distillation, and aggregation components. See
[dataset preparation](DATA_SETUP.md) for supported corpora and auxiliary inputs.

## Client training and server distillation

`--method fedcompress` combines local weight clustering with server distillation.
`--method client` clusters client models only; `--method fedavg` trains without
clustering. The latter two modes use no auxiliary data.

TensorFlow Model Optimization wraps convolution kernels; dense and
batch-normalization weights remain unclustered. Local warmup uses cross-entropy,
then training adds the learnable-centroid clustering objective. `--epochs`
counts both phases: the default ten local epochs contain five warmup and five
clustered epochs. Baselines use the same total local epoch budget.

After aggregation, the fixed global model teaches a separate clustered student
through temperature-scaled KL on unlabeled server inputs. Teacher parameters
remain unchanged. Server compression runs at initialization, each configured
interval, and the final round. The saved final parameters include this step.

```bash
python train.py --dataset cifar10 --data-root datasets --method fedcompress \
  --clients 20 --rounds 20 --epochs 10 --warmup-epochs 5 \
  --server-epochs 10 --ood-dir data/stylegan \
  --output runs/cifar10-fedcompress-seed42 --seed 42
```

`--init-model` loads an explicitly selected Keras H5 model. Otherwise,
initialization is seeded and random. Vision clients use seeded nonoverlapping
random partitions and reserve local validation examples. Client sizes vary
with assignment; a fixed 25% quantity variation is a separate experiment design.
The image frontend retains fixed-origin CIFAR cutout. See
[data formats](DATA_FORMATS.md) for audio preprocessing.

## Adaptive cluster capacity

Each client computes effective rank from its row-normalized embeddings. The
server averages client scores using training-example counts. Zero singular
values contribute zero to entropy; an all-zero embedding matrix has score zero.

Capacity increases after `--patience` consecutive changes in a `--window`-round
moving mean are at or below `--score-tolerance`, up to `--max-clusters`.
The default window and patience are both 3. Record the tolerance and cluster
bounds with the run; they directly determine the schedule.

## Auxiliary data

Image distillation accepts RGB images and resizes them to model input size using
the training normalization/augmentation. Audio distillation accepts mono 16 kHz
WAVs. Every supported file under the supplied directory participates, including
tail batches. The paper uses StyleGAN images and LibriSpeech audio; select an
auxiliary corpus explicitly and retain its preparation manifest.

## Communication and outputs

Both directions use lossless NPZ/DEFLATE data inside a standard Flower uint8
ndarray. The payload preserves shapes and dtypes while compressing repeated
clustered values. `upstream_protobuf_bytes` and `downstream_protobuf_bytes`
count serialized fit messages, including metadata. Zipped H5 size is recorded
separately as a model-artifact metric. Network latency and device inference
speed require their own measurements.

A fresh output directory receives `results.json`, `final-parameters.npz`, and
aggregated/compressed Keras H5 exports. Reports include configuration, versions,
example counts, effective rank, cluster counts, metrics, and message sizes.
Weight exports initialize models; optimizer and private client state belong to
the current run. The default generated fixture contains 32 client examples,
eight test examples, and eight independent auxiliary examples.
