# Dataset formats and audio caches

The README installation includes the Speech Commands and VoxForge dependencies
in `requirements.txt`.

## Inputs and commands

- **CIFAR-100:** `--dataset cifar100`, downloaded by the original Keras loader;
  uses 100 output classes, including when a supplied partition lacks classes.
- **PathMNIST:** supply `DATA/raw/pathmnist/pathmnist.npz`, with uint8 RGB
  `train_images`, `test_images` shaped `(N,28,28,3)` and integer `train_labels`,
  `test_labels` shaped `(N,1)`. The original adapter retains the released
  train/test split, CIFAR mean/std normalization and 28-pixel crop. Official
  validation arrays may be present but are unused; each client reserves its
  own validation subset before local training.
- **SpeechCommands/VoxForge:** `--dataset spcm` or `--dataset voxforge`, read by
  TensorFlow Datasets with deterministic file order and explicit 16-file,
  16-record interleaving. `--data-root DATA` selects its cache. Labels are
  enumerated once per verified prepared source into
  `DATA/{speech_commands,voxforge}/train_labels-<sha256>.npy`. The key covers
  TFDS `info.full_name` (including configuration/version), split, ordered file
  instructions (paths, skips, takes and shard counts), reader settings and
  TensorFlow/TFDS versions. Each local shard's resolved path, size, nanosecond
  modification/change times, device and inode are checked before reuse. A
  different version, order or changed shard therefore cannot reuse a cache
  solely because it has the same number of examples. Temporary files are
  flushed and atomically replaced; malformed caches are rebuilt. Legacy
  unkeyed `train_labels.npy` files are ignored and left untouched.
  If prepared-file metadata cannot be verified (including remote shards),
  labels are enumerated on each load without reading or writing a disk cache.
  Keep prepared corpora unchanged while training: these are filesystem
  identities, not full-content hashes or a concurrent dataset-update protocol.
  Changes detected during label enumeration raise an error.
  SpeechCommands can download/build through TFDS when invoked on an empty
  cache. VoxForge must already be prepared: this reader sets `download=False`.
  Its upstream TFDS builder requires manually downloaded archives and Apache
  Beam construction (about 100 GB temporary disk space), documented in the
  [TFDS VoxForge builder](https://github.com/tensorflow/datasets/blob/v4.9.4/tensorflow_datasets/audio/voxforge.py).
  No corpus is bundled in this repository.

Vision server self-compression takes a directory of user-prepared images.
Audio server self-compression takes recursively discovered **mono 16 kHz WAVs**.
A different sample rate raises an error instead of being mislabeled as 16 kHz.
Decoded float waveforms retain their scale; the integer-PCM conversion used by
supervised TFDS audio is not reapplied. Tail batches are retained, including
when fewer auxiliary files exist than the batch size.

```bash
python train.py --dataset pathmnist --data-root DATA \
  --ood-dir /path/to/prepared/stylegan/images --output runs/pathmnist
python train.py --dataset voxforge --data-root DATA \
  --ood-dir /path/to/prepared/librispeech-wavs --output runs/voxforge
python train.py --dataset spcm --data-root DATA \
  --ood-dir /path/to/prepared/librispeech-wavs --output runs/spcm
```

Both audio paths use `YAMNet`, a MobileNet-derived architecture. Audio preprocessing retains
40 mel bins, 400-sample frames, 160-sample hops, 1024 FFT and logarithmic
magnitude features. Training uses one-second segments; evaluation retains full
utterances and batches one example. Very short evaluation audio is padded to
one second to avoid empty spectra.

## Corpus selection

Choose auxiliary images or utterances explicitly and keep the preparation
manifest with results. Vision partitions use seeded random nonoverlapping
assignment; client sample counts vary with the assignment. Device deployment
and quantized inference measurements require a separate evaluation harness.
