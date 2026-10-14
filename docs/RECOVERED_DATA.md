# Restored dataset and auxiliary-data paths

The author's `fed_model_compress/latest/src/data.py` and `data_utils.py` on
Grill contained CIFAR-100, PathMNIST and VoxForge adapters omitted from the
published `release/` subtree. The inspected checkout HEAD is
`fb6cf7344f63e1d3e235627fe86d9d2d2113a145`; `latest/src/data.py` has uncommitted
research edits, so file hashes, not HEAD alone, identify this source. Exact
source hashes are recorded in PROVENANCE.md. The existing SpeechCommands,
ResNet-20 and YAMNet source is retained.

The maintained runner now covers the five paper datasets. Use the normal
installation; for SpeechCommands/VoxForge also install:

```bash
python -m pip install -r requirements-audio.txt
```

## Inputs and commands

- **CIFAR-100:** `--dataset cifar100`, downloaded by the original Keras loader;
  uses 100 output classes, including when a supplied partition lacks classes.
- **PathMNIST:** supply `DATA/raw/pathmnist/pathmnist.npz`, with uint8 RGB
  `train_images`, `test_images` shaped `(N,28,28,3)` and integer `train_labels`,
  `test_labels` shaped `(N,1)`. The original adapter retains the released
  train/test split, CIFAR mean/std normalization and 28-pixel crop. Official
  validation arrays may be present but are unused; each client reserves its
  own validation subset as in recovered source.
- **SpeechCommands/VoxForge:** `--dataset spcm` or `--dataset voxforge`, read by
  TensorFlow Datasets with stable file order. `--data-root DATA` selects its
  cache. On first training load, labels are enumerated from that exact dataset
  into `DATA/speech_commands/train_labels.npy` or `DATA/voxforge/train_labels.npy`.
  Subsequent loads reuse them. Use a fresh root for a different dataset version;
  an old same-length cache from another ordering cannot be detected by count.
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

Both audio paths use the recovered `YAMNet` implementation, a MobileNet-derived
architecture. The paper calls its audio model MobileNet; equivalence to its
exact historical checkpoint is not established. Audio preprocessing retains
40 mel bins, 400-sample frames, 160-sample hops, 1024 FFT and logarithmic
magnitude features. Training uses one-second segments; evaluation retains full
utterances and batches one example. Very short evaluation audio is padded to
one second to avoid empty spectra.

## Execution coverage and remaining gaps

Generated fixtures cover supplied PathMNIST files, CIFAR-100 class count,
SpeechCommands/VoxForge loader preprocessing and real YAMNet optimization,
and actual auxiliary WAV decoding/scale. Two-client one-round tests additionally
exercise PathMNIST and VoxForge local clustering, stock Flower transport and
server self-distillation with supplied generated images/WAVs.

The main method now has executable data adapters for all five paper datasets
and both auxiliary-data modalities. Still absent are the exact historical
StyleGAN image-generation recipe/selected image subset, LibriSpeech selected
utterances, explicit 25% client sample-count variance, and deployment/quantized
inference harnesses used for device measurements. Directory loaders accept
user-supplied auxiliary corpora; they do not regenerate those historical inputs.
No original trained checkpoint, full benchmark run or reported result is
bundled or claimed. Dataset downloads and full TFDS corpus construction were
not part of the generated-data validation.
