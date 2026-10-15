# Dataset preparation

Run commands from the repository root in the README's environment.
`python scripts/prepare_data.py --help` lists datasets and modes.
`--download` explicitly enables network acquisition; `--source` imports a safe
NPZ or already-prepared directory. Import rejects existing destination files and
symlinks in directory trees. Imports validate in a temporary staging directory before publishing; failed imports leave no dataset or manifest behind. Directory imports copy only the selected dataset subtree, and auxiliary imports copy only supported media files. `--validate` reads schema and content again and
compares the deterministic SHA-256 manifest; it never downloads. Omitting all
three modes registers an already-present dataset after validating its schema.

Prepared image files contain `train_images`, `train_labels`, `test_images`, and
`test_labels`. RGB images are uint8 NHWC with shape `(N,32,32,3)` for CIFAR. PathMNIST uses `(N,28,28,3)`.
Labels are integer IDs shaped `(N,)` or `(N,1)`. Preparation checks shapes,
class ranges, nonempty splits, and counts. Training reads these files directly
and preserves their row order.

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --source /path/to/cifar10.npz
python scripts/prepare_data.py --dataset cifar10 --root datasets --validate
```

A manifest records dataset name, split sizes, class count and relative file
paths/content hashes. Retain the source publication, data license, and acquisition record alongside
this manifest. A changed file requires a new destination/preparation run;
manifests are not silently regenerated over changed data.

## Public vision sources

[CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) use the official
train/test splits via the pinned Keras loader. Files are saved
as `ROOT/cifar10.npz` and `ROOT/cifar100.npz` for training:

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --download
python scripts/prepare_data.py --dataset cifar100 --root datasets --download
```

PathMNIST uses the official 28x28 RGB NPZ from
[MedMNIST](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py),
[Zenodo record 10519652](https://zenodo.org/records/10519652).
Download checks MD5 `a8b06965200029087d5bd730944a56c1` before import. The
89,996-image train and 7,180-image test splits are used; the official validation
split is retained in the downloaded NPZ but is not merged into training.

```bash
python scripts/prepare_data.py --dataset pathmnist --root datasets --download
python scripts/prepare_data.py --dataset pathmnist --root datasets --validate
```
The trainer reads `datasets/raw/pathmnist/pathmnist.npz` with `--data-root datasets`.

## Speech Commands and VoxForge

```bash
python scripts/prepare_data.py --dataset spcm --root datasets --download
python scripts/prepare_data.py --dataset spcm --root datasets --validate
```

Speech Commands uses the [TFDS 4.9.4 builder](https://github.com/tensorflow/datasets/blob/v4.9.4/tensorflow_datasets/datasets/speech_commands/speech_commands_dataset_builder.py).
VoxForge requires manual corpus acquisition and Apache Beam preparation under
[the upstream builder's instructions](https://github.com/tensorflow/datasets/blob/v4.9.4/tensorflow_datasets/audio/voxforge.py).
That build may require about 100 GB temporary disk; this repository does not
silently launch it. Use a separate suitable preparation environment, then
register/validate the existing TFDS cache in place:

```bash
python scripts/prepare_data.py --dataset voxforge --root /path/to/prepared-tfds-cache
python scripts/prepare_data.py --dataset voxforge --root /path/to/prepared-tfds-cache --validate
```

The validator enumerates train/test audio and labels, checks counts/class space,
and hashes local prepared files. Training uses the same root and the
16-file/16-record deterministic interleave. Source-bound label-cache validation
in [DATA_FORMATS.md](DATA_FORMATS.md) remains active. Keep a prepared corpus unchanged during training. Client partitions are generated
from the configured seed and dataset order.

## Auxiliary server-distillation data

Import or register RGB images or mono 16 kHz 16-bit PCM WAVs. The source directory is
copied only to a new root; existing in-place corpora can be registered without
`--source`. Validated files are decoded and hashed:

```bash
python scripts/prepare_data.py --dataset ood-images --source /path/to/stylegan-images --root data/stylegan
python scripts/prepare_data.py --dataset ood-images --root data/stylegan --validate
python scripts/prepare_data.py --dataset ood-audio --source /path/to/librispeech-wavs --root data/librispeech
python scripts/prepare_data.py --dataset ood-audio --root data/librispeech --validate
```

Pass the resulting root as `--ood-dir`. Images support PNG/JPEG/BMP/GIF; WAVs
must be nonempty, mono, 16 kHz and 16-bit PCM, with lowercase `.wav` filenames to match the training reader. Image/waveform tails remain in training.
The paper uses StyleGAN images and LibriSpeech speech. Select the auxiliary
images or utterances explicitly; preparation records the files used by this run.
