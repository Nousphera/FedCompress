"""Prepared image/audio datasets, deterministic client splits and paper frontends."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF

from .models import AUDIO, CLASSES


@lru_cache(maxsize=1)
def mel_matrix():
    # Triangles in HTK-mel space, matching linear_to_mel_weight_matrix.
    hz = torch.linspace(0, 8000, 513, dtype=torch.float64)
    mel = 1127 * torch.log1p(hz / 700)
    edges = torch.linspace(
        1127 * np.log1p(60 / 700), 1127 * np.log1p(7800 / 700), 42, dtype=torch.float64
    )
    lower = (mel[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2])
    upper = (edges[2:] - mel[:, None]) / (edges[2:] - edges[1:-1])
    weights = torch.minimum(lower, upper).clamp(min=0)
    weights[0] = 0
    return weights.float()


def audio_features(waveform, training=False):
    x = torch.as_tensor(waveform, dtype=torch.float32).flatten()
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Audio must be a nonempty finite waveform")
    missing = max(0, 16000 - len(x))
    x = torch.nn.functional.pad(x, (missing // 2, missing - missing // 2))
    if training and len(x) > 16000:
        start = int(torch.randint(len(x) - 16000 + 1, (1,)))
        x = x[start : start + 16000]
    # torch.stft with n_fft=1024 would center the 400-sample window
    # differently. Explicit framing preserves the original 400/160 frontend.
    frames = x.unfold(0, 400, 160) * torch.hann_window(400)
    spectrum = torch.fft.rfft(frames, n=1024).abs()
    return (spectrum @ mel_matrix() + 1e-6).log().unsqueeze(0)


def image_features(image, training=False, size=32, *, cutout=True):
    image = (
        TF.to_pil_image(np.asarray(image))
        if not isinstance(image, Image.Image)
        else image
    )
    image = image.convert("RGB")
    if image.size != (size, size):
        image = TF.resize(image, [size, size])
    x = TF.to_tensor(image)
    if training:
        if torch.rand(()) < 0.5:
            x = x.flip(-1)
        x = torch.nn.functional.pad(x, (4, 4, 4, 4))
        top, left = torch.randint(9, (2,)).tolist()
        x = x[:, top : top + size, left : left + size]
    x = TF.normalize(x, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if training and cutout:
        # Preserve the fixed-origin cutout used in the released experiments.
        x[:, :8, :8] = 0
    return x


class ImageData(Dataset):
    def __init__(self, directory, split, training=False):
        self.images = np.load(
            directory / f"{split}_images.npy", mmap_mode="r", allow_pickle=False
        )
        self.labels = np.load(
            directory / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False
        )
        self.training = training

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return image_features(
            self.images[index], self.training, self.images.shape[1]
        ), int(self.labels[index])


class AudioData(Dataset):
    def __init__(self, directory, split, training=False):
        self.records = json.loads((directory / f"{split}.json").read_text())
        self.labels = np.array([r["label"] for r in self.records], dtype=np.int64)
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with sf.SoundFile(record["path"]) as stream:
            if stream.samplerate != 16000 or stream.channels != 1:
                raise ValueError("Audio must be mono 16 kHz")
            stream.seek(record.get("start", 0))
            # The released frontend divides integer PCM by int16.max.
            x = (
                stream.read(record.get("frames", -1), dtype="int16").astype(np.float32)
                / 32767
            )
        return audio_features(x, self.training), int(record["label"])


class AuxiliaryData(Dataset):
    def __init__(self, root, name, size=32, limit=0):
        manifest = json.loads((Path(root) / name / "manifest.json").read_text())
        self.files = manifest["media"][: limit or None]
        self.audio = name == "librispeech"
        self.size = size
        if not self.files:
            raise ValueError("Empty auxiliary corpus")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]["path"]
        if self.audio:
            with sf.SoundFile(path) as stream:
                if stream.samplerate != 16000 or stream.channels != 1:
                    raise ValueError("Auxiliary audio must be mono 16 kHz")
                # Read one segment, not an entire audiobook utterance.
                start = int(torch.randint(max(1, len(stream) - 16000 + 1), (1,)))
                stream.seek(start)
                x = stream.read(16000, dtype="int16").astype(np.float32) / 32767
            return audio_features(x, True)
        with Image.open(path) as image:
            return image_features(image, True, self.size, cutout=False)


def load_dataset(name, root, split="train", training=False):
    if name not in CLASSES:
        raise ValueError(f"Unsupported dataset: {name}")
    directory = Path(root) / name
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Prepare data first: python scripts/prepare_data.py --dataset {name}"
        )
    return (
        AudioData(directory, split, training)
        if name in AUDIO
        else ImageData(directory, split, training)
    )


def partitions(count, clients, seed, quantity_skew=0.25):
    if clients < 1 or count < clients * 2 or not 0 <= quantity_skew <= 1:
        raise ValueError(
            "Each client needs at least two examples; quantity skew must be in [0,1]"
        )
    rng = np.random.default_rng(seed)
    # Relative spread around equal allocation; labels are randomly distributed.
    weights = rng.uniform(1 - quantity_skew, 1 + quantity_skew, clients)
    raw = weights / weights.sum() * (count - 2 * clients)
    sizes = np.floor(raw).astype(int) + 2
    remaining = count - int(sizes.sum())
    sizes[np.argsort(-(raw - np.floor(raw)))[:remaining]] += 1
    return np.split(rng.permutation(count), np.cumsum(sizes)[:-1])


def client_data(cfg, cid):
    train = load_dataset(cfg["dataset"], cfg["data-root"], training=True)
    valid = load_dataset(cfg["dataset"], cfg["data-root"])
    indexes = partitions(
        len(train), cfg["nr-clients"], cfg["seed"], cfg["quantity-skew"]
    )[cid]
    reserved = max(1, int(np.ceil(len(indexes) * cfg["validation-fraction"])))
    reserved = min(reserved, len(indexes) - 1)
    return Subset(train, indexes[reserved:].tolist()), Subset(
        valid, indexes[:reserved].tolist()
    )


def loader(dataset, batch_size, seed, training=False, audio=False):
    return DataLoader(
        dataset,
        batch_size=1 if audio and not training else batch_size,
        shuffle=training,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )
