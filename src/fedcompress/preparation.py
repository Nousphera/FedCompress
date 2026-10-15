"""Download, prepare and validate the five paper datasets and auxiliary corpora."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
from PIL import Image

from .models import AUDIO, CLASSES

WORDS = [
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes",
    "_silence_",
    "_unknown_",
]
LANGUAGES = ["de", "en", "es", "fr", "it", "ru"]
DATASETS = (*CLASSES, "stylegan", "librispeech")
SPEECH = {
    "train": (
        "speech_commands_v0.02.tar.gz",
        "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
    ),
    "test": (
        "speech_commands_test_set_v0.02.tar.gz",
        "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df",
    ),
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, expected=None, algorithm="sha256"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected and digest(path, algorithm) != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        return path
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(
        url,
        headers={"User-Agent": "FedCompress/0.2", "Range": f"bytes={offset}-"}
        if offset
        else {"User-Agent": "FedCompress/0.2"},
    )
    with urlopen(request, timeout=60) as response:
        append = offset and response.status == 206
        size = int(response.headers.get("Content-Length", 0))
        if size and shutil.disk_usage(path.parent).free < size + 128 * 2**20:
            raise OSError(f"Insufficient disk space to download {url}")
        with partial.open("ab" if append else "wb") as stream:
            shutil.copyfileobj(response, stream, length=2**20)
        if size and partial.stat().st_size != size + (offset if append else 0):
            raise OSError(f"Incomplete download; rerun to resume: {url}")
    if expected and digest(partial, algorithm) != expected:
        partial.unlink()
        raise ValueError(f"Download checksum mismatch: {url}")
    partial.replace(path)
    return path


def extract(archive, destination):
    destination = Path(destination).resolve()
    marker = destination / ".extracted.json"
    identity = {"archive": str(Path(archive).resolve()), "sha256": digest(archive)}
    if marker.is_file():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Extracted archive identity changed")
        return
    destination.mkdir(parents=True, exist_ok=True)

    def safe(name):
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Unsafe archive path")
        return target

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = safe(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive links are not allowed")
                if member.is_dir():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
    else:
        with tarfile.open(archive, "r:*") as source:
            for member in source:
                target = safe(member.name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("Archive links and special files are not allowed")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
    marker.write_text(json.dumps(identity, indent=2) + "\n")


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def audio_record(path, label, start=0, frames=-1, speaker=None):
    info = sf.info(path)
    if (
        info.samplerate != 16000
        or info.channels != 1
        or info.frames <= 0
        or info.subtype != "PCM_16"
    ):
        raise ValueError(f"Expected nonempty mono 16 kHz PCM16 audio: {path}")
    row = {"path": str(Path(path).resolve()), "label": int(label)}
    if start or frames >= 0:
        row.update(start=start, frames=frames)
    if speaker is not None:
        row["speaker"] = speaker
    return row


def vision(name, raw, stage):
    if name in ("cifar10", "cifar100"):
        from torchvision.datasets import CIFAR10, CIFAR100

        cls = CIFAR10 if name == "cifar10" else CIFAR100
        splits = {
            split: cls(str(raw), train=split == "train", download=True)
            for split in ("train", "test")
        }
        arrays = {
            f"{split}_{key}": np.asarray(
                ds.data if key == "images" else ds.targets,
                dtype=np.uint8 if key == "images" else np.int64,
            )
            for split, ds in splits.items()
            for key in ("images", "labels")
        }
    else:
        source = download(
            "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1",
            raw / "pathmnist.npz",
            "a8b06965200029087d5bd730944a56c1",
            algorithm="md5",
        )
        with np.load(source, allow_pickle=False) as data:
            arrays = {
                f"{split}_{key}": data[f"{split}_{key}"]
                for split in ("train", "val", "test")
                for key in ("images", "labels")
            }
    counts = {}
    for key, value in arrays.items():
        if key.endswith("labels"):
            value = value.reshape(-1).astype(np.int64)
            if not len(value) or value.min() < 0 or value.max() >= CLASSES[name]:
                raise ValueError("Invalid class labels")
            counts[key.split("_")[0]] = len(value)
        else:
            if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
                raise ValueError("Invalid image arrays")
        np.save(stage / (key + ".npy"), value, allow_pickle=False)
    return {
        "splits": counts,
        "classes": CLASSES[name],
        "source": "official train/test splits",
    }


def speech(raw, stage):
    for split, (filename, checksum) in SPEECH.items():
        corpus = raw / split
        if not corpus.exists() or not (corpus / ".extracted.json").exists():
            archive = download(
                "https://download.tensorflow.org/data/" + filename,
                raw / filename,
                checksum,
            )
            extract(archive, corpus)
    train_root = raw / "train"
    validation = set((train_root / "validation_list.txt").read_text().splitlines())
    testing = set((train_root / "testing_list.txt").read_text().splitlines())
    validation.add("_background_noise_/running_tap.wav")
    rows = {"train": [], "validation": [], "test": []}
    for source in ("train", "test"):
        corpus = raw / source
        for path in sorted(corpus.rglob("*.wav")):
            relative = path.relative_to(corpus).as_posix()
            if source == "train" and relative in testing:
                continue
            split = (
                "test"
                if source == "test"
                else "validation"
                if relative in validation
                else "train"
            )
            word = path.parent.name
            label = (
                WORDS.index(word)
                if word in WORDS
                else 10
                if word == "_background_noise_"
                else 11
            )
            if word == "_background_noise_":
                for start in range(0, sf.info(path).frames - 16000, 8000):
                    rows[split].append(audio_record(path, label, start, 16000))
            else:
                rows[split].append(audio_record(path, label))
    for split, records in rows.items():
        (stage / f"{split}.json").write_text(json.dumps(records))
    return {
        "splits": {k: len(v) for k, v in rows.items()},
        "classes": 12,
        "labels": WORDS,
        "source": "Speech Commands v0.02; 12 classes; dedicated test archive; TFDS 0.0.3 split rules",
    }


def voxforge(raw, stage):
    archives = sorted(raw.rglob("*.tgz")) + sorted(raw.rglob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(
            "Download the VoxForge archives listed at https://storage.googleapis.com/tfds-data/downloads/voxforge/voxforge_urls.txt into raw/voxforge/<language>/Trunk/... before preparation."
        )
    index = download(
        "https://storage.googleapis.com/tfds-data/downloads/voxforge/voxforge_urls.txt",
        raw / "voxforge_urls.txt",
        "9cc218a5bce16640dc4eeb263aa423ce63ca59fc6587a9487dc397642b76af6c",
    )

    def archive_key(path):
        parts = Path(path).parts
        languages = [part for part in parts if part in LANGUAGES]
        if not languages:
            raise ValueError(f"Missing language directory: {path}")
        return languages[-1], Path(path).name

    expected = {
        archive_key(line.strip().strip("\"'"))
        for line in index.read_text().splitlines()
        if line.strip()
    }
    actual = {archive_key(path): path for path in archives}
    missing = expected - set(actual)
    if missing:
        raise FileNotFoundError(
            f"VoxForge corpus is incomplete: {len(missing)} archives from voxforge_urls.txt are missing"
        )
    archives = [actual[key] for key in sorted(expected)]
    groups = sorted({p.name.split("-")[0] for p in archives})
    if len(groups) < 10:
        raise ValueError(
            "VoxForge needs the full corpus, not a partial speaker collection"
        )
    np.random.RandomState(0).shuffle(groups)
    mapping = {
        speaker: "train"
        if i < int(0.7 * len(groups))
        else "validation"
        if i < int(0.8 * len(groups))
        else "test"
        for i, speaker in enumerate(groups)
    }
    rows = {"train": [], "validation": [], "test": []}
    seen = set()
    for archive in archives:
        candidates = [p for p in archive.parts if p in LANGUAGES]
        if not candidates:
            raise ValueError(
                f"Missing language directory (de/en/es/fr/it/ru): {archive}"
            )
        language = candidates[-1]
        speaker = archive.name.split("-")[0]
        target = (
            raw
            / "extracted"
            / language
            / archive.name.replace(".tar.gz", "").replace(".tgz", "")
        )
        extract(archive, target)
        for path in sorted(target.rglob("*.wav")):
            if str(path) in seen:
                raise ValueError("Duplicate VoxForge utterance")
            seen.add(str(path))
            rows[mapping[speaker]].append(
                audio_record(path, LANGUAGES.index(language), speaker=speaker)
            )
    for split, records in rows.items():
        if {r["label"] for r in records} != set(range(6)):
            raise ValueError(f"VoxForge {split} must contain all six languages")
        (stage / f"{split}.json").write_text(json.dumps(records))
    return {
        "splits": {k: len(v) for k, v in rows.items()},
        "classes": 6,
        "labels": LANGUAGES,
        "source": "VoxForge; speaker-disjoint 70/10/20 split, seed 0",
    }


def auxiliary(name, raw, stage):
    extensions = {".jpg", ".jpeg", ".png"} if name == "stylegan" else {".flac", ".wav"}
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in extensions)
    archive_name = (
        "stylegan-oriented.zip" if name == "stylegan" else "train-clean-100.tar.gz"
    )
    corpus = raw / "corpus"
    # An interrupted managed extraction can already contain valid media. Finish
    # the archive before accepting those files as a complete auxiliary corpus.
    if not files or (raw / archive_name).exists() or corpus.exists():
        if name == "stylegan":
            archive = download(
                "https://data.csail.mit.edu/noiselearning/zipped_data/small_scale/stylegan-oriented.zip",
                raw / archive_name,
            )
        else:
            archive = download(
                "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
                raw / archive_name,
                "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
            )
        extract(archive, corpus)
        files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in extensions)
    for path in files:
        if name == "stylegan":
            with Image.open(path) as image:
                image.verify()
        else:
            audio_record(path, 0)
    if not files:
        raise ValueError(f"No media found under {raw}")
    return {
        "media": [identity(p) for p in files],
        "source": "StyleGAN-Oriented small-scale"
        if name == "stylegan"
        else "LibriSpeech train-clean-100",
    }


def validate(name, root, full=True, *, directory=None):
    directory = Path(directory) if directory is not None else Path(root) / name
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("dataset") != name or manifest.get("schema") != 1:
        raise ValueError("Invalid dataset manifest")
    if len({item["name"] for item in manifest["files"]}) != len(manifest["files"]):
        raise ValueError("Duplicate prepared files")
    for item in manifest["files"]:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Prepared files must be inside the dataset directory")
        path = directory / item["name"]
        if path.stat().st_size != item["bytes"] or (
            full and digest(path) != item["sha256"]
        ):
            raise ValueError(f"Dataset changed: {path}")
    if full:
        for item in manifest.get("media", []):
            path = Path(item["path"])
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Media changed: {path}")
    if name in CLASSES:
        if manifest.get("classes") != CLASSES[name] or not {"train", "test"} <= set(
            manifest["splits"]
        ):
            raise ValueError("Dataset class count or required splits are missing")
        for split, count in manifest["splits"].items():
            if count < 1:
                raise ValueError(f"Empty {split} split")
            if name in AUDIO:
                rows = json.loads((directory / f"{split}.json").read_text())
                if len(rows) != count:
                    raise ValueError("Wrong audio split count")
                for row in rows:
                    if not 0 <= row["label"] < CLASSES[name]:
                        raise ValueError("Invalid audio label")
                    info = sf.info(row["path"])
                    if (
                        info.samplerate != 16000
                        or info.channels != 1
                        or info.subtype != "PCM_16"
                    ):
                        raise ValueError("Invalid audio encoding")
                    start, frames = row.get("start", 0), row.get("frames", -1)
                    if (
                        start < 0
                        or frames == 0
                        or start >= info.frames
                        or (frames > 0 and start + frames > info.frames)
                    ):
                        raise ValueError("Invalid audio slice")
            else:
                x = np.load(
                    directory / f"{split}_images.npy", mmap_mode="r", allow_pickle=False
                )
                y = np.load(
                    directory / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False
                )
                if (
                    x.dtype != np.uint8
                    or x.ndim != 4
                    or x.shape[1:]
                    != ((28, 28, 3) if name == "pathmnist" else (32, 32, 3))
                    or len(x) != count
                    or y.shape != (count,)
                    or y.dtype != np.int64
                    or y.min() < 0
                    or y.max() >= CLASSES[name]
                ):
                    raise ValueError("Invalid prepared images or labels")
    return manifest


def prepare(name, root, raw_root):
    root, raw_root = Path(root).resolve(), Path(raw_root).resolve()
    target, raw = root / name, raw_root / name
    if target.exists():
        return validate(name, root)
    raw.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary:
        stage = Path(temporary)
        metadata = (
            vision(name, raw, stage)
            if name not in AUDIO and name in CLASSES
            else speech(raw, stage)
            if name == "spcm"
            else voxforge(raw, stage)
            if name == "voxforge"
            else auxiliary(name, raw, stage)
        )
        if name in AUDIO:
            paths = sorted(
                {
                    row["path"]
                    for p in stage.glob("*.json")
                    for row in json.loads(p.read_text())
                }
            )
            metadata["media"] = [identity(p) for p in paths]
        files = [
            dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(stage.iterdir())
            if p.is_file()
        ]
        manifest = {"schema": 1, "dataset": name, "files": files, **metadata}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Publication occurs only after every source file and split was read.
        validate(name, root, full=False, directory=stage)
        os.rename(stage, target)
    return validate(name, root, full=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--raw-root", type=Path, default=Path("raw"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    failures = []
    for name in (args.dataset,) if args.dataset else DATASETS:
        try:
            manifest = (
                validate(name, args.root)
                if args.validate
                else prepare(name, args.root, args.raw_root)
            )
            print(
                f"{name}: ready ({manifest.get('splits', {'media': len(manifest.get('media', []))})})",
                flush=True,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append(name)
            print(f"{name}: {exc}", flush=True)
    if failures:
        parser.exit(2, "Preparation incomplete: " + ", ".join(failures) + "\n")
