"""Preparation helpers for the recovered audio and auxiliary-data adapters."""
import os
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import numpy as np
import tensorflow as tf
from data_utils import AugmentDataset


def _file_identity(filename):
    """Identify a local prepared shard without scanning a large audio corpus."""
    if '://' in str(filename):
        raise ValueError('Remote shard metadata is not verified for caching')
    path = Path(filename).resolve(strict=True)
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError('Prepared shard must be a regular file')
    return {'path': str(path), 'size': metadata.st_size,
            'mtime_ns': metadata.st_mtime_ns, 'ctime_ns': metadata.st_ctime_ns,
            'device': metadata.st_dev, 'inode': metadata.st_ino}


def prepared_tfds_identity(info, split, reader_identity):
    """Return verified local shard metadata, or None to disable label caching.

    File instructions retain their order, slices, and shard example counts.
    The reader's explicit interleave settings and library versions also affect
    example order. Prepared files must remain unchanged while training runs.
    """
    try:
        if not isinstance(info.full_name, str) or not info.full_name:
            return None
        split_info = info.splits[split]
        files = []
        for instruction in split_info.file_instructions:
            filename = instruction.filename
            if not Path(filename).is_absolute() and '://' not in str(filename):
                filename = Path(info.data_dir) / filename
            files.append({'file': _file_identity(filename),
                          'skip': int(instruction.skip), 'take': int(instruction.take),
                          'examples_in_shard': int(instruction.examples_in_shard)})
        if not files:
            return None
        return {'schema': 1, 'full_name': info.full_name, 'split': str(split),
                'num_examples': int(split_info.num_examples),
                'reader': dict(reader_identity), 'files': files}
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        # Missing metadata, unsupported providers, and synthetic datasets must
        # enumerate labels rather than trust an unverified cache.
        return None


def _prepared_files_match(identity):
    try:
        return all(_file_identity(item['file']['path']) == item['file']
                   for item in identity['files'])
    except (KeyError, OSError, TypeError, ValueError):
        return False


def audio_labels(dataset, name, expected_count, source_identity=None):
    """Reuse labels only for an identified, unchanged prepared TFDS source."""
    path = None
    if source_identity is not None and _prepared_files_match(source_identity):
        key = hashlib.sha256(json.dumps(
            {'source': source_identity, 'expected_count': expected_count},
            sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        path = Path(os.environ['TFDS_DATA_DIR']) / name / f'train_labels-{key}.npy'
        if path.exists():
            try:
                labels = np.load(path, allow_pickle=False)
                if labels.shape == (expected_count,) and labels.dtype == np.int64:
                    return labels
            except (OSError, ValueError, EOFError):
                pass  # Rebuild a malformed cache from the identified source.

    # Old train_labels.npy files have no source identity. Leave them untouched.
    labels = np.asarray([int(label.numpy()) for _, label in dataset], dtype=np.int64)
    if len(labels) != expected_count:
        raise ValueError('TFDS example count does not match label enumeration')
    if path is not None:
        if not _prepared_files_match(source_identity):
            raise ValueError('Prepared TFDS shards changed during label enumeration')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.labels-', delete=False) as handle:
                temporary = Path(handle.name)
                np.save(handle, labels, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return labels


def load_ood_audio(directory,batch_size=128,seed=0,reshape_size=None):
    """Read user-prepared mono 16 kHz WAVs without scaling decoded floats twice."""
    paths=sorted(str(p) for p in Path(directory).rglob('*.wav'))
    if not paths:raise ValueError('The auxiliary audio directory has no .wav files')
    def prepare(path):
        waveform,rate=tf.audio.decode_wav(tf.io.read_file(path),desired_channels=1)
        with tf.control_dependencies([tf.debugging.assert_equal(rate,16000,message='Expected 16 kHz auxiliary WAV')]):
            waveform=tf.identity(waveform[:,0])
        return AugmentDataset.audio_train_prep(waveform,bins=40,to_float=False)
    return tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths),seed=seed).map(prepare).batch(batch_size).prefetch(1)


def load_ood_images(directory,batch_size=128,seed=0,reshape_size=(32,32)):
    dataset=tf.keras.utils.image_dataset_from_directory(str(directory),labels=None,
        shuffle=True,seed=seed,batch_size=batch_size,image_size=reshape_size)
    return dataset.map(lambda x:AugmentDataset.image_ood_prep(x,size=reshape_size))
