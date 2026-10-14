"""Prepared TFDS identity and cache behavior, without downloading a corpus."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pytest
import tensorflow as tf

from recovered_data import audio_labels


def prepared_info(root, full_name='audio/config/1.0.0'):
    root.mkdir(parents=True, exist_ok=True)
    instructions = []
    for index in range(2):
        shard = root / f'train-{index}.tfrecord'
        if not shard.exists():
            shard.write_bytes(b'generated shard')
        instructions.append(SimpleNamespace(
            filename=str(shard), skip=0, take=2, examples_in_shard=2))
    return SimpleNamespace(full_name=full_name, data_dir=root, splits={
        'train': SimpleNamespace(num_examples=4, file_instructions=instructions)})


def identity(info):
    from recovered_data import prepared_tfds_identity
    return prepared_tfds_identity(info, 'train', {
        'tfds_version': '4.9.4', 'tensorflow_version': tf.__version__,
        'interleave_cycle_length': 16, 'interleave_block_length': 16,
        'deterministic': True, 'shuffle_files': False})


class Labels:
    def __init__(self, values):
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        for value in self.values:
            yield None, tf.constant(value)


@pytest.mark.parametrize('change', ['version', 'order', 'slice', 'shard'])
def test_same_length_changed_source_never_reuses_old_labels(tmp_path, monkeypatch, change):
    monkeypatch.setenv('TFDS_DATA_DIR', str(tmp_path))
    info = prepared_info(tmp_path / 'prepared')
    original = identity(info)
    first = Labels([0, 1, 2, 3])
    np.testing.assert_array_equal(audio_labels(first, 'audio', 4, original), first.values)
    audio_labels(first, 'audio', 4, original)
    assert first.iterations == 1  # An unchanged prepared source reuses the cache.

    if change == 'version':
        info.full_name = 'audio/config/2.0.0'
    elif change == 'order':
        info.splits['train'].file_instructions.reverse()
    elif change == 'slice':
        info.splits['train'].file_instructions[0].skip = 1
    else:
        shard = Path(info.splits['train'].file_instructions[0].filename)
        before = shard.stat()
        shard.write_bytes(b'different shard')  # Same byte count, different contents.
        os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    changed = identity(info)
    assert changed != original
    second = Labels([3, 2, 1, 0])
    np.testing.assert_array_equal(audio_labels(second, 'audio', 4, changed), second.values)
    assert second.iterations == 1


def test_missing_identity_ignores_legacy_cache_without_deleting_it(tmp_path, monkeypatch):
    monkeypatch.setenv('TFDS_DATA_DIR', str(tmp_path))
    folder = tmp_path / 'audio'
    folder.mkdir()
    legacy = folder / 'train_labels.npy'
    np.save(legacy, np.array([9, 9]))
    ds = Labels([1, 2])
    np.testing.assert_array_equal(audio_labels(ds, 'audio', 2), ds.values)
    audio_labels(ds, 'audio', 2)
    assert ds.iterations == 2
    np.testing.assert_array_equal(np.load(legacy, allow_pickle=False), [9, 9])
    assert list(folder.iterdir()) == [legacy]


def test_unverifiable_source_has_no_identity(tmp_path):
    assert identity(SimpleNamespace()) is None
    info = prepared_info(tmp_path)
    Path(info.splits['train'].file_instructions[0].filename).unlink()
    assert identity(info) is None
    info.splits['train'].file_instructions[0].filename = 'gs://bucket/train-0'
    assert identity(info) is None


def test_failed_atomic_write_leaves_no_usable_partial_cache(tmp_path, monkeypatch):
    import recovered_data
    monkeypatch.setenv('TFDS_DATA_DIR', str(tmp_path))
    source = identity(prepared_info(tmp_path / 'prepared'))
    def interrupted_replace(*args):
        raise OSError('simulated interrupted publication')
    monkeypatch.setattr(recovered_data.os, 'replace', interrupted_replace)
    with pytest.raises(OSError, match='interrupted publication'):
        audio_labels(Labels([1, 2, 3, 4]), 'audio', 4, source)
    assert not list((tmp_path / 'audio').rglob('*'))


def test_source_modified_during_enumeration_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv('TFDS_DATA_DIR', str(tmp_path))
    info = prepared_info(tmp_path / 'prepared')
    source = identity(info)
    def changing_dataset():
        yield None, tf.constant(1)
        Path(info.splits['train'].file_instructions[0].filename).write_bytes(b'changed')
        yield None, tf.constant(2)
    with pytest.raises(ValueError, match='changed during label enumeration'):
        audio_labels(changing_dataset(), 'audio', 2, source)
    assert not (tmp_path / 'audio').exists()


def test_malformed_keyed_cache_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setenv('TFDS_DATA_DIR', str(tmp_path))
    source = identity(prepared_info(tmp_path / 'prepared'))
    ds = Labels([0, 1, 2, 3])
    audio_labels(ds, 'audio', 4, source)
    cache, = (tmp_path / 'audio').glob('train_labels-*.npy')
    cache.write_bytes(b'interrupted old write')
    np.testing.assert_array_equal(audio_labels(ds, 'audio', 4, source), ds.values)
    assert ds.iterations == 2
    np.testing.assert_array_equal(np.load(cache, allow_pickle=False), ds.values)


@pytest.mark.parametrize('name,method', [
    ('speech_commands', 'get_speech_commands'), ('voxforge', 'get_voxforge')])
def test_tfds_loader_passes_verified_identity_and_explicit_ordering(tmp_path, monkeypatch, name, method):
    from data_utils import LoadDataset
    info = prepared_info(tmp_path)
    info.features = {'label': SimpleNamespace(num_classes=3)}
    sentinel = object()
    def load(dataset_name, **kwargs):
        assert dataset_name == name and not kwargs['shuffle_files']
        config = kwargs['read_config']
        assert config.options.deterministic is True
        assert config.interleave_cycle_length == config.interleave_block_length == 16
        if name == 'voxforge':
            assert kwargs['download'] is False
        return sentinel, info
    monkeypatch.setitem(sys.modules, 'tensorflow_datasets', SimpleNamespace(
        __version__='4.9.4', ReadConfig=lambda **kw: SimpleNamespace(**kw), load=load))
    ds, metadata = getattr(LoadDataset, method)('train')
    assert ds is sentinel and metadata['num_examples'] == 4
    assert metadata['source_identity'] == identity(info)


@pytest.mark.parametrize('name,loader', [('spcm', 'get_spcm'), ('voxforge', 'get_voxforge')])
def test_client_adapter_forwards_identity_to_cache(tmp_path, monkeypatch, name, loader):
    import data
    import recovered_data
    monkeypatch.setenv('TFDS_DATA_DIR', str(tmp_path))
    source = identity(prepared_info(tmp_path / 'prepared'))
    ds = tf.data.Dataset.from_tensor_slices((np.zeros((8, 16000), np.int64), np.arange(8) % 2))
    monkeypatch.setitem(data.DATASETS[name], 'get_data_fn', lambda **kw: (
        ds, {'num_classes': 2, 'num_examples': 8, 'source_identity': source}))
    calls = []
    def capture(dataset, cache_name, count, source_identity=None):
        calls.append((dataset, cache_name, count, source_identity))
        return np.arange(8) % 2
    monkeypatch.setattr(recovered_data, 'audio_labels', capture)
    getattr(data, loader)(id=0, num_clients=2, batch_size=2, valid_split=0)
    assert calls == [(ds, 'speech_commands' if name == 'spcm' else name, 8, source)]
