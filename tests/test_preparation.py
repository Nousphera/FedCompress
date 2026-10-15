"""Preparation contracts tested with bounded files, never a corpus download."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts/prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def fixture(path, shape=(32,32,3), classes=10):
    rng = np.random.default_rng(7)
    np.savez(path, train_images=rng.integers(0,256,(40,*shape),dtype=np.uint8),
             train_labels=np.arange(40) % classes,
             test_images=rng.integers(0,256,(10,*shape),dtype=np.uint8), test_labels=np.arange(10)%classes)


def test_prepare_import_validate_no_overwrite_and_tamper(tmp_path):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    args = ["--dataset", "cifar10", "--root", str(target)]
    first = prep.main([*args, "--source", str(source)])
    assert first["metadata"]["train_rows"] == 40
    assert prep.main([*args, "--validate"]) == first
    with pytest.raises(FileExistsError):
        prep.main([*args, "--source", str(source)])
    with (target / "cifar10.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        prep.main([*args, "--validate"])


@pytest.mark.parametrize("shape,classes", [((28,28,3),10),((32,32,3),11)])
def test_reject_wrong_shape_or_out_of_range_labels(tmp_path, shape, classes):
    source = tmp_path / "invalid.npz"
    fixture(source, shape, classes)
    with pytest.raises(ValueError):
        prep.main(["--dataset","cifar10","--source",str(source),"--root",str(tmp_path/"target")])
    assert not (tmp_path/"target/cifar10.npz").exists()


def test_prepared_arrays_reach_training_loader(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    fixture(source)
    root = tmp_path / "data"
    prep.main(["--dataset","cifar10","--source",str(source),"--root",str(root)])
    monkeypatch.setenv("TFDS_DATA_DIR", str(root))
    from data import get_cifar10
    (ds, val), classes, counts = get_cifar10(id=0, num_clients=2, batch_size=4, seed=42)
    x, y = next(iter(ds))
    assert classes == 10 and counts[0] > 0 and x.shape[1:] == (32,32,3)


def test_public_archive_extraction_blocks_traversal(tmp_path):
    import io
    import tarfile
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as out:
        info = tarfile.TarInfo("../escaped")
        info.size = 1
        out.addfile(info, io.BytesIO(b"x"))
    with tarfile.open(archive) as source:
        with pytest.raises(tarfile.FilterError):
            source.extractall(tmp_path / "safe")
    assert not (tmp_path / "escaped").exists()


def test_existing_manifest_cannot_leave_a_half_published_array(tmp_path):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    target.mkdir()
    saved = target / "cifar10.manifest.json"
    saved.write_text('{"existing":true}')
    with pytest.raises((ValueError, FileExistsError)):
        prep.main(["--dataset", "cifar10", "--source", str(source), "--root", str(target)])
    assert not (target / "cifar10.npz").exists()
    assert saved.read_text() == '{"existing":true}'


def test_auxiliary_import_is_bounded_and_failed_decode_leaves_no_output(tmp_path):
    from PIL import Image
    source = tmp_path / "raw"
    source.mkdir()
    Image.new("RGB", (40, 40)).save(source / "image.jpg")
    (source / "unrelated-private.txt").write_text("not training input")
    target = tmp_path / "prepared"
    first = prep.main(["--dataset", "ood-images", "--source", str(source), "--root", str(target)])
    assert prep.main(["--dataset", "ood-images", "--root", str(target), "--validate"]) == first
    from recovered_data import load_ood_images
    assert next(iter(load_ood_images(target, batch_size=1))).shape == (1, 32, 32, 3)
    assert not (target / "unrelated-private.txt").exists()
    bad = source / "image.jpg"
    bad.write_bytes(bad.read_bytes()[:-20])
    failed = tmp_path / "failed"
    with pytest.raises(OSError):
        prep.main(["--dataset", "ood-images", "--source", str(source), "--root", str(failed)])
    assert not failed.exists()


def test_array_publication_failure_rolls_back_both_files(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    real_link = prep.os.link
    calls = []
    def fail_manifest(source, destination):
        calls.append(destination)
        if str(destination).endswith(".manifest.json"):
            raise OSError("publication failed")
        return real_link(source, destination)
    monkeypatch.setattr(prep.os, "link", fail_manifest)
    with pytest.raises(OSError, match="publication failed"):
        prep.main(["--dataset", "cifar10", "--source", str(source), "--root", str(target)])
    assert len(calls) == 2
    assert not (target / "cifar10.npz").exists()
    assert not (target / "cifar10.manifest.json").exists()
    assert not list(target.glob(".prepare-*"))


def test_tfds_import_relocation_preserves_reader_and_manifest(tmp_path, monkeypatch):
    import json
    import tensorflow as tf
    tfds = pytest.importorskip('tensorflow_datasets')

    class SpeechCommands(tfds.core.GeneratorBasedBuilder, skip_registration=True):
        VERSION = tfds.core.Version('0.0.3')
        def _info(self):
            return tfds.core.DatasetInfo(builder=self, features=tfds.features.FeaturesDict({
                'audio': tfds.features.Tensor(shape=(160,), dtype=np.int16),
                'label': tfds.features.ClassLabel(num_classes=12),
            }))
        def _split_generators(self, dl_manager):
            return {'train': self._generate_examples(3), 'test': self._generate_examples(2)}
        def _generate_examples(self, count):
            for index in range(count):
                yield index, {'audio': np.full(160, index, np.int16), 'label': index}

    monkeypatch.setattr(tfds.core.DatasetInfo, 'initialize_from_bucket', lambda self: None)
    source = tmp_path / 'source'
    builder = SpeechCommands(data_dir=str(source))
    builder.download_and_prepare(download_config=tfds.download.DownloadConfig(try_download_gcs=False))
    (source / 'unrelated-private.txt').write_text('not dataset input')
    def load_prepared(name, data_dir):
        assert name == 'speech_commands'
        return SpeechCommands(data_dir=data_dir)
    monkeypatch.setattr(tfds, 'builder', load_prepared)
    target = tmp_path / 'imported'
    args = ['--dataset', 'spcm', '--root', str(target)]
    first = prep.main([*args, '--source', str(source)])
    assert first['metadata']['splits'] == {'train': 3, 'test': 2}
    assert prep.main([*args, '--validate']) == first
    assert '.prepare-' not in json.dumps(first) and str(source) not in json.dumps(first)
    assert not (target / 'unrelated-private.txt').exists()
    assert not list(tmp_path.glob('.prepare-*'))
    moved = SpeechCommands(data_dir=str(target))
    original = [int(x['label']) for x in tfds.as_numpy(builder.as_dataset(split='test', shuffle_files=False))]
    assert [int(x['label']) for x in tfds.as_numpy(moved.as_dataset(split='test', shuffle_files=False))] == original


def test_auxiliary_audio_import_validate_and_actual_reader(tmp_path):
    import wave
    from recovered_data import load_ood_audio
    source = tmp_path / 'source'
    source.mkdir()
    with wave.open(str(source / 'fixture.wav'), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(np.zeros(16000, np.int16).tobytes())
    (source / 'unrelated.txt').write_text('not consumed')
    target = tmp_path / 'imported'
    args = ['--dataset', 'ood-audio', '--root', str(target)]
    first = prep.main([*args, '--source', str(source)])
    assert prep.main([*args, '--validate']) == first
    assert not (target / 'unrelated.txt').exists()
    assert next(iter(load_ood_audio(target, batch_size=1))).shape[0] == 1


@pytest.mark.parametrize('width,suffix', [(1, '.wav'), (3, '.wav'), (2, '.WAV')])
def test_auxiliary_audio_rejects_files_unsupported_by_training_reader(tmp_path, width, suffix):
    import wave
    source = tmp_path / 'source'
    source.mkdir()
    with wave.open(str(source / ('fixture' + suffix)), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(width); stream.setframerate(16000)
        stream.writeframes(bytes(16000 * width))
    target = tmp_path / 'imported'
    with pytest.raises(ValueError, match='16-bit|lowercase'):
        prep.main(['--dataset', 'ood-audio', '--source', str(source), '--root', str(target)])
    assert not target.exists()
