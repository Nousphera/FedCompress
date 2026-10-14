"""Actual recovered vision/audio runs with supplied generated data and OOD files."""
import importlib.util
import os
from pathlib import Path
import sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pytest
import tensorflow as tf


@pytest.mark.parametrize('dataset',['pathmnist','voxforge'])
def test_recovered_adapter_runs_client_clustering_and_server_distillation(tmp_path,monkeypatch,dataset):
    import data
    auxiliary=tmp_path/'auxiliary';auxiliary.mkdir()
    if dataset=='pathmnist':
        folder=tmp_path/'raw/pathmnist';folder.mkdir(parents=True)
        rng=np.random.default_rng(3)
        np.savez(folder/'pathmnist.npz',train_images=rng.integers(0,256,(32,28,28,3),dtype=np.uint8),
            train_labels=(np.arange(32)%9)[:,None],test_images=rng.integers(0,256,(4,28,28,3),dtype=np.uint8),test_labels=np.arange(4)[:,None])
        for i in range(2):tf.io.write_file(str(auxiliary/f'{i}.png'),tf.io.encode_png(tf.zeros((28,28,3),tf.uint8)))
    else:
        wave=(np.sin(np.arange(16000)/100)*10000).astype(np.int64)
        ds=tf.data.Dataset.from_tensor_slices((np.stack([wave]*16),np.arange(16)%2))
        monkeypatch.setitem(data.DATASETS[dataset],'get_data_fn',lambda **kw:(ds,{'num_classes':6,'num_examples':16}))
        tf.io.write_file(str(auxiliary/'example.wav'),tf.audio.encode_wav(tf.cast(wave[:,None],tf.float32)/32767,16000))
    spec=importlib.util.spec_from_file_location('recovered_runner',ROOT/'train.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    report=runner.main(['--dataset',dataset,'--data-root',str(tmp_path),'--ood-dir',str(auxiliary),
        '--clients','2','--rounds','1','--epochs','1','--warmup-epochs','0','--server-epochs','1',
        '--batch-size','4','--min-clusters','2','--max-clusters','3','--output',str(tmp_path/'run')])
    row=report['rounds'][0]
    assert row['server_self_compression'] and row['parameters_changed']
    assert row['upstream_protobuf_bytes']>0 and np.isfinite(row['loss'])
    assert (tmp_path/'run/compressed.h5').is_file()
