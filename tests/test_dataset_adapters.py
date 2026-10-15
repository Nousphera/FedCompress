"""Dataset and auxiliary-data adapters on generated files and waveforms."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
import tensorflow as tf


def test_pathmnist_loader_uses_supplied_npz_and_original_28_pixel_preprocessing(tmp_path, monkeypatch):
    import data
    monkeypatch.setenv('TFDS_DATA_DIR',str(tmp_path))
    folder=tmp_path/'raw/pathmnist';folder.mkdir(parents=True)
    np.savez(folder/'pathmnist.npz',train_images=np.zeros((32,28,28,3),np.uint8),
        train_labels=(np.arange(32)%9)[:,None],test_images=np.zeros((4,28,28,3),np.uint8),test_labels=np.arange(4)[:,None])
    (train,valid),classes,counts=data.get_pathmnist(id=0,num_clients=2,batch_size=4)
    assert next(iter(train))[0].shape[1:]==(28,28,3)
    assert classes==9 and sum(len(y) for _,y in train)==counts[0]
    (test,_),classes,count=data.get_pathmnist(return_eval_ds=True)
    assert count==4 and next(iter(test))[0].shape[1:]==(28,28,3)


def test_cifar100_adapter_uses_100_class_labels(monkeypatch):
    import data
    arrays=(np.zeros((32,32,32,3),np.uint8),np.arange(32)[:,None])
    monkeypatch.setattr(tf.keras.datasets.cifar100,'load_data',lambda **kw:(arrays,arrays))
    (_, _),classes,count=data.get_cifar100(return_eval_ds=True)
    assert classes==100 and count==32


@pytest.mark.parametrize('name',['spcm','voxforge'])
def test_audio_adapter_trains_real_yamnet_update_on_generated_pcm(tmp_path,monkeypatch,name):
    import data
    from network import get_model
    monkeypatch.setenv('TFDS_DATA_DIR',str(tmp_path))
    waves=np.stack([(np.sin(np.arange(16000)*(i%2+1)/100)*10000).astype(np.int64) for i in range(24)])
    labels=np.arange(24)%2
    ds=tf.data.Dataset.from_tensor_slices((waves,labels))
    def source(split,with_info=True):return ds,{'num_classes':12 if name=='spcm' else 6,'num_examples':24}
    monkeypatch.setitem(data.DATASETS[name],'get_data_fn',source)
    loader=data.get_spcm if name=='spcm' else data.get_voxforge
    (train,valid),classes,counts=loader(id=0,num_clients=2,batch_size=4)
    x,y=next(iter(train))
    assert x.shape[1:]==(98,40,1) and sum(len(t) for _,t in train)==counts[0]
    model=get_model('yamnet')(input_shape=x.shape[1:],num_classes=classes)
    before=model.get_weights()[-1].copy()
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    loss=model.train_on_batch(x,y)
    assert np.isfinite(loss) and not np.array_equal(before,model.get_weights()[-1])


def test_unlabeled_wav_loader_preserves_float_audio_scale_and_partial_batch(tmp_path):
    from dataset_adapters import load_ood_audio
    wave=tf.sin(tf.range(16000,dtype=tf.float32)/100)[:,None]*.5
    tf.io.write_file(str(tmp_path/'sample.wav'),tf.audio.encode_wav(wave,16000))
    ds=load_ood_audio(tmp_path,batch_size=4,seed=3)
    batch=next(iter(ds))
    assert batch.shape==(1,98,40,1) and np.isfinite(batch.numpy()).all()
    # A second 1/32767 division would move the loudest log-mel below -5.
    assert float(tf.reduce_max(batch))>0
