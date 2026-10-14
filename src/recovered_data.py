"""Preparation helpers for the recovered audio and auxiliary-data adapters."""
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from data_utils import AugmentDataset


def audio_labels(dataset, name, expected_count):
    """Cache labels in the same deterministic TFDS order used by the loader."""
    folder=Path(os.environ['TFDS_DATA_DIR'])/name
    path=folder/'train_labels.npy'
    if path.exists():
        labels=np.load(path,allow_pickle=False).reshape(-1)
    else:
        labels=np.asarray([int(label.numpy()) for _,label in dataset],dtype=np.int64)
        if len(labels)!=expected_count:raise ValueError('TFDS example count does not match label enumeration')
        folder.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:np.save(f,labels)
    if len(labels)!=expected_count:raise ValueError('Cached labels do not match this TFDS dataset; choose a fresh data root')
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
