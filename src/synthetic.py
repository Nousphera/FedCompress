"""Bounded independent synthetic client and server data; no paper data."""
import numpy as np
import tensorflow as tf


def get_data(id=0,num_clients=2,return_eval_ds=False,batch_size=8,seed=42):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(40,16,16,3)).astype(np.float32)
    y=np.arange(40)%2
    if return_eval_ds:
        return (tf.data.Dataset.from_tensor_slices((x[-8:],y[-8:])).batch(batch_size),None),2,8
    ids=np.array_split(np.arange(32),num_clients)[id]
    if len(ids)<4: raise ValueError('Synthetic fixture supports at most eight clients')
    train,valid=ids[:-2],ids[-2:]
    make=lambda ids: tf.data.Dataset.from_tensor_slices((x[ids],y[ids])).batch(batch_size)
    return (make(train),make(valid)),2,(len(train),len(valid))


def get_ood(batch_size=8,seed=42,reshape_size=(16,16)):
    # Independently sampled unlabeled data, explicitly a smoke fixture.
    x=np.random.default_rng(seed+10000).normal(size=(8,*reshape_size,3)).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
