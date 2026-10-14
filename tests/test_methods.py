import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import utils


def test_effective_rank_is_finite_for_rank_deficient_embeddings():
    assert utils.compute_score(np.eye(3, dtype=np.float32)) == pytest.approx(3, rel=1e-5)
    assert utils.compute_score(np.ones((4,3), dtype=np.float32)) == pytest.approx(1, rel=1e-5)
    assert np.isfinite(utils.compute_score(np.zeros((4,3), dtype=np.float32)))


def test_score_is_sample_weighted_on_client_scalars():
    result = utils.weighted_average_train_metrics([
        (2, {'train_loss':1.,'train_accuracy':.5,'val_score':2.}),
        (6, {'train_loss':2.,'train_accuracy':.7,'val_score':4.})])
    assert result['val_score'] == pytest.approx(3.5)


def test_compression_schedule_includes_every_interval_and_final():
    assert utils.set_compression_rnds(2, 6) == [0, 2, 4, 6]
    with pytest.raises(ValueError): utils.set_compression_rnds(0, 3)


def test_cluster_update_requires_score_plateau_not_accuracy():
    plateau = pd.DataFrame({'val_score':[2.]*7, 'val_accuracy':np.arange(7), 'nb_clusters':[8]*7})
    assert utils.compression_flag(plateau, patience=3, window=3)
    rising = plateau.copy(); rising['val_score'] = np.arange(7, dtype=float)
    assert not utils.compression_flag(rising, patience=3, window=3)
    plateau['nb_clusters'] = 20
    assert not utils.compression_flag(plateau)


def test_client_and_server_clustering_keep_centroid_bound_and_teacher():
    import compress
    x = tf.keras.Input((8,8,1))
    y = tf.keras.layers.Conv2D(4,3, name='conv')(x)
    y = tf.keras.layers.GlobalAveragePooling2D(name='GMP_layer')(y)
    base = tf.keras.Model(x, tf.keras.layers.Dense(2)(y))
    loader = lambda input_shape, num_classes: tf.keras.models.clone_model(base)
    data = tf.data.Dataset.from_tensor_slices((tf.ones((4,8,8,1)),tf.constant([0,1,0,1]))).batch(2)
    client, history = compress.self_compress_with_data(base, data, nb_clusters=2, epochs=1)
    assert len(np.unique(client.get_layer('conv').get_weights()[0])) <= 2
    teacher_before = [w.copy() for w in client.get_weights()]
    ood_loader = lambda **kw: data.map(lambda x,y:x)
    student = compress.self_compress(client, 2, loader, ood_loader, data_shape=(None,8,8,1),nb_clusters=2,epochs=1,batch_size=2)
    assert len(np.unique(student.get_layer('conv').get_weights()[0])) <= 2
    for a,b in zip(teacher_before,client.get_weights()): np.testing.assert_array_equal(a,b)
    assert student.loss is not None
    assert np.isfinite(student.evaluate(data,verbose=0)[0])


def test_wire_payload_is_lossless_and_compresses_clustered_weights():
    arrays=[np.zeros((1024,),np.float32),np.array([1,2,3],np.int64)]
    payload=utils.pack_parameters(arrays)
    assert payload[0].nbytes < sum(a.nbytes for a in arrays)
    for a,b in zip(arrays,utils.unpack_parameters(payload)):
        np.testing.assert_array_equal(a,b)
        assert a.dtype==b.dtype


def test_tiny_positive_participation_schedules_one_client():
    from types import SimpleNamespace
    from server import _Server
    dataset = tf.data.Dataset.from_tensor_slices((tf.zeros((2, 4)), tf.constant([0, 1]))).batch(2)
    def model_loader(input_shape, num_classes):
        x = tf.keras.Input(input_shape)
        return tf.keras.Model(x, tf.keras.layers.Dense(num_classes)(x))
    loader = lambda **kw: ((dataset, dataset), 2, 2)
    server = _Server(model_loader, loader, num_rounds=1, num_clients=2, participation=.01)
    manager = SimpleNamespace(num_available=lambda: 2,
        sample=lambda num_clients, min_num_clients: [SimpleNamespace(cid=str(i)) for i in range(num_clients)])
    instructions = server.strategy.configure_fit(1, server.strategy.initial_parameters, manager)
    assert len(instructions) == 1
