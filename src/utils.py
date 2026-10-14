import os
import zipfile
import tempfile
import errno
import contextlib
import pickle
from time import sleep

" Compute embedding from GMP layer."
def compute_embeddings(model_loader, data, num_classes, weights):
	import tensorflow as tf
	# Need to create a new model to avoid errors.
	model = model_loader(input_shape=data.element_spec[0].shape[1:], num_classes=num_classes)
	model.set_weights(weights)
	# Set new model up to 'GMP_layer'.
	new_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("GMP_layer").output)
	new_model.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
	# Compute embeddings
	embeddings = new_model.predict(data, verbose=0)
	return embeddings

" Compute embedding score from SVD analysis. "
def compute_score(embeddings, normalize=True):
	import numpy as np
	embeddings = np.asarray(embeddings, dtype=np.float64)
	if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
		raise ValueError('Expected a finite matrix of embeddings')
	if normalize:
		embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
	s = np.linalg.svd(embeddings, compute_uv=False)
	if s.sum() <= 1e-12: return 0.0
	p = s / s.sum()
	return float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-7)))))

" Compute energy score on data."
def compute_energy_score(model, data, temp=-1.0):
	import numpy as np
	import tensorflow as tf
	energy_score = []
	for (x,_) in data:
		preds = model(x,training=False)
		score = tf.math.reduce_logsumexp(preds/temp, axis=-1)
		energy_score.extend(score.numpy())
	return np.asarray(energy_score).mean()

"Agreggate metrics from clients fit results."
def weighted_average_train_metrics(metrics):
	results = {}
	# Store metrics names
	metrics_keys = [m.keys() for _, m in metrics]
	# Compute standard metrics
	train_examples = [num_examples for num_examples, _ in metrics]
	results['train_loss'] = sum([num_examples * m["train_loss"] for num_examples, m in metrics]) / sum(train_examples)
	results['train_accuracy'] = sum([num_examples * m["train_accuracy"] for num_examples, m in metrics]) / sum(train_examples)
	# Compute validation metrics
	if all([set(['val_loss', 'val_accuracy', 'num_val_samples']) <= set(list(keys)) for keys in metrics_keys]):
		val_examples = [m['num_val_samples'] for _, m in metrics]
		results['val_loss'] = sum([num_examples * m["val_loss"] for num_examples,(_,m) in zip(val_examples,metrics)]) / sum(val_examples)
		results['val_accuracy'] = sum([num_examples * m["val_accuracy"] for num_examples,(_,m) in zip(val_examples,metrics)]) / sum(val_examples)
	# Compute model size
	if all('model_size' in keys for keys in metrics_keys):
		results['models_sizes'] = [m["model_size"] for _, m in metrics]
		results['models_size'] = sum(results['models_sizes'])/len(results['models_sizes'])
	# Paper: compute effective rank locally, then sample-weight client scores.
	if all('val_score' in keys for keys in metrics_keys):
		results['val_score'] = sum(n * m['val_score'] for n,m in metrics) / sum(train_examples)

	return results

" Extract train metrics for a given round."
def set_train_metrics(config={}):
	results={}
	# Store metrics names
	config_keys = list(config.keys())
	# Store train loss/accuracy
	if set(['train_loss','train_accuracy']) <= set(config_keys):
		results['train_loss'] = config['train_loss']
		results['train_accuracy'] = config['train_accuracy']
	# Store validation loss/accuracy
	if set(['val_loss','val_accuracy']) <= set(config_keys):
		results['val_loss'] = config['val_loss']
		results['val_accuracy'] = config['val_accuracy']
	# Store train model size(s)
	if set(['models_size','models_sizes']) <= set(config_keys):
		results['client_model_size'] = config['models_size']
		results['clients_model_size'] = config['models_sizes']
	# Score embedding scores(s)
	if set(['val_score']) <= set(config_keys):
		results['val_score'] = config['val_score']
	if set(['odd_score']) <= set(config_keys):
		results['odd_score'] = config['odd_score']
	return results

" Set number of federated rounds to perform server-side compression."
def set_compression_rnds(compression_step, num_rounds):
	if compression_step <= 0 or num_rounds < 0:
		raise ValueError('compression_step must be positive and num_rounds nonnegative')
	return sorted(set([0, num_rounds, *range(compression_step, num_rounds+1, compression_step)]))

" Check mechanism for increasing number of clusters."
def compression_flag(df, init_num_clusters=8, max_num_clusters=20, init_rounds=1, patience=3, window=3, base=2, limit=0.01):
	"""Increase capacity after P consecutive stalled rolling score increments."""
	if window < 1 or patience < 1: raise ValueError('window and patience must be positive')
	if len(df) < max(init_rounds, window+patience): return False
	current = df['nb_clusters'].iloc[-1]
	if current >= max_num_clusters or (df['nb_clusters'] == current).sum() < patience: return False
	progress = df['val_score'].rolling(window=window, min_periods=window).mean().diff().iloc[-patience:]
	return bool(progress.notna().all() and (progress <= limit).all())

" Get number of cluster between two boundaries based on current round."
def get_num_clusters(rnd, init_num_clusters, compression_rnds, min_num_clusters=1):
	# Find closest (upper) index of rnd in compression_rnds
	upper_closest = lambda r: compression_rnds.index(min([i for i in compression_rnds if i>=r], key=lambda x:abs(x-r)))
	# Decrease step of clusters per new compression
	decrease_step = (init_num_clusters-min_num_clusters)/len(compression_rnds)
	# Number of cluster in rnd
	num_clusters = int(init_num_clusters-(upper_closest(rnd)+1)*decrease_step)
	return (max(num_clusters, min_num_clusters))

" None or Str datatype."
def none_or_str(value):
    if value == 'None':
        return None
    return value

" None or Int datatype."
def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

" Create if exist without warning."
def silentcreate(filename):
	try:
		os.makedirs(filename)
	except FileExistsError:
		pass

" Remove if exist without warning."
def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def setup(id='0', mem=12000):
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = id
	import tensorflow as tf
	gpus = tf.config.list_physical_devices("GPU")
	if gpus:
		try:
			tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=mem)])
		except RuntimeError as e:
			print(e)
	else:
		print('No gpu available')

def get_gzipped_model_size(file):
	fd, zipped_file = tempfile.mkstemp(".zip")
	os.close(fd)
	try:
		with zipfile.ZipFile(zipped_file, "w", compression=zipfile.ZIP_DEFLATED) as f:
			f.write(file)
		return os.path.getsize(zipped_file)/1000
	finally:
		silentremove(zipped_file)

def get_gzipped_model_size_from_model(model):
	fd, file = tempfile.mkstemp(".h5")
	os.close(fd)
	try:
		with contextlib.redirect_stdout(None):
			model.save(file, include_optimizer=False)
			return get_gzipped_model_size(file)
	finally:
		silentremove(file)

def store_history(history, args, store_dir_fn=None, score_files=None):
	import numpy as np
	import pandas as pd

	exceptions = ['num_clusters','model_size','accuracy','compression','compressed_model_size','compressed_accuracy']
	data = history.metrics_centralized
	losses = history.losses_centralized
	data['loss'] = list(zip(*losses))[1][1:]
	data['rnd'] = list(zip(*losses))[0][1:]
	for key in data.keys():
		if key in ['rnd','loss']: break
		data[key] = list(zip(*data[key]))[1][(1 if key in exceptions else 0):]

	# Add scores
	if score_files:
		types = ['val_score','val_norm_score','ood_score','odd_norm_score']
		names = ['embeddings','w_embeddings','ood_embeddings','w_ood_embeddings']
		rnd = lambda x: int(x.split('_')[-1].split('.')[0])
		# Initialize
		for t in types: data[t] = {r:[] for r in data['rnd']}
		# Load
		for f in score_files:
			temp = pickle.load(open(f,'rb'))
			for i,j in zip(names,types):
				if i in temp.keys(): data[j][rnd(f)].extend(temp[i].numpy() if not isinstance(temp[i],np.ndarray) else temp[i])
		# Combine
		for i in types:
			data[i] = [compute_score(np.asarray(data[i][j])) for j in data['rnd']]

	df = pd.DataFrame(data)
	df = df.rename(columns={"accuracy": "test_accuracy", "loss": "test_loss"})

	if (not args.server_compression) and (not args.client_compression):
		df = df.drop('num_clusters', axis=1)

	if store_dir_fn is not None:
		# Store results
		df.to_pickle(store_dir_fn(('metrics','pkl')))
		pickle.dump(args, open(store_dir_fn(('args','pkl')), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	return df

def create_dataloader_fn(ood=False):
	import data
	if ood:
		return {"cifar10": data.get_stylegan, "spcm": data.get_librispeech,}
	return {"cifar10": data.get_cifar10, "spcm": data.get_spcm,}


def pack_parameters(parameters):
	"""Lossless NPZ/DEFLATE payload carried as one standard uint8 ndarray."""
	import io
	import numpy as np
	stream = io.BytesIO()
	np.savez_compressed(stream, *parameters)
	return [np.frombuffer(b'FCP1' + stream.getvalue(), dtype=np.uint8).copy()]


def unpack_parameters(parameters):
	import io
	import numpy as np
	if len(parameters) != 1 or parameters[0].dtype != np.uint8:
		return parameters
	payload = parameters[0].tobytes()
	if not payload.startswith(b'FCP1'): raise ValueError('Invalid FedCompress payload header')
	with np.load(io.BytesIO(payload[4:]), allow_pickle=False) as archive:
		return [archive[f'arr_{i}'] for i in range(len(archive.files))]
