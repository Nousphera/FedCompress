import flwr as fl
import tensorflow as tf
import compress
import utils

class _Client(fl.client.NumPyClient):

	def __init__(self, cid, num_clients, model_loader, data_loader, batch_size, client_compression, seed):
		self.cid = cid
		self.data_loader = data_loader
		self.num_clients = num_clients
		self.batch_size = batch_size
		self.seed = seed
		(self.data,self.val_data), self.num_classes, (self.num_samples,self.num_val_samples) = data_loader(cid, num_clients, batch_size=batch_size, seed=seed)
		self.model_loader = model_loader
		self.input_shape = self.data.element_spec[0].shape
		self.client_compression = client_compression
		self.combined_data = False

	def set_parameters(self, parameters, config):
		""" Set model weights """
		if not hasattr(self, 'model'):
			self.model = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes)

		self.model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
			loss=tf.keras.losses.SparseCategoricalCrossentropy(name='loss', from_logits=True),
			metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
		)

		if parameters is not None:
			self.model.set_weights(utils.unpack_parameters(parameters))

	def get_parameters(self, config={}):
		""" Get model weights """
		return self.model.get_weights()

	def fit(self, parameters, config):
		results = {}
		if config['epochs'] <= 0 and (not self.client_compression or config['compression_epochs'] <= 0):
			raise ValueError('At least one local training epoch is required')

		# Set parameters
		self.set_parameters(parameters, config)

		# Perform standard local update step
		if config['epochs'] != 0:
			h = self.model.fit(self.data, epochs=config['epochs'], verbose=0)
			print(f"[Client {self.cid}] - Accurary: {float(h.history['accuracy'][-1]):0.4f}.")

		# Perform compressed local update step
		if self.client_compression:
			self.model, h = compress.self_compress_with_data(model=self.model,
                data=self.data,
                epochs=config['compression_epochs'],
				nb_clusters=config['num_clusters'],
				learning_rate=config['compression_lr'],
			)
			print(f"[Client {self.cid}] - Compr. Accurary: {float(h.history['accuracy'][-1]):0.4f} (Clusters: {config['num_clusters']}).")
		# Store results
		results['model_size'] = utils.get_gzipped_model_size_from_model(self.model)
		results['train_loss'] = float(h.history['loss'][-1])
		results['train_accuracy'] = float(h.history['accuracy'][-1])

		# Measure validation accuracy for elbow method.
		metrics = self.model.evaluate(self.val_data,verbose=0)
		results['val_loss'] = float(metrics[0])
		results['val_accuracy'] = float(metrics[1])
		results['num_val_samples'] = int(self.num_val_samples)

		# Measure model embeddings
		results['val_score'] = utils.compute_score(utils.compute_embeddings(self.model_loader, self.val_data, self.num_classes, weights=self.get_parameters()))

		return utils.pack_parameters(self.get_parameters()), self.num_samples, results

	def evaluate(self, parameters, config):
		raise NotImplementedError('Client-side evaluation is not implemented!')