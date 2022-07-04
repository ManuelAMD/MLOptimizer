from __future__ import absolute_import, division, print_function, unicode_literals
import time
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, regularizers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from app.common.inception_module import InceptionV1ModuleBN
from app.common.search_space import *
from app.common.dataset import Dataset
from app.common.model_communication import *
from system_parameters import SystemParameters as SP
from app.common.Callbacks import EndEpoch
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
print(gpu_devices)
for device in gpu_devices:
	tf.config.experimental.set_memory_growth(device, True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Model:
	def __init__(self, model_training_request: ModelTrainingRequest, dataset: Dataset, socket = None):
		if socket != None:
			SocketCommunication.isSocket = socket.isSocket
		
		self.id = model_training_request.id
		self.experiment_id = model_training_request.experiment_id
		self.training_type = model_training_request.training_type
		self.search_space_type: SearchSpaceType = SearchSpaceType(model_training_request.search_space_type)
		self.model_params = model_training_request.architecture
		self.epochs = model_training_request.epochs
		self.early_stopping_patience = model_training_request.early_stopping_patience
		self.is_partial_training = model_training_request.is_partial_training
		self.model: tf.keras.Model
		self.dataset: Dataset = dataset

	def build_model_CPU(self, input_shape: tuple, class_count: int):
		try:
			strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
			with strategy.scope():
				m = self.build_model(input_shape, class_count)
		except ValueError as e:
			logging.warning(e)
		return m

	def build_model(self, input_shape: tuple, class_count: int):
		if self.search_space_type == SearchSpaceType.IMAGE:
			return self.build_image_model(self.model_params, input_shape, class_count)
		elif self.search_space_type == SearchSpaceType.REGRESSION:
			return self.build_regression_model(self.model_params, input_shape, class_count)
		elif self.search_space_type == SearchSpaceType.TIME_SERIES:
			return self.build_time_series_model(self.model_params, input_shape, class_count)
		elif self.search_space_type == SearchSpaceType.IMAGE_TIME_SERIES:
			return self.build_image_time_series_model(self.model_params, input_shape, class_count)

	def remove_img(self, path):
		if os.path.exists(path):
			os.remove(path)
			return True
		return False

	def create_model_image(self, route):
		model = self.build_model(self.dataset.get_input_shape(), self.dataset.get_classes_count())
		self.remove_img(route)
		plot_model(model, to_file=route, show_shapes=True, show_layer_names=False)

	def build_and_train_cpu(self, save_path=None):
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Training with CPU"})
		try:
			strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
			with strategy.scope():
				self.build_and_train(save_path)
		except ValueError as e:
			logging.warning(e)

	def build_and_dual_gpu(self):
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Training with CPU"})
		try:
			strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
			with strategy.scope():
				self.build_and_train()
		except ValueError as e:
			logging.warning(e)

	def build_and_train(self, save_path=None) -> float:
		print("Status socket: ", SocketCommunication.isSocket)
		if self.search_space_type == SearchSpaceType.IMAGE:
			use_augmentation = not self.is_partial_training
		else:
			use_augmentation = False
		
		config = tf.compat.v1.ConfigProto()
		config.gpu_options.allow_growth= True
		session= tf.compat.v1.Session(config=config)
		
		input_shape = self.dataset.get_input_shape()
		class_count = self.dataset.get_classes_count()
		model = self.build_model(input_shape, class_count)
		train = self.dataset.get_train_data(use_augmentation)
		training_steps = self.dataset.get_training_steps(use_augmentation)
		validation = self.dataset.get_validation_data()
		validation_steps = self.dataset.get_validation_steps()
		test = self.dataset.get_test_data()
		def scheduler(epoch):
			if epoch < 10:
				return 0.001
			else:
				return 0.001 * tf.math.exp(0.01 * (10 - epoch))
		scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
		early_stopping: keras.callbacks.EarlyStopping = None
		if self.search_space_type == SearchSpaceType.IMAGE:
			monitor_exploration_training = 'val_loss'
			monitor_full_training = 'val_accuracy'
		elif self.search_space_type == SearchSpaceType.TIME_SERIES:
			monitor_exploration_training = 'loss'
			monitor_full_training = 'loss'
		else: 
			monitor_exploration_training = 'val_loss'
			monitor_full_training = 'val_loss'

		if self.is_partial_training:
			early_stopping = keras.callbacks.EarlyStopping(monitor=monitor_exploration_training, patience=self.early_stopping_patience, verbose=1, restore_best_weights=False)
		else:
			early_stopping = keras.callbacks.EarlyStopping(monitor=monitor_full_training, patience=self.early_stopping_patience, verbose=1, restore_best_weights=True)

		model_stage = "exp" if self.is_partial_training else "hof"
		log_dir = "logs/{}/{}-{}".format(self.experiment_id, model_stage, str(self.id))
		tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
		end_callback = EndEpoch()
		callbacks = [early_stopping, tensorboard, scheduler_callback, end_callback]
		total_weights = np.sum([np.prod(v.get_shape().as_list()) for v in model.variables])
		cad = 'Total weights ' + str(total_weights)
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': cad})
		if save_path != None:
			self.remove_img(save_path)
			print("Saving model at:", SP.DATA_ROUTE+save_path)
			plot_model(model, to_file=SP.DATA_ROUTE+save_path+".png", show_shapes=True, show_layer_names=False)
		else:
			print("The model can't be saved")
		if self.search_space_type == SearchSpaceType.IMAGE_TIME_SERIES:
			history = model.fit(
				train,
				epochs=self.epochs,
				steps_per_epoch=training_steps,
				callbacks=callbacks,
				validation_data = validation,
				validation_steps=validation_steps,
				shuffle=True
			)
		else:
			history = model.fit(
				train,
				epochs=self.epochs,
				steps_per_epoch=training_steps,
				callbacks=callbacks,
				validation_data = validation,
				validation_steps=validation_steps,
			)
		did_finish_epochs = self._did_finish_epochs(history, self.epochs)
		if self.search_space_type == SearchSpaceType.IMAGE:
			loss, training_val = model.evaluate(test, verbose=0)
			cad = 'Model accuracy ' + str(training_val)
		else:
			training_val = model.evaluate(test, verbose=0)
			cad = 'Model accuracy ' + str(training_val)
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': cad})
		tf.keras.backend.clear_session()
		return training_val, did_finish_epochs

	def is_model_valid(self) -> bool:
		is_valid = True
		try:
			strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
			with strategy.scope():
				input_shape = self.dataset.get_input_shape()
				class_count = self.dataset.get_classes_count()
				self.build_model(input_shape, class_count)
		except ValueError as e:
			logging.warning(e)
			is_valid = False
		tf.keras.backend.clear_session()
		return is_valid

	@staticmethod
	def _did_finish_epochs(history, requested_epochs: int) -> bool:
		h = history.history
		trained_epochs = len(h['loss'])
		return requested_epochs == trained_epochs

	def _add_cnn_architecture(self, model: keras.Model, model_parameters = ImageModelArchitectureParameters, activation='relu', padding='same', kernel_initializer='he_uniform'):
		cnn_layers_per_block = model_parameters.cnn_blocks_conv_layers_n
		weight_decay = SP.WEIGHT_DECAY
		for n in range(0, model_parameters.cnn_blocks_n):
			filters = model_parameters.cnn_block_conv_filters[n]
			filter_size = model_parameters.cnn_block_conv_filter_sizes[n]
			max_pooling_size = model_parameters.cnn_block_max_pooling_sizes[n]
			dropout_value = model_parameters.cnn_block_dropout_values[n]
			for m in range(0, cnn_layers_per_block):
				model.add(layers.Conv2D(filters, (filter_size, filter_size), padding=padding, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(weight_decay)))
				model.add(layers.BatchNormalization())
			model.add(keras.layers.MaxPooling2D(3, 2, padding=padding))
			model.add(keras.layers.Dropout(dropout_value))

	def _add_inception_architecture(self, model: keras.Model, model_parameters: ImageModelArchitectureParameters, activation='relu', padding='same'):
		for n in range(0, model_parameters.inception_stem_blocks_n):
			filters = model_parameters.inception_stem_block_conv_filters[n]
			conv_size = model_parameters.inception_stem_block_conv_filter_sizes[n]
			model.add(layers.Conv2D(filters, (conv_size, conv_size), padding='valid', activation=activation))
			model.add(layers.BatchNormalization())
			pool_size = model_parameters.inception_stem_block_max_pooling_sizes[n]
			model.add(keras.layers.MaxPooling2D((pool_size, pool_size)))

		for n in range(0, model_parameters.inception_blocks_n):
			conv1x1_filters = model_parameters.inception_modules_conv1x1_filters[n]
			conv3x3_reduce_filters = model_parameters.inception_modules_conv3x3_reduce_filters[n]
			conv3x3_filters = model_parameters.inception_modules_conv3x3_filters[n]
			conv5x5_reduce_filters = model_parameters.inception_modules_conv5x5_reduce_filters[n]
			conv5x5_filters = model_parameters.inception_modules_conv5x5_filters[n]
			pooling_conv_filters = model_parameters.inception_modules_pooling_conv_filters[n]
		for i in range(0, model_parameters.inception_modules_n):
			model.add(
				InceptionV1ModuleBN(
					conv1x1_filters,
					conv3x3_reduce_filters,
					conv3x3_filters,
					conv5x5_reduce_filters,
					conv5x5_filters,
					pooling_conv_filters,
				)
			)
		model.add(keras.layers.MaxPool2D(3, 2, padding=padding))

	def _add_mlp_architecture(self, model: keras.Model, model_parameters, class_count: int, kernel_initializer='normal', activation='relu'):
		for n in range(0, model_parameters.classifier_layers_n):
			units = model_parameters.classifier_layers_units[n]
			model.add(keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer))
			dropout = model_parameters.classifier_dropouts[n]
			model.add(keras.layers.Dropout(dropout))

	def _add_time_series_lstm_architecture(self, model: keras.Model, model_parameters: TimeSeriesModelArchitectureParameters, class_count: int, activation='tanh'):
		for n in range(model_parameters.lstm_layers_n-1):
			units = model_parameters.lstm_layers_units[n]
			model.add(keras.layers.LSTM(units, return_sequences=True, activation=activation))
		units = model_parameters.lstm_layers_units[model_parameters.lstm_layers_n-1]
		model.add(keras.layers.LSTM(units, activation=activation))

	def build_image_model(self, model_parameters: ImageModelArchitectureParameters, input_shape: tuple, class_count: int) -> keras.Sequential:
		start_time = int(round(time.time() * 1000))
		#policy = mixed_precision.Policy('mixed_float16')
		#mixed_precision.set_policy(policy)
		#print('Compute dtype: %s' % policy.compute_dtype)
		#print('Variable dtype: %s' % policy.variable_dtype)
		model = keras.Sequential()
		model.add(keras.layers.Input(input_shape))
		if model_parameters.base_architecture == 'cnn':
			self._add_cnn_architecture(model, model_parameters, SP.LAYERS_ACTIVATION_FUNCTION, SP.PADDING, SP.KERNEL_INITIALIZER)
		elif model_parameters.base_architecture == 'inception':
			self._add_inception_architecture(model, model_parameters, SP.LAYERS_ACTIVATION_FUNCTION, SP.PADDING)

		if model_parameters.classifier_layer_type == 'gap':
			model.add(keras.layers.Conv2D(class_count, (1,1), activation=SP.LAYERS_ACTIVATION_FUNCTION, kernel_initializer=SP.KERNEL_INITIALIZER))
			model.add(keras.layers.GlobalAveragePooling2D())
			model.add(keras.layers.Activation(SP.OUTPUT_ACTIVATION_FUNCTION, dtype=SP.DTYPE))
		elif model_parameters.classifier_layer_type == 'mlp':
			model.add(keras.layers.Flatten())
			self._add_mlp_architecture(model, model_parameters, class_count, SP.KERNEL_INITIALIZER, SP.LAYERS_ACTIVATION_FUNCTION)
			model.add(keras.layers.Dense(class_count))
			model.add(keras.layers.Activation(SP.OUTPUT_ACTIVATION_FUNCTION, dtype=SP.DTYPE))
		model.compile(optimizer=SP.OPTIMIZER, loss=SP.LOSS_FUNCTION, metrics=SP.METRICS)
		elapsed_seconds = int(round(time.time() * 1000)) - start_time
		print("Model building took", elapsed_seconds, "(miliseconds)")
		model.summary()
		return model

	def build_regression_model(self, model_parameters: RegressionModelArchitectureParameters, input_shape: tuple, class_count: int) -> keras.Sequential:
		start_time = int(round(time.time() * 1000))
		policy = mixed_precision.Policy('mixed_float16')
		mixed_precision.set_policy(policy)
		model = keras.Sequential()
		model.add(keras.layers.Input(input_shape))
		if model_parameters.base_architecture == 'mlp':
			self._add_mlp_architecture(model, model_parameters, class_count, SP.KERNEL_INITIALIZER, SP.LAYERS_ACTIVATION_FUNCTION)
		model.add(keras.layers.Dense(class_count))
		model.add(keras.layers.Activation(SP.OUTPUT_ACTIVATION_FUNCTION, dtype=SP.DTYPE))
		model.compile(optimizer=SP.OPTIMIZER, loss=SP.LOSS_FUNCTION)
		elapsed_seconds = int(round(time.time() * 1000)) - start_time
		print('Model building took', elapsed_seconds, '(miliseconds)')
		model.summary()
		return model

	def build_time_series_model(self, model_parameters: TimeSeriesModelArchitectureParameters, input_shape: tuple, class_count: int) -> keras.Sequential:
		start_time = int(round(time.time() * 1000))
		#policy = mixed_precision.Policy('mixed_float16')
		#mixed_precision.set_policy(policy)
		model = keras.Sequential()
		model.add(keras.layers.Input(input_shape))
		if model_parameters.base_architecture == 'lstm':
			self._add_time_series_lstm_architecture(model, model_parameters, class_count, SP.LSTM_ACTIVATION_FUNCTION)
		if model_parameters.base_architecture == 'mlp' or model_parameters.classifier_layer == 'mlp':
			self._add_mlp_architecture(model, model_parameters, class_count, SP.KERNEL_INITIALIZER, SP.LAYERS_ACTIVATION_FUNCTION)
		#All combination has the same final layers
		model.add(keras.layers.Dense(class_count))
		model.add(keras.layers.Activation(SP.OUTPUT_ACTIVATION_FUNCTION, dtype=SP.DTYPE))
		model.compile(optimizer=SP.OPTIMIZER, loss=SP.LOSS_FUNCTION)
		elapsed_seconds = int(round(time.time() * 1000))- start_time
		print("Model building took", elapsed_seconds, "(miliseconds)")
		model.summary()
		return model

	def build_image_time_series_model(self, model_parameters: ImageTimeSeriesModelArchitectureParameters, input_shape: tuple, class_count: int) -> keras.Sequential:
		start_time = int(round(time.time() * 1000))
		#policy = mixed_precision.Policy('mixed_float16')
		#mixed_precision.set_policy(policy)
		model = keras.Sequential()
		model.add(keras.layers.Input(input_shape))
		for i in range(model_parameters.codifier_layers_n):
			model.add(keras.layers.Conv2D(model_parameters.codifier_units[i], model_parameters.conv_kernels[i], padding=SP.PADDING, activation=SP.LAYERS_ACTIVATION_FUNCTION))
			model.add(keras.layers.MaxPooling2D((model_parameters.kernels_x[i], model_parameters.kernels_y[i])))
		for i in reversed(range(model_parameters.codifier_layers_n)):
			model.add(keras.layers.Conv2D(model_parameters.codifier_units[i], model_parameters.conv_kernels[i], padding=SP.PADDING, activation=SP.LAYERS_ACTIVATION_FUNCTION))
			model.add(keras.layers.UpSampling2D((model_parameters.kernels_x[i], model_parameters.kernels_y[i])))
		print(model_parameters)
		#print(model.summary())
		#Output Layer
		model.add(keras.layers.Conv2D(input_shape[2], (3,3), activation=SP.OUTPUT_ACTIVATION_FUNCTION, padding=SP.PADDING))
		#Compilación del modelo
		model.compile(optimizer=SP.OPTIMIZER, loss=SP.LOSS_FUNCTION)
		elapsed_seconds = int(round(time.time() * 1000)) - start_time
		print("Model building took", elapsed_seconds, "(miliseconds)")
		print(model.summary())
		return model

	