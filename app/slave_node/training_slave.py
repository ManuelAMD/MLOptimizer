import asyncio
import time
import concurrent
import logging
from dataclasses import asdict
import aio_pika
import pika
import tensorflow as tf
from app.common.model import Model
from app.common.model_communication import *
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.search_space import *
from app.slave_node.slave_rabbitmq_client import SlaveRabbitMQClient
from app.common.dataset import *
from system_parameters import SystemParameters as SP

class TrainingSlave:

	def __init__(self, dataset: Dataset, model_architecture_factory: ModelArchitectureFactory, loop = None):
		if loop == None:
			asyncio.set_event_loop(asyncio.new_event_loop())
			self.loop = asyncio.get_event_loop()
		else:
			self.loop = loop
		self.dataset = dataset
		rabbit_connection_params = RabbitConnectionParams.new()
		self.rabbitmq_client = SlaveRabbitMQClient(rabbit_connection_params, self.loop)
		model_architecture_factory = model_architecture_factory
		self.search_space_hash = model_architecture_factory.get_search_space().get_hash()
		cad = 'Hash ' + str(self.search_space_hash) 
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': cad})
		self.model_type = model_architecture_factory.get_search_space().get_type()

	def start_slave(self):
		#loop = asyncio.get_event_loop()
		#loop.run_until_complete(asyncio.wait(futures))
		#connection = asyncio.run(self._start_listening())
		#asyncio.new_event_loop(self._start_listening())
		#await connection = self._start_listening()
		connection = self.loop.run_until_complete(self._start_listening())
		try:
			self.loop.run_forever()
		finally:
			self.loop.run_until_complete(connection.close())

	@staticmethod
	def fake_blocking_training():
		# Method for testing broker connection timeout
		for i in range(0, 240):
			time.sleep(1)
			print(i)
		return 0.5

	@staticmethod
	def train_model(info_dict: dict) -> float:
		dataset = info_dict['dataset']
		print(dataset)
		model_training_request = info_dict['model_request']
		dataset.load(info_dict['init_route'])
		model = Model(model_training_request, dataset)
		if SP.TRAIN_GPU:
			return model.build_and_train(info_dict['model_img'])
		else:
			return model.build_and_train_cpu(info_dict['model_img'])

	async def _start_listening(self):
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Worker started!"})
		return await self.rabbitmq_client.listen_for_model_params(self._on_model_params_received)

	async def _on_model_params_received(self, model_params):
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Received model training request"})
		try:
			#Enter if it's a image time series training
			self.temporal = model_params['temp_file']
			print(self.temporal)
			aux_train = np.load(self.temporal+'/train_data.npy')
			aux_validation = np.load(self.temporal+'/validation_data.npy')
			aux_test = np.load(self.temporal+'/test_data.npy')
			self.train_matrix = aux_train[model_params['init_part_train']:model_params['finish_part_train']]
			self.validation_matrix = aux_validation[model_params['init_part_validation']:model_params['finish_part_validation']]
			self.test_matrix = aux_test[model_params['init_part_test']:model_params['finish_part_test']]
			self.window_size = model_params['window_size']
			self.perform_image_time_series_training(model_params)
			res = {'res':'Worker process finished'}
			print('publishing')
			await self._send_performance_to_broker(res)
		except:
			#print(model_params)
			self.model_type = int(model_params['training_type'])
			model_training_request = ModelTrainingRequest.from_dict(model_params, self.model_type)
			if not self.search_space_hash == model_training_request.search_space_hash:
				raise Exception("Search space of master is different to this worker's search space")
			SocketCommunication.decide_print_form(MSGType.RECIEVED_MODEL, {'node':2, 'msg':' New model recieved', 'epochs':model_training_request.epochs})
			info_dict = {
				'model_img': SP.MODEL_IMG,
				'init_route': SP.DATA_ROUTE,
				'dataset': self.dataset,
				'model_request': model_training_request,
				'isSocket': SocketCommunication.isSocket
			}
			if SocketCommunication.isSocket:
				training_val, did_finish_epochs = self.train_model(info_dict)
			else:
				with concurrent.futures.ProcessPoolExecutor() as pool:
					training_val, did_finish_epochs = await self.loop.run_in_executor(pool, self.train_model, info_dict)
			model_training_response = ModelTrainingResponse(id=model_training_request.id, performance=training_val, finished_epochs=did_finish_epochs)
			print("KEYYY:", model_training_response) 
			await self._send_performance_to_broker(model_training_response)

	async def perform_image_time_series_training(self, params):
		init_count = params['init_part_train']
		cont = 0
		cant = 50
		fails = 0
		limit = 0.05
		fields = ['Modelid', 'loss', 'val_loss', 'retrain_loss', 'retrain_val_loss']
		info = []
		for i in range(len(self.train_matrix)):
			x_train, y_train, x_validation, y_validation = self.generate_dataset(i)
			model, history = self.train_one_model(x_train, y_train, x_validation, y_validation, SP.IMAGES_TIME_SERIES_EPOCHS, batch_size=SP.IMAGES_TIME_SERIES_BATCH_SIZE, early_patience=SP.IMAGES_TIME_SERIES_EARLY)
			if history.history['val_loss'][-1] > limit:
				print('Model number:', (i+init_count), 'does not complete the enough score')
				print('val_loss:', history.history['val_loss'][-1])
				fails += 1
				info.append([i+init_count, history.history['loss'][-1], history.history['val_loss'][-1]])
			else:
				model.save(self.temporal+'/'+str(i+init_count)+'.h5')
			tf.keras.backend.clear_session()
			cont += 1
		pass

	async def train_one_model(self, x_train, y_train, x_validation, y_validation, epochs, batch_size=64, early_patience=5, restore_weights=True):
		shape = (int(x_train.shape[1]), int(x_train.shape[2]))
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Input(shape))
		model.add(tf.keras.layers.LSTM(units=64, return_sequences=True, activation='elu'))
		model.add(tf.keras.layers.LSTM(units=32, activation='elu'))
		model.add(tf.keras.layers.Dense(32, activation='elu'))
		model.add(tf.keras.layers.Dense(16, activation='elu'))
		model.add(tf.keras.layers.Dense(1, activation='elu'))
		model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
		es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=early_patience, restore_best_weights=restore_weights)
		history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, y_validation), verbose=0, shuffle=True, callbacks=[es])
		return model, history

	async def generate_dataset(self, i):
		x_train_part = np.array([])
		y_train_part = np.array([])
		x_validation_part = np.array([])
		y_validation_part = np.array([])
		for j in range(len(self.train_matrix[i]) - self.window_size):
			x_train_part = np.append(x_train_part, self.train_matrix[i][j:self.window_size+j])
			y_train_part = np.append(y_train_part, self.train_matrix[i][j+self.window_size])
		x_train = x_train_part.reshape((len(self.train_matrix[i])-self.window_size, 1, self.window_size))
		y_train = y_train_part.reshape((len(y_train_part),1,1))
		for j in range(len(self.validation_matrix[i]) - self.window_size):
			x_validation_part = np.append(x_validation_part, self.validation_matrix[i][j:self.window_size+j])
			y_validation_part = np.append(y_validation_part, self.validation_matrix[i][j+self.window_size])
		x_validation = x_validation_part.reshape((len(self.validation_matrix[i])-self.window_size, 1, self.window_size))
		y_validation = y_validation_part.reshape((len(y_validation_part),1,1))
		return x_train, y_train, x_validation, y_validation

	async def _send_performance_to_broker(self, model_training_response: ModelTrainingResponse):
		print(model_training_response)
		model_training_response_dict = asdict(model_training_response)
		print(model_training_response_dict)
		await self.rabbitmq_client.publish_model_performance(model_training_response_dict)


def handle_exception(loop, context):
    # context["message"] will always be there; but context["exception"] may not
    msg = context.get("exception", context["message"])
    logging.error(f"Caught exception: {msg}")
    logging.error(context["exception"])
    logging.info("Shutting down...")
    loop.stop()
