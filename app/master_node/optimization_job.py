import asyncio
import time
import datetime
import json
import tempfile
import shutil
import os
from dataclasses import asdict
import aio_pika
import tensorflow as tf
import matplotlib as mat
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from app.common.model import Model
from app.common.model_communication import *
from app.common.rabbit_connection_params import RabbitConnectionParams
from app.common.search_space import *
from app.master_node.communication.master_rabbitmq_client import *
from app.master_node.communication.rabbitmq_monitor import *
from app.master_node.optimization_strategy import OptimizationStrategy, Action, Phase
from app.common.dataset import * 
from system_parameters import SystemParameters as SP
from app.common.socketCommunication import *
from app.common.tools import *

class OptimizationJob:

	def __init__(self, dataset: Dataset, model_architecture_factory: ModelArchitectureFactory):
		asyncio.set_event_loop(asyncio.new_event_loop())
		self.loop = asyncio.get_event_loop()
		self.dataset = dataset
		self.search_space: ModelArchitectureFactory = model_architecture_factory
		self.optimization_strategy = OptimizationStrategy(self.search_space, self.dataset, SP.EXPLORATION_SIZE, SP.HALL_OF_FAME_SIZE)
		#Creates a connection with a connection types as a parameter
		rabbit_connection_params = RabbitConnectionParams.new()
		self.rabbitmq_client = MasterRabbitMQClient(rabbit_connection_params, self.loop)
		self.rabbitmq_monitor = RabbitMQMonitor(rabbit_connection_params)
		self.autoencoder_finished = False

	def start_optimization(self, trials: int):
		self.start_time = time.time()
		self.loop.run_until_complete(self._run_optimization_startup())
		connection = self.loop.run_until_complete(self._run_optimization_loop(trials))
		try:
			self.loop.run_forever()
			if self.search_space.get_search_space().get_type() == SearchSpaceType.IMAGE_TIME_SERIES:
				self.autoencoder_finished = True
				self.temporal = tempfile.mkdtemp()
				try:
					self.loop.run_until_complete(self.perform_image_time_series_training(self.model.build_model_CPU(self.dataset.get_input_shape(), self.dataset.get_classes_count())))
					encoder = None
					decoder = None
					self.loop.run_forever()
				except:
					SocketCommunication.decide_print_form(MSGType.FINISHED_TRAINING, {'node': 1, 'msg': 'Something went wrong trying to perform multiple trainings'})
					raise
				finally:
					shutil.rmtree(self.temporal)
		finally:
			self.loop.run_until_complete(connection.close())
		#Inicializar el proceso de pronostico para generación de multiples imágenes a futuro

	async def _run_optimization_startup(self):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': '*** Running optimization startup ***'})
		await self.rabbitmq_client.prepare_queues()
		queue_status: QueueStatus = await self.rabbitmq_monitor.get_queue_status()
		self.consumers = queue_status.consumer_count
		print("Este es el numero de consumidores",range(self.consumers))
		for i in range (0, queue_status.consumer_count + 1 ): #aqui va un mas 1
			await self.generate_model()
	
	async def _run_image_time_series_startup(self):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': '*** Running Image time series startup ***'})
		await self.rabbitmq_client.prepare_queues()
		queue_status: QueueStatus = await self.rabbitmq_monitor.get_queue_status()
		self.consumers = queue_status.consumer_count

	async def _run_optimization_loop(self, trials: int) -> aio_pika.Connection:
		connection = await self.rabbitmq_client.listen_for_model_results(self.on_model_results)
		return connection

	async def _run_optimization_image_time_series_loop(self) -> aio_pika.Connection:
		connection = await self.rabbitmq_client.listen_for_model_results(self.on_image_time_series_results)
		return connection

	async def on_image_time_series_results(self, response:dict):
		print("A worker has finished")
		self.process_count +=1
		print(self.process_count, " consumers:", self.consumers)
		if self.process_count == self.consumers:
			self.stop = False

	async def on_model_results(self, response: dict):
		if self.search_space.get_search_space().get_type() == SearchSpaceType.IMAGE_TIME_SERIES and self.autoencoder_finished:
			print("A worker has finished")
			self.process_count +=1
			print(self.process_count, " consumers:", self.consumers)
			if self.process_count == self.consumers:
				await self.perform_last_image_time_series_steps()
			return
		model_training_response = ModelTrainingResponse.from_dict(response)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Received response'})
		cad = str(model_training_response.id) + ' | ' + str(model_training_response.performance) + ' | ' + str(model_training_response.finished_epochs)
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': cad})
		action: Action = self.optimization_strategy.report_model_response(model_training_response)
		best = self.optimization_strategy.get_best_model(action=action)
		if action==Action.START_NEW_PHASE or  self.optimization_strategy.phase == Phase.EXPLORATION:
			SocketCommunication.decide_print_form(MSGType.FINISHED_MODEL, {'node': 1, 'msg': 'Finished a model', 'total': self.optimization_strategy.get_training_total(), 'best_id':best.model_training_request.id,'performance':best.performance})
		else:
			SocketCommunication.decide_print_form(MSGType.FINISHED_MODEL, {'node': 1, 'msg': 'Finished a model', 'total': self.optimization_strategy.get_training_total(), 'best_id':best.model_training_request.id,'performance':best.performance_2})
		if action == Action.GENERATE_MODEL:
			await self.generate_model()
		elif action == Action.WAIT:
			SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Wait for models'})
		elif action == Action.START_NEW_PHASE:
			queue_status: QueueStatus = await self.rabbitmq_monitor.get_queue_status()
			SocketCommunication.decide_print_form(MSGType.CHANGE_PHASE, {'node': 1, 'msg': 'New phase, deep training'})
			for i in range(0, queue_status.consumer_count + 1 ):#aui va un + 1
				await self.generate_model()
		elif action == Action.FINISH:
			SocketCommunication.decide_print_form(MSGType.FINISHED_TRAINING, {'node': 1, 'msg': 'Finished training'})
			self.best_model = self.optimization_strategy.get_best_model()
			await self._log_results(self.best_model)
			self.model = Model(self.best_model.model_training_request, self.dataset)
			self.model.is_model_valid()
			self.loop.stop()

	async def perform_image_time_series_training(self, model):
		self.init_train_time = datetime.datetime.now()
		print(model)
		try:
			strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
			with strategy.scope():
				encoder, decoder, self.decoder_input = self.get_encoder_decoder(model)
				print(encoder.summary())
				print(decoder.summary())
				train_matrix, val_matrix, self.test_matrix = self.transform_info(encoder, self.decoder_input)
				self.save_info_in_disk(model, encoder, decoder, train_matrix, val_matrix, self.test_matrix)
				SocketCommunication().decide_print_form(MSGType.FINISHED_TRAINING, {'node':1, 'msg': 'Processing data, quantity:'+str(len(train_matrix))})
				mid = int(len(train_matrix)/self.consumers)
				SocketCommunication().decide_print_form(MSGType.FINISHED_TRAINING, {'node':1, 'msg': 'Processing data per consumer, quantity:'+str(mid)})
				mid2 = int(len(val_matrix)/self.consumers)
				mid3 = int(len(self.test_matrix)/self.consumers)
				jsons = self.create_dictionaries(mid, mid2, mid3, len(train_matrix))
				#await self._run_optimization_image_time_series_loop()
				self.process_count = 0
				for j in jsons:
					await self._send_part_info_to_broker(j)
				#tf.keras.backend.clear_session()
		except ValueError as e:
			raise
		#self.loop.run_until_complete(self._run_optimization_startup())
		#try:
		#	self.loop.run_forever()
		#except:
		#	print("Trono la espera")
		#	raise

	async def perform_last_image_time_series_steps(self):
		SocketCommunication().decide_print_form(MSGType.FINISHED_TRAINING, {'node':1, 'msg': 'Training finished for model'})
		SocketCommunication().decide_print_form(MSGType.FINISHED_TRAINING, {'node':1, 'msg': 'Predicting next elements.'})
		predictions = self.predict_next_elements(self.temporal, self.test_matrix)
		decoder = tf.keras.models.load_model(self.temporal+'/decoder.h5')
		print(decoder.summary())
		predictions = np.array(predictions)
		predictions = predictions.reshape((SP.PREDICTION_SIZE, self.decoder_input[0], self.decoder_input[1], self.decoder_input[2]))
		print("Prediction shape:", predictions.shape)
		pred = decoder.predict(predictions)
		print(pred[0][400])
		pred = pred * 255
		pred = pred.astype('uint8')
		mat.image.imsave('aux/NoMono.png', pred[0].reshape(480, 640), cmap='Greys')
		if SP.DATASET_COLOR_SCALE == 3:
			print(pred[0][400])
			predM = to_monochromatic(pred)
			mat.image.imsave('aux/Mono.png', predM[0].reshape(480, 640), cmap='Greys')
			im = Image.fromarray(predM[0])
			im.save("aux/MonoPIL.png")
			plt.imshow(predM[0])
			plt.savefig("aux/MonoSavefig.png")
			print(predM[0][400])
		print("Model prediction shape:",pred.shape)
		finish_train_time = datetime.datetime.now()
		train_time = finish_train_time - self.init_train_time
		#model_info_json = json.dumps(asdict(self.best_model))
		model_info_json = self.best_model
		folder_results = SP.IMAGES_TIME_SERIES_RES_FOLDER+'/train'+self.init_train_time.strftime("%Y%m%d-%H%M%S")+'_'+str(self.decoder_input)+'_'+self.strfdelta(train_time,"{d}d-{h}h-{m}m-{s}s")+'_'+str(model_info_json.performance_2)
		try:
			os.mkdir(folder_results)
		except OSError():
			print("Creation of the directory ResDrought/%s failed" %folder_results)
			print("Saving in the root directory")
			folder_results = "train"+self.init_train_time.strftime("%Y%m%d-%H%M%S")+'_'+str(self.decoder_input)+'_'+self.strfdelta(train_time,"{d}d-{h}h-{m}m-{s}s")+'_'+str(model_info_json.performance_2)
			os.mkdir(folder_results)
		else:
			print("Successfully created the directory %s" %folder_results)
		self.save_imgs(pred, folder_results, SP.DATASET_SHAPE[0], SP.DATASET_SHAPE[1], SP.DATASET_SHAPE[2], begin = 0)
		self.save_imgs(predM, folder_results, SP.DATASET_SHAPE[0], SP.DATASET_SHAPE[1], SP.DATASET_SHAPE[2], begin = 0, start_name=11)
		#self.finish = False

	def strfdelta(self, tdelta, fmt):
		d = {"d": tdelta.days}
		d["h"], rem = divmod(tdelta.seconds, 3600)
		d["m"], d["s"] = divmod(rem, 60)
		return fmt.format(**d)

	def get_encoder_decoder(self, model):
		input_shape = tf.keras.layers.Input(self.dataset.get_input_shape())
		encoder = input_shape
		cant = int(len(model.layers)/2)
		for i in range(cant):
			print(encoder)
			encoder = model.layers[i](encoder)
		encoder = tf.keras.Model(input_shape, encoder)
		cant = int(len(model.layers)/2)
		in_shape = tf.keras.layers.Input(shape=self.get_model_shape(model, cant))
		decoder = in_shape
		for i in reversed(range(cant)):
			print(decoder)
			decoder = model.layers[-(i+1)](decoder)
		decoder = tf.keras.Model(in_shape, decoder)
		return encoder, decoder, self.get_model_shape(model, cant)

	def get_model_shape(self, model, layer_i=-1):
		layer = model.layers[layer_i]
		layer_shape = layer.output_shape[1:]
		return layer_shape

	def transform_info(self, encoder, decoder_input):
		train_encoded_imgs = encoder.predict(self.dataset.get_train_data())
		validation_encoded_imgs = encoder.predict(self.dataset.get_validation_data())
		test_encoded_imgs = encoder.predict(self.dataset.get_test_data())
		print("decoder input:", decoder_input)
		print("train encoded shape:", train_encoded_imgs.shape)
		tam_imgs = decoder_input[0] * decoder_input[1] * decoder_input[2]
		print("tam imgs:", tam_imgs)
		train_matrix = self.create_matrix(train_encoded_imgs, tam_imgs)
		validation_matrix = self.create_matrix(validation_encoded_imgs, tam_imgs)
		test_matrix = self.create_matrix(test_encoded_imgs, tam_imgs)
		return train_matrix, validation_matrix, test_matrix
	
	def create_matrix(self, data, tam):
		matrix = np.array([])
		for i in data:
			matrix = np.append(matrix, i.reshape(1,tam))
		matrix = matrix.reshape(len(data), tam)
		matrix = matrix.transpose()
		return matrix
	
	def save_info_in_disk(self, autoencoder, encoder, decoder, train, validation, test):
		decoder.save(self.temporal+'/decoder.h5')
		encoder.save(self.temporal+'/encoder.h5')
		autoencoder.save(self.temporal+'/autoencoder.h5')
		np.save(self.temporal+'/train_data.npy', train)
		np.save(self.temporal+'/validation_data.npy', validation)
		np.save(self.temporal+'/test_data.npy', test)

	def create_dictionaries(self, middle_train, middle_validation, middle_test, total):
		jsons = []
		for i in range(self.consumers):
			init_train = i * middle_train
			init_validation = i * middle_validation
			init_test = i * middle_test
			finish_train = (i+1) * middle_train
			finish_validation = (i+1) * middle_validation
			finish_test = (i+1) * middle_test
			if i == self.consumers-1:
				if finish_train < total:
					finish_train = total
				if finish_validation < total:
					finish_validation = total
				if finish_test < total:
					finish_test = total
			dictionary = self.create_dictionary(init_train, finish_train, init_validation, finish_validation, init_test, finish_test, self.temporal)
			jsons.append(json.dumps(dictionary))
		return jsons

	def create_dictionary(self, i_train, f_train, i_validation, f_validation, i_test, f_test, temp_folder):
		dictionary = {
			'init_part_train': i_train,
			'finish_part_train': f_train,
			'init_part_validation': i_validation,
			'finish_part_validation': f_validation,
			'init_part_test': i_test,
			'finish_part_test': f_test,
			'temp_file': temp_folder,
			'window_size': SP.DATASET_WINDOW_SIZE
		}
		return dictionary

	def predict_next_elements(self, temporal, info_matrix):
		predictions = []
		aux_matrix = info_matrix[:, -SP.DATASET_WINDOW_SIZE:]
		for i in range(SP.PREDICTION_SIZE):
			predictions.append([])
		for i in range(len(aux_matrix)):
			model = tf.keras.models.load_model(temporal+'/'+str(i)+'.h5')
			sample_part = aux_matrix[i]
			for j in range(SP.PREDICTION_SIZE):
				pred = model.predict(sample_part.reshape(1,1,SP.DATASET_WINDOW_SIZE))
				predictions[j].append(pred[0,0])
				sample_part = np.delete(sample_part, 0)
				sample_part = np.append(sample_part, pred[0,0])
			tf.keras.backend.clear_session()
		return predictions

	def save_imgs(self, data, folder, rows, cols, channels=1, color_map='Greys', begin=-10, end=-1, start_name=0):
		if end == -1:
			save = data[begin:]
		else:
			save = data[begin:end]
		print(save.shape)
		for i in range(len(save)):
			if channels == 1:
				mat.image.imsave(folder+'/'+str(i+start_name)+'.png', save[i].reshape(rows, cols), cmap=color_map)
			else:
				mat.image.imsave(folder+'/'+str(i+start_name)+'.png', save[i].reshape(rows, cols, channels), cmap=color_map)
		
	async def generate_model(self):
		SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Generating new model'})
		model_training_request: ModelTrainingRequest = self.optimization_strategy.recommend_model()
		model = Model(model_training_request, self.dataset)
		if not model.is_model_valid():
			SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Model is not valid'})
			self.generate_model()
		else:
			await self._send_model_to_broker(model_training_request)
			SocketCommunication.decide_print_form(MSGType.MASTER_STATUS, {'node': 1, 'msg': 'Sent model to broker'})

	async def _send_model_to_broker(self, model_training_request: ModelTrainingRequest):
		model_training_request_dict = asdict(model_training_request)
		print("Model training request", model_training_request_dict)
		await self.rabbitmq_client.publish_model_params(model_training_request_dict)

	async def _send_part_info_to_broker(self, json_file):
		print("Image time series request:", json_file)
		await self.rabbitmq_client.publish_model_params(json_file)

	async def _log_results(self, best_model):
		filename = best_model.model_training_request.experiment_id
		f = open('Results/'+filename, "a")
		model_info_json = json.dumps(asdict(best_model))
		f.write(model_info_json)
		f.close()

		print('Finished optimization')
		print('Best model: ')
		print(model_info_json)

		self.dataset.load()
		ranges = self.dataset.get_ranges()
		print('Information ranges from normalization')
		print(ranges)

		f = open('Results/'+filename, "a")
		ranges_json = json.dumps(ranges)
		f.write('\n'+ranges_json)
		f.close()

		elapsed_seconds = time.time() - self.start_time
		elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))

		time_text = "\nOptimization took: " + str(elapsed_time) + " (hh:mm:ss) " + str(elapsed_seconds) + " (Seconds) "
		print(time_text)

		f = open('Results/'+filename, "a")
		f.write(time_text)
		f.close()

		print("\n ********************************************** \n")