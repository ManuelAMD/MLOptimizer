import asyncio
import time
import concurrent
import logging
from dataclasses import asdict
import aio_pika
import pika
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
		return await self.rabbitmq_client.listen_for_model_params_pika(self._on_model_params_received)

	async def _on_model_params_received(self, model_params):
		SocketCommunication.decide_print_form(MSGType.SLAVE_STATUS, {'node': 2, 'msg': "Received model training request"})
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
		await self._send_performance_to_broker(model_training_response)

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
