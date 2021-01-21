import asyncio
import time
import concurrent
import logging
from dataclasses import asdict
import aio_pika
from app.common.model import Model
from app.common.model_communication import *
from app.common.rabbit_connection_params import RabbitConnectionParams, ConnectionType
from app.common.search_space import *
from app.slave_node.slave_rabbitmq_client import SlaveRabbitMQClient
from app.common.dataset import *
from system_parameters import *

class TrainingSlave:

	def __init__(self, model_architecture_factory: ModelArchitectureFactory, slave_number=0):
		self.loop = asyncio.get_event_loop()
		rabbit_connection_params = RabbitConnectionParams.new(ConnectionType.SLAVE, slave_number)
		self.rabbitmq_client = SlaveRabbitMQClient(rabbit_connection_params, self.loop)
		model_architecture_factory = model_architecture_factory
		self.search_space_hash = model_architecture_factory.get_search_space().get_hash()
		print('Hash', self.search_space_hash)
		self.model_type = model_architecture_factory.get_search_space().get_type()

	def start_slave(self):
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
	def train_model(model_training_request) -> float:
		if DATASET_TYPE == 1:
			dataset = ImageClassificationBenchmarkDataset(DATASET_NAME, DATASET_SHAPE, DATASET_CLASSES, DATASET_BATCH_SIZE, DATASET_VALIDATION_SPLIT)
		elif DATASET_TYPE == 2:
			dataset = RegressionBenchmarkDataset(DATASET_NAME, DATASET_SHAPE, DATASET_FEATURES, DATASET_LABELS, DATASET_BATCH_SIZE, DATASET_VALIDATION_SPLIT)
		elif DATASET_TYPE == 3:
			dataset = TimeSeriesBenchmarkDataset(DATASET_NAME, DATASET_WINDOW_SIZE, DATASET_DATA_SIZE, DATASET_BATCH_SIZE, DATASET_VALIDATION_SPLIT)
		else:
			print("Please enter a valid dataset type")
			return
		dataset.load()
		model = Model(model_training_request, dataset)
		if TRAIN_GPU:
			return model.build_and_train()
		else:
			return model.build_and_train_cpu()

	async def _start_listening(self) -> aio_pika.Connection:
		print("Worker started!")
		return await self.rabbitmq_client.listen_for_model_params(self._on_model_params_received)

	async def _on_model_params_received(self, model_params):
		print("Received model training request")
		print(model_params)
		self.model_type = int(model_params['training_type'])
		model_training_request = ModelTrainingRequest.from_dict(model_params, self.model_type)
		if not self.search_space_hash == model_training_request.search_space_hash:
			raise Exception("Search space of master is different to this worker's search space")
		with concurrent.futures.ProcessPoolExecutor() as pool:
			training_val, did_finish_epochs = await self.loop.run_in_executor(pool, self.train_model, model_training_request)
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