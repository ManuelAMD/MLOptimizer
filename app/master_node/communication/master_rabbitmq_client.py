import aio_pika

from app.common.base_rabbitmq_client import BaseRabbitMQClient

class MasterRabbitMQClient(BaseRabbitMQClient):
	
	async def publish_model_params(self, model_params: dict) -> aio_pika.Connection:
		return await super().publish(self.model_parameter_queue, model_params, auto_close_connection=False)

	async def listen_for_model_results(self, callback) -> aio_pika.Connection:
		return await super().listen(self.model_performance_queue, callback, auto_close_connection=False)