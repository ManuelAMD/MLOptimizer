import aio_pika
from app.common.base_rabbitmq_client import BaseRabbitMQClient

class SlaveRabbitMQClient(BaseRabbitMQClient):
	async def publish_model_performance(self, model_params: dict):
		await super().publish(self.model_performance_queue, model_params, auto_close_connection=False)
		print('[X] Sent model performance')
		print(model_params)

	async def listen_for_model_params(self, callback) -> aio_pika.Connection:
		return await super().listen(self.model_parameter_queue, callback, auto_close_connection=False)