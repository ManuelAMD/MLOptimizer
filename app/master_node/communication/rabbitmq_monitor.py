from aiohttp import BasicAuth, ClientSession
from dataclasses import dataclass
from app.common.rabbit_connection_params import RabbitConnectionParams

@dataclass
class QueueStatus:
	queue_name: str
	consumer_count: int
	message_count: int

class RabbitMQMonitor(object):

	def __init__(self, params: RabbitConnectionParams):
		self.cp = params
		self.auth = BasicAuth(login=self.cp.user, password=self.cp.password)

	async def get_queue_status(self) -> QueueStatus:
		print("Requesting queue status...")
		async with ClientSession(auth=self.auth) as session:
			if self.cp.host_url == 'localhost':
				url = 'http://localhost:15672/api/queues/%2F/parameters'
			else:
				url = "https://{}/api/queues/{}/{}".format(self.cp.host_url, self.cp.vitual_host, self.cp.model_parameter_queue)
			print(url)
			async with session.get(url) as resp:
				print(resp.status)
				body = await resp.json()
				consumer_count = body['consumers']
				message_count = body['messages']
				queue_name = body['name']
				queue_status = QueueStatus(
					queue_name=queue_name,
					consumer_count=consumer_count,
					message_count=message_count
				)
				print("Received queue status")
				print(queue_status)
				return queue_status