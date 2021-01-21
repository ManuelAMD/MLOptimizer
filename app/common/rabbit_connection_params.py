from dataclasses import dataclass
from system_parameters import *
from enum import Enum

class ConnectionType(Enum):
    MASTER = 1
    SLAVE = 2

#Connection params for rabbitMQ
@dataclass(frozen=True)
class RabbitConnectionParams:
	port: int
	model_parameter_queue: str
	model_performance_queue: str
	host_url: str
	user: str
	password: str
	virtual_host: str

	def new(connection_type: int, slave_number=0):
		if connection_type == ConnectionType.MASTER:
			return RabbitConnectionParams(
				port = int(MASTER_CONNECTION[0]),
				model_parameter_queue = MASTER_CONNECTION[1],
				model_performance_queue = MASTER_CONNECTION[2],
				host_url = MASTER_CONNECTION[3],
				user = MASTER_CONNECTION[4],
				password = MASTER_CONNECTION[5],
				virtual_host = MASTER_CONNECTION[6],
			)
		elif connection_type == ConnectionType.SLAVE:
			return RabbitConnectionParams(
				port = int(MASTER_CONNECTION[0]),
				model_parameter_queue = MASTER_CONNECTION[1],
				model_performance_queue = MASTER_CONNECTION[2],
				host_url = MASTER_CONNECTION[3],
				user = MASTER_CONNECTION[4],
				password = MASTER_CONNECTION[5],
				virtual_host = MASTER_CONNECTION[6],
			)