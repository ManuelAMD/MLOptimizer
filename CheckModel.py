import asyncio
import time
import json
from dataclasses import asdict
import aio_pika
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

def reproduce_Architecture(f):
    info = json.load(f)
    model = info['model_training_request']
    print(model)
    request = ModelTrainingRequest.from_dict(model, 3)
    print(request)
    m = Model(request, None)
    m.build_model((1,1),1)
    return None

filename = 'Architecture Results/monthly-beer-production-in-austr-20210617-162711'
f = open(filename)
reproduce_Architecture(f)