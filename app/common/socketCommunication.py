from flask_socketio import SocketIO, send

class MSGType():
	MASTER_STATUS = 'masterStatus'
	SLAVE_STATUS = 'slaveStatus'
	NEW_MODEL = 'newModel'
	MASTER_MODEL_COUNT = 'modelCount'
	FINISHED_MODEL = 'finishModel'
	CHANGE_PHASE = 'changePhase'
	MASTER_ERROR = 'masterError'
	FINISHED_TRAINING = 'finishedTrain'
	END_EPOCH = 'endEpoch'
	RECIEVED_MODEL = 'recievedModel'

class SocketCommunication:
	isSocket: bool = False
	socket: SocketIO = None

	@staticmethod	
	def decide_print_form(msgType: MSGType, info):
		if SocketCommunication.isSocket:
			print("Mandando mensaje, con socket")
			SocketCommunication.socket.emit(msgType, info)
			#socket.send(info, json=True, broadcast=True)
		else:
			print(info['msg'])