from flask import Flask, request
from flask_socketio import SocketIO, send
import threading
import time
import asyncio
from app.init_nodes import InitNodes
import json
import sys

def run_master_node(masterParameters):
    initNode = InitNodes()
    InitNodes.change_system_info(masterParameters)
    initNode.master_socket(socketio)

def run_slave_node(slaveParameters, loop=None):
    initNode = InitNodes()
    InitNodes.change_slave_system_parameters(slaveParameters)
    initNode.slave_socket(socketio, loop)

def send_message_to_nodes(cad: str):
    socketio.send(cad, bradcast=True)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError("Not running in the werkzeug server")
    func()

master_thread = threading.Thread(target=run_master_node, args="")
slave_thread = threading.Thread(target=run_slave_node, args="")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'master-node'
socketio = SocketIO(app, logger=True, engineio_logger=True)

@app.route('/')
def index():
    return "this is a socketio test"

@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route('/test', methods=['GET'])
def testMSG(json):
    socketio.send('Te mando un mensaje, python',broadcast=True)
    return "hi"

@socketio.on('initMaster')
def initMaster(masterParameters):
    global master_thread
    if master_thread.is_alive():
        return "Another master is running"
    params = json.loads(masterParameters)
    master_thread = threading.Thread(target=run_master_node, args=(params,))
    master_thread.start()
    return "Master initialized"

@socketio.on('initSlave')
def initSlave(slaveParameters):
    global slave_thread
    if slave_thread.is_alive():
        return "Another slave is running"
    params = json.loads(slaveParameters)
    slave_thread = threading.Thread(target=run_slave_node, args=(params,asyncio.get_event_loop()))
    slave_thread.start()
    return "slave Initialized"

@socketio.on('tes')
def tes(msg):
    print("Socket test")
    socketio.send('Te mando un mensaje desde un socket, python',broadcast=True)
    return 'Hi im python'

@socketio.on('message')
def handleMessage(msg):
    print('Message:',msg)
    send('Llego un mensaje, python', broadcast=True)

@socketio.on('connect')
def connect():
    print('CONNECTED')

if __name__ == '__main__':
    if sys.argv[1] == 'master':
        socketio.run(app, port=5000)
    elif sys.argv[1] == 'slave':
        socketio.run(app, port=5001)
    else:
        print("No node type specified")
    