import time
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello world!'

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError("Not running in the werkzeug server")
    func()

@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == "__main__":
    app.run()