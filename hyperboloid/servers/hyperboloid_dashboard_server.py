import requests
import numpy as np
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread
import base64
import cv2
from simulation.hyperboloid_config import *

PHYSICS_SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/api/get_state"
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = DASHBOARD_PORT
DOWNSAMPLE_SIZE = 64

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
socketio = SocketIO(app, cors_allowed_origins="*")
session = requests.Session()
sim_state = {'active_paths': ['A-B'], 'mode': 'standard'}

def data_fetcher_loop():
    print("Background data fetcher thread started.")
    while True:
        try:
            payload = {"paths": sim_state['active_paths'], "grid_size": 1000, "mode": sim_state['mode']}
            response = session.post(PHYSICS_SERVER_URL, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            grid_info = data.get('grid', {})
            raw_bytes = base64.b64decode(grid_info['data_b64'])
            raw_grid = np.frombuffer(raw_bytes, dtype=np.float32).reshape(grid_info['shape'])
            force_field_mag = cv2.resize(raw_grid, (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), interpolation=cv2.INTER_AREA)
            grad_y, grad_x = np.gradient(force_field_mag)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_x = np.divide(grad_x, magnitude, out=np.zeros_like(grad_x), where=magnitude!=0)
            grad_y = np.divide(grad_y, magnitude, out=np.zeros_like(grad_y), where=magnitude!=0)
            force_field_flat = np.stack([grad_x, grad_y], axis=-1).ravel().tolist()
            for side in ['A', 'B']:
                readings = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/api/readings/{side}").json()
                socketio.emit('readings_update', {'side': side, 'readings': readings})
            socketio.emit('force_field_update', {'field': force_field_flat})
        except Exception as e:
            print(f"Error: {e}")
            socketio.emit('server_error', {'message': str(e)})
        socketio.sleep(0.1)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@socketio.on('connect')
def h_connect():
    print("Dashboard client connected.")

@socketio.on('set_active_paths')
def handle_set_paths_from_dashboard(data):
    paths = data.get('paths', ['A-B'])
    mode = data.get('mode', 'standard')
    print(f"COMMAND from dashboard: set paths to {paths}, mode {mode}")
    sim_state['active_paths'] = paths
    sim_state['mode'] = mode

@socketio.on('set_laser_state')
def handle_set_laser(data):
    side = data.get('side', 'A')
    pulse = data.get('pulse', 0)
    mode = data.get('mode', 'standard')
    print(f"COMMAND from dashboard: set laser {side} to pulse {pulse}, mode {mode}")
    sim_state['mode'] = mode
    requests.post(f"http://{SERVER_HOST}:{SERVER_PORT}/api/set_laser/{side}", json={'pulse': pulse})

if __name__ == '__main__':
    print("-> Starting Ferrocella Toroid Dashboard Server...")
    data_thread = Thread(target=data_fetcher_loop)
    data_thread.daemon = True
    data_thread.start()
    socketio.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT)
