# run_dashboard_server.py
# A real-time server dedicated to the Three.js visual dashboard.

import requests
import numpy as np
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread
import base64  # <-- THE MISSING IMPORT
import cv2     # <-- THE OTHER MISSING IMPORT

# --- Configuration ---
PHYSICS_SERVER_URL = "http://ferrocella.ooda.wiki/api/get_state"
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 5001
DOWNSAMPLE_SIZE = 64

# --- Server Setup ---
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
socketio = SocketIO(app, cors_allowed_origins="*")
session = requests.Session()

# --- Background Thread to Fetch and Process Data ---
def data_fetcher_loop():
    """Continuously fetches data from the physics server and pushes it to clients."""
    # We will control the simulation state from the dashboard itself
    active_paths = [] 
    
    while True:
        try:
            payload = {"paths": active_paths, "grid_size": 1000}
            response = session.post(PHYSICS_SERVER_URL, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Decode the raw grid data
            grid_info = data.get('grid', {})
            raw_bytes = base64.b64decode(grid_info['data_b64'])
            raw_grid = np.frombuffer(raw_bytes, dtype=np.float32).reshape(grid_info['shape'])

            # Process the data: Downsample and calculate force vectors
            force_field_mag = cv2.resize(raw_grid, (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), interpolation=cv2.INTER_AREA)
            grad_y, grad_x = np.gradient(force_field_mag)

            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            # Use a safe divide to avoid NaNs
            grad_x = np.divide(grad_x, magnitude, out=np.zeros_like(grad_x), where=magnitude!=0)
            grad_y = np.divide(grad_y, magnitude, out=np.zeros_like(grad_y), where=magnitude!=0)

            # Emit the processed force field to the browser
            force_field_flat = np.stack([grad_x, grad_y], axis=-1).ravel().tolist()
            socketio.emit('force_field_update', {'field': force_field_flat, 'paths': active_paths})

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from physics server: {e}")
            socketio.emit('server_error', {'message': str(e)})
        except Exception as e:
            print(f"An error occurred in the data loop: {e}")

        socketio.sleep(0.1)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@socketio.on('connect')
def h_connect(): print("Dashboard client connected.")

# New handler to allow the dashboard to control the simulation
@socketio.on('set_active_paths')
def handle_set_paths_from_dashboard(data):
    global active_paths
    paths = data.get('paths', [])
    print(f"COMMAND from dashboard: set paths to {paths}")
    active_paths = paths

if __name__ == '__main__':
    print("-> Starting Ferrocella Dashboard Server...")
    socketio.start_background_task(target=data_fetcher_loop)
    socketio.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT)
