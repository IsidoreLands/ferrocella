# run_dashboard_server.py
# The real-time server dedicated to the Three.js visual dashboard.
# FINAL VERSION with robust background thread management.

import requests
import numpy as np
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread
import base64
import cv2

# --- Configuration ---
PHYSICS_SERVER_URL = "http://ferrocella.ooda.wiki/api/get_state"
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 5001
DOWNSAMPLE_SIZE = 64

# --- Server Setup ---
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
socketio = SocketIO(app, cors_allowed_origins="*")
session = requests.Session()

# --- Global state for simulation control ---
# We use a simple dictionary to share state with the background thread.
sim_state = {'active_paths': []}

# --- Background Thread to Fetch and Process Data ---
def data_fetcher_loop():
    """Continuously fetches data from the physics server and pushes it to clients."""
    print("Background data fetcher thread started.")
    while True:
        try:
            # Read the current command from the shared state
            payload = {"paths": sim_state['active_paths'], "grid_size": 1000}
            
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
            # Push the update to the browser clients
            socketio.emit('force_field_update', {'field': force_field_flat})

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from physics server: {e}")
            socketio.emit('server_error', {'message': str(e)}) # Inform the client
        except Exception as e:
            print(f"An error occurred in the data loop: {e}")

        # Use the server's sleep function to be cooperative
        socketio.sleep(0.1)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@socketio.on('connect')
def h_connect(): print("Dashboard client connected.")

@socketio.on('set_active_paths')
def handle_set_paths_from_dashboard(data):
    # Update the shared state, which the background thread will read
    paths = data.get('paths', [])
    print(f"COMMAND from dashboard: set paths to {paths}")
    sim_state['active_paths'] = paths

if __name__ == '__main__':
    print("-> Starting Ferrocella Dashboard Server...")
    
    # --- THE FIX IS HERE ---
    # We manually create and start the background thread *before*
    # starting the blocking web server.
    data_thread = Thread(target=data_fetcher_loop)
    data_thread.daemon = True # Allows main thread to exit even if this one is running
    data_thread.start()
    # --- END FIX ---
    
    # Now, run the web server.
    socketio.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT)
