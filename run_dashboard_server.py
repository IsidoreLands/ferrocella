# run_dashboard_server.py
# A real-time server dedicated to the Three.js visual dashboard.
# It acts as a client to the main physics server and serves the processed data.

import requests
import numpy as np
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread

# --- Configuration ---
# This is the address of the main AetherOS API server (running on Colab/ngrok)
# We can use your permanent URL.
PHYSICS_SERVER_URL = "http://ferrocella.ooda.wiki/api/get_state"
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 5001 # Run on a different port than the main server
DOWNSAMPLE_SIZE = 64 # The resolution of the force field we send to the browser

# --- Server Setup ---
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
socketio = SocketIO(app, cors_allowed_origins="*")
session = requests.Session()

# --- Background Thread to Fetch and Process Data ---
def data_fetcher_loop():
    """Continuously fetches data from the physics server and pushes it to clients."""
    active_paths = [] # We can control this later
    
    while True:
        try:
            # 1. Fetch data from the main physics server
            payload = {"paths": active_paths, "grid_size": 1000}
            response = session.post(PHYSICS_SERVER_URL, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            # 2. Decode the raw grid data
            grid_info = data.get('grid', {})
            raw_bytes = base64.b64decode(grid_info['data_b64'])
            raw_grid = np.frombuffer(raw_bytes, dtype=np.float32).reshape(grid_info['shape'])

            # 3. Process the data: Downsample and calculate force vectors
            # We use OpenCV's resize for efficient downsampling
            import cv2
            force_field_mag = cv2.resize(raw_grid, (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), interpolation=cv2.INTER_AREA)
            
            # Calculate the gradient (the direction of the force)
            grad_y, grad_x = np.gradient(force_field_mag)

            # Normalize the vectors
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            np.divide(grad_x, magnitude, out=grad_x, where=magnitude!=0)
            np.divide(grad_y, magnitude, out=grad_y, where=magnitude!=0)

            # 4. Emit the processed force field to the browser
            # We send it as a simple list of numbers [x1, y1, x2, y2, ...]
            force_field_flat = np.stack([grad_x, grad_y], axis=-1).ravel().tolist()
            socketio.emit('force_field_update', {'field': force_field_flat})

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from physics server: {e}")
            socketio.emit('server_error', {'message': str(e)})
        except Exception as e:
            print(f"An error occurred in the data loop: {e}")

        socketio.sleep(0.1) # Fetch data ~10 times per second


@app.route('/dashboard')
def dashboard():
    """Serves the main HTML page for the dashboard."""
    return render_template('dashboard.html') # Use a new HTML file

@socketio.on('connect')
def h_connect(): print("Dashboard client connected.")

if __name__ == '__main__':
    # Add opencv-python to requirements.txt!
    print("-> Starting Ferrocella Dashboard Server...")
    socketio.start_background_task(target=data_fetcher_loop)
    socketio.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT)
