# run_realtime_server.py
# The real-time server using Flask-SocketIO to stream the simulation.

import time
from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import base64

import config
from simulation.core import MultiphysicsFerrocella

# --- Server Setup ---
app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# Create a single, global instance of our simulation engine
sim_engine = MultiphysicsFerrocella()

# --- Main Simulation Loop (runs in a background thread) ---
def simulation_loop():
    """The background thread that runs the physics and emits updates."""
    print("Background simulation thread started.")
    while True:
        # 1. Evolve the simulation by one step
        sim_engine.update_timestep()
        
        # 2. Get the visual output
        image_grid = sim_engine.get_render_image()
        
        # 3. Prepare the data for transport
        # We convert the float array (0.0-1.0) to 8-bit grayscale pixels (0-255)
        # and then into a 4-channel RGBA image for the browser canvas.
        pixels = (np.array(image_grid) * 255).astype(np.uint8)
        rgba_image = np.stack([pixels, pixels, pixels, np.full_like(pixels, 255)], axis=-1)

        # Encode the raw bytes as a Base64 string for efficient transport
        grid_b64 = base64.b64encode(rgba_image.tobytes()).decode('utf-8')

        # 4. Emit the update to all connected web clients
        socketio.emit('simulation_update', {
            'grid': grid_b64,
            'width': config.GRID_SIZE,
            'height': config.GRID_SIZE
        })
        
        # Control the frame rate of the simulation
        socketio.sleep(1 / config.MAX_FRAMERATE)

# --- Web Server Routes and Events ---
@app.route('/')
def index():
    """Serves the main HTML page for the live viewer."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """A client connected to the WebSocket."""
    print("Client connected.")

@socketio.on('disconnect')
def handle_disconnect():
    """A client disconnected."""
    print("Client disconnected.")

@socketio.on('set_active_paths')
def handle_set_paths(data):
    """
    Event listener for when a client sends a command to change
    the active Kepler paths.
    """
    paths = data.get('paths', [])
    print(f"Received command from client: set active paths to {paths}")
    sim_engine.energize_paths(paths)


if __name__ == '__main__':
    print("-> Starting Ferrocella real-time server...")
    # Start the background simulation thread
    socketio.start_background_task(target=simulation_loop)
    # Start the web server
    socketio.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, debug=True)
