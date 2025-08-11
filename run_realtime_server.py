# run_realtime_server.py
# The real-time server, now with full LED control.

import time
from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import base64
import config
from simulation.core import MultiphysicsFerrocella

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
sim_engine = MultiphysicsFerrocella()

def simulation_loop():
    # ... (this function is unchanged) ...
    print("Background simulation thread started.")
    while True:
        sim_engine.update_timestep()
        image_grid = sim_engine.get_render_image()
        # Normalization for display
        min_val, max_val = np.min(image_grid), np.max(image_grid)
        if max_val > min_val:
            image_grid = (image_grid - min_val) / (max_val - min_val)
        pixels = (np.array(image_grid) * 255).astype(np.uint8)
        rgba_image = np.stack([pixels, pixels, pixels, np.full_like(pixels, 255)], axis=-1)
        grid_b64 = base64.b64encode(rgba_image.tobytes()).decode('utf-8')
        socketio.emit('simulation_update', {
            'grid': grid_b64, 'width': sim_engine.size, 'height': sim_engine.size
        })
        socketio.sleep(1 / config.MAX_FRAMERATE)

@app.route('/')
def index(): return render_template('index.html')

@socketio.on('connect')
def h_connect(): print("Client connected.")

@socketio.on('disconnect')
def h_disconnect(): print("Client disconnected.")

@socketio.on('set_active_paths')
def h_set_paths(data):
    paths = data.get('paths', [])
    print(f"COMMAND: set active paths to {paths}")
    sim_engine.energize_paths(paths)

# --- NEW: Event handler for LED commands ---
@socketio.on('set_led_state')
def h_set_leds(data):
    """
    Event listener for when a client sends a command to change
    the LED color and brightness.
    """
    color = data.get('color', 'black')
    brightness = data.get('brightness', 0)
    print(f"COMMAND: set LEDs to {color} at brightness {brightness}")
    sim_engine.set_leds(color, brightness)
# --- END NEW ---


if __name__ == '__main__':
    print("-> Starting Ferrocella real-time server...")
    socketio.start_background_task(target=simulation_loop)
    socketio.run(
        app, host=config.SERVER_HOST, port=config.SERVER_PORT,
        allow_unsafe_werkzeug=True
    )
