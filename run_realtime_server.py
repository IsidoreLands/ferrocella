# run_realtime_server.py
# The real-time server using Flask-SocketIO to stream the simulation.

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
    print("Background simulation thread started.")
    while True:
        sim_engine.update_timestep()
        image_grid = sim_engine.get_render_image()
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

if __name__ == '__main__':
    print("-> Starting Ferrocella real-time server...")
    socketio.start_background_task(target=simulation_loop)
    
    # --- THE FIX IS HERE ---
    # We are acknowledging that we are using the development server.
    socketio.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        allow_unsafe_werkzeug=True
    )
    # --- END FIX ---
