import time
from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import base64
from simulation.hyperboloid_config import *
from simulation.hyperboloid_core import MultiphysicsFerrocella
from hyperboloid_sensor_hook import FerrocellSensor

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
sim_engine = MultiphysicsFerrocella(mode='standard')
sensor = FerrocellSensor(mock_mode=True)

def simulation_loop():
    print("Background simulation thread started.")
    while True:
        sim_engine.update_timestep()
        image_grid = sim_engine.get_render_image()
        min_val, max_val = np.min(image_grid), np.max(image_grid)
        if max_val > min_val:
            image_grid = (image_grid - min_val) / (max_val - min_val)
        pixels = (np.array(image_grid) * 255).astype(np.uint8)
        rgba_image = np.stack([pixels, pixels, pixels, np.full_like(pixels, 255)], axis=-1)
        grid_b64 = base64.b64encode(rgba_image.tobytes()).decode('utf-8')
        socketio.emit('simulation_update', {
            'grid': grid_b64, 'width': sim_engine.size, 'height': sim_engine.size,
            'readings': {'A': sim_engine.get_readings('A'), 'B': sim_engine.get_readings('B')}
        })
        socketio.sleep(1 / MAX_FRAMERATE)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def h_connect():
    print("Client connected.")

@socketio.on('disconnect')
def h_disconnect():
    print("Client disconnected.")

@socketio.on('set_active_paths')
def h_set_paths(data):
    paths = data.get('paths', ['A-B'])
    mode = data.get('mode', 'standard')
    print(f"COMMAND: set active paths to {paths}, mode {mode}")
    sim_engine.mode = mode
    sim_engine.energize_paths('A' if 'A-B' in paths else 'B', paths)

@socketio.on('set_laser_state')
def h_set_laser(data):
    side = data.get('side', 'A')
    pulse = data.get('pulse', 0)
    mode = data.get('mode', 'standard')
    print(f"COMMAND: set laser {side} to pulse {pulse}, mode {mode}")
    sim_engine.mode = mode
    sensor.set_laser(side, pulse)
    sim_engine.set_laser(side, pulse)

if __name__ == '__main__':
    print("-> Starting Ferrocella Toroid Real-Time Server...")
    socketio.start_background_task(target=simulation_loop)
    socketio.run(app, host=SERVER_HOST, port=REALTIME_PORT, allow_unsafe_werkzeug=True)
