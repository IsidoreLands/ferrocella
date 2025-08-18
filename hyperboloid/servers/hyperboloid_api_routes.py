from flask import Blueprint, request, jsonify
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from simulation.hyperboloid_core import MultiphysicsFerrocella
from hyperboloid_sensor_hook import FerrocellSensor
from simulation.hyperboloid_config import *

api_bp = Blueprint('api', __name__)
sim = MultiphysicsFerrocella(mode='standard')
sensor = FerrocellSensor(mock_mode=True)

@api_bp.route('/get_state', methods=['POST'])
def get_state():
    data = request.get_json()
    if not data or 'paths' not in data:
        return jsonify({"error": "Request must include a 'paths' list."}), 400
    paths = data.get('paths', ['A-B'])
    mode = data.get('mode', 'standard')
    sim.mode = mode
    sim.set_baseline('A', sensor.get_sextet('A'))
    sim.set_baseline('B', sensor.get_sextet('B'))
    sim.update_timestep()
    b_field = sim.get_render_image()
    grid_data = base64.b64encode(b_field.tobytes()).decode('utf-8')
    return jsonify({
        'grid': {
            'data_b64': grid_data,
            'shape': b_field.shape,
            'dtype': str(b_field.dtype)
        }
    })

@api_bp.route('/readings/<side>', methods=['GET'])
def get_readings(side):
    if side not in ['A', 'B']:
        return jsonify({'error': 'Invalid side'}), 400
    return jsonify(sim.get_readings(side))

@api_bp.route('/set_laser/<side>', methods=['POST'])
def set_laser(side):
    if side not in ['A', 'B']:
        return jsonify({'error': 'Invalid side'}), 400
    pulse = request.json.get('pulse', 0)
    sensor.set_laser(side, pulse)
    sim.set_laser(side, pulse)
    return jsonify({'status': 'success'})
