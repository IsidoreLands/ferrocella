# run_realtime_server.py
# The server for Ferrocella, now upgraded to serve the AetherOS API.

from flask import Flask, request, jsonify, render_template
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

# --- AetherOS API Endpoint ---
@app.route('/api/get_state', methods=['POST'])
def get_aetheros_state():
    """
    This is the primary endpoint for AetherOS.
    It receives a command, runs one simulation step, and returns the result.
    """
    command = request.get_json()
    if not command:
        return jsonify({"error": "Request must be a JSON body."}), 400

    # 1. ACT: Update the simulation state based on the command from AetherOS
    paths_to_activate = command.get('paths', [])
    led_command = command.get('leds', {'color': 'black', 'brightness': 0})
    
    sim_engine.energize_paths(paths_to_activate)
    sim_engine.set_leds(led_command.get('color'), led_command.get('brightness'))

    # 2. SIMULATE: Evolve the simulation by one full time step
    sim_engine.update_timestep() # This is where electromagnetism, heat, etc. happen

    # 3. SENSE: Gather the results for AetherOS
    
    # Get the raw 1000x1000 grid data. This is the "visual" perception.
    raw_grid = sim_engine.get_render_image() # This is the unnormalized, physically accurate data
    
    # TODO: Implement a real sextet calculation based on the grid state
    sextet_data = {
        'resistance': np.var(raw_grid) * 1e5,
        'capacitance': np.mean(raw_grid) * 1e-5,
        'permeability': 1.0, # Placeholder
        'magnetism': np.mean(sim_engine.b_field[:,:,2]), # Z-component of B-field
        'permittivity': 1.0, # Placeholder
        'dielectricity': 0.0 # Placeholder
    }
    
    # Encode the raw grid for transport
    grid_b64 = base64.b64encode(np.array(raw_grid).astype(np.float32).tobytes()).decode('utf-8')

    return jsonify({
        "grid": {
            "shape": raw_grid.shape,
            "dtype": str(raw_grid.dtype),
            "data_b64": grid_b64
        },
        "sextet": sextet_data
    })


# --- The Live Dashboard (for humans) ---
# This will be broken for now, and we will create a new viewer for it later.
@app.route('/')
def index():
    return "<h1>AetherOS API Server is running.</h1><p>The visual dashboard is disabled.</p>"


if __name__ == '__main__':
    print("-> Starting Ferrocella AetherOS API Server...")
    # We don't need the background loop or SocketIO for this simple API
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
    )
