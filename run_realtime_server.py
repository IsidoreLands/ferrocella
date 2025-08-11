# run_realtime_server.py
# The server for Ferrocella, now with correct data type conversion for JSON responses.

from flask import Flask, request, jsonify
import numpy as np
import base64
import config
from simulation.core import MultiphysicsFerrocella

# --- Server Setup ---
app = Flask(__name__) # Removed dashboard-specific folders for this API server

# Create a single, global instance of our simulation engine
sim_engine = MultiphysicsFerrocella()

# --- AetherOS API Endpoint ---
@app.route('/api/get_state', methods=['POST'])
def get_aetheros_state():
    """
    Receives a command, runs one simulation step, and returns the result.
    """
    command = request.get_json()
    if not command:
        return jsonify({"error": "Request must be a JSON body."}), 400

    # 1. ACT: Update simulation state from the command
    paths_to_activate = command.get('paths', [])
    led_command = command.get('leds', {'color': 'black', 'brightness': 0})
    sim_engine.energize_paths(paths_to_activate)
    sim_engine.set_leds(led_command.get('color'), led_command.get('brightness'))

    # 2. SIMULATE: Evolve the simulation
    sim_engine.update_timestep()

    # 3. SENSE: Gather results
    raw_grid = sim_engine.get_render_image()
    
    # --- THE FIX IS HERE ---
    # Convert JAX arrays to NumPy arrays before putting them in the dictionary.
    # The standard JSON encoder does not know how to handle JAX's 'ArrayImpl'.
    # We also convert single values to native Python floats.
    sextet_data = {
        'resistance': float(np.var(np.array(raw_grid)) * 1e5),
        'capacitance': float(np.mean(np.array(raw_grid)) * 1e-5),
        'permeability': 1.0,
        'magnetism': float(np.mean(np.array(sim_engine.b_field[:,:,2]))),
        'permittivity': 1.0,
        'dielectricity': 0.0
    }
    
    # Convert the main grid to a NumPy array for encoding
    numpy_grid = np.array(raw_grid, dtype=np.float32)
    grid_b64 = base64.b64encode(numpy_grid.tobytes()).decode('utf-8')

    return jsonify({
        "grid": {
            "shape": numpy_grid.shape,
            "dtype": str(numpy_grid.dtype),
            "data_b64": grid_b64
        },
        "sextet": sextet_data
    })
    # --- END FIX ---


# --- Root endpoint for health checks ---
@app.route('/')
def health_check():
    return "Ferrocella AetherOS API Server is running."

if __name__ == '__main__':
    print("-> Starting Ferrocella AetherOS API Server...")
    # Use Gunicorn or another production server to run this in production.
    # The `app.run` command is for local development and testing.
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT)
