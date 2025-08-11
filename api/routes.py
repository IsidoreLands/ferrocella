# api/routes.py
# This file defines the API endpoints for the Ferrocella server.

from flask import Blueprint, request, jsonify
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Import our simulation engine and configuration
from simulation import physics
import config

# Create a "Blueprint" for our API to keep routes organized.
api_bp = Blueprint('api', __name__)

# Load the topology once at server startup for efficiency.
print("API: Loading Kepler topology...")
topology = physics.load_kepler_topology()
print("API: Topology loaded successfully.")

@api_bp.route('/field', methods=['POST'])
def get_field_image():
    """
    API endpoint to calculate and return a static image of the B-field.
    Expects JSON: {"paths": ["A-B", "C-E"], "grid_size": 300}
    """
    data = request.get_json()
    if not data or 'paths' not in data:
        return jsonify({"error": "Request must include a 'paths' list."}), 400
    
    paths_to_sim = data.get('paths')
    vis_grid_size = data.get('grid_size', 200)

    try:
        all_segments = []
        all_dl_vectors = []
        all_midpoints = []
        for path_key in paths_to_sim:
            segments = physics.get_path_segments(topology, path_key)
            all_segments.extend(segments)
            dl, mid = physics.discretize_segments(segments)
            all_dl_vectors.append(dl)
            all_midpoints.append(mid)

        if not all_segments:
            return jsonify({"error": "No valid segments found."}), 400

        final_dl = jnp.concatenate(all_dl_vectors)
        final_midpoints = jnp.concatenate(all_midpoints)
        
        x = y = jnp.linspace(0, config.GRID_SIZE, vis_grid_size)
        xv, yv = jnp.meshgrid(x, y)
        grid_points = jnp.stack([xv, yv, jnp.zeros_like(xv)], axis=-1)
        
        b_field = physics.biot_savart_kernel(grid_points, final_dl, final_midpoints, config.KEPLER_GRID_CURRENT)
        bz_component = b_field[:, :, 2]

        # Generate the plot image in memory
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.imshow(np.array(bz_component).T, origin='lower', cmap='seismic')
        ax.axis('off')
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return jsonify({
            "image_png_base64": img_base64,
            "info": { "paths_processed": paths_to_sim, "num_dl_vectors": final_dl.shape[0] }
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during simulation: {e}"}), 500
