# simulation/physics.py
# Contains the low-level, high-performance physics functions powered by JAX.
# This version incorporates all major feedback from the v2 consultant review.

import json
import functools
import warnings
import argparse

import jax
import jax.numpy as jnp
import jax.lib.xla_bridge as xb
import matplotlib.pyplot as plt

import config

# --- Utility Functions ---

@functools.lru_cache(maxsize=1)
def load_kepler_topology():
    """
    Loads, validates, and caches the Kepler grid geometry from the asset file.
    IMPROVED: Now validates the presence of required keys.
    """
    try:
        with open(config.KEPLER_ASSET_FILE, 'r') as f:
            topology = json.load(f)
        # Validate structure
        if 'terminals' not in topology or 'paths' not in topology:
            raise ValueError("Topology JSON must contain 'terminals' and 'paths' keys.")
        return topology
    except FileNotFoundError:
        raise RuntimeError(f"FATAL: Asset file not found at '{config.KEPLER_ASSET_FILE}'")
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"FATAL: Failed to load or validate topology file: {e}")

def _pad_to_3d(point_array):
    """Helper to ensure a coordinate array is 3D."""
    if point_array.shape[0] == 2:
        return jnp.append(point_array, 0.0)
    return point_array

def get_path_segments(topology, path_key):
    """
    Extracts the 3D line segments for a given path key (e.g., 'A-C').
    """
    # ... (Implementation unchanged from previous version, as it was already robust) ...
    # This function correctly parses keys and pads to 3D.
    parts = path_key.upper().split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid path_key format: '{path_key}'. Expected 'START-END'.")
    
    start_terminal, end_terminal = parts
    path_key_fwd = f"{start_terminal}-{end_terminal}"
    path_key_rev = f"{end_terminal}-{start_terminal}"

    if path_key_fwd in topology['paths'] and topology['paths'][path_key_fwd]['segments']:
        segments = topology['paths'][path_key_fwd]['segments']
    elif path_key_rev in topology['paths'] and topology['paths'][path_key_rev]['segments']:
        segments = topology['paths'][path_key_rev]['segments']
    else:
        warnings.warn(f"Path '{path_key_fwd}' not found. Falling back to straight line.")
        try:
            p1 = jnp.array(topology['terminals'][start_terminal]['coords'], dtype=config.PRECISION)
            p2 = jnp.array(topology['terminals'][end_terminal]['coords'], dtype=config.PRECISION)
            segments = [[p1.tolist(), p2.tolist()]]
        except KeyError as e:
            raise ValueError(f"Terminal '{e}' not found for path '{path_key_fwd}'.")

    return [[_pad_to_3d(jnp.array(p1)), _pad_to_3d(jnp.array(p2))] for p1, p2 in segments]

@functools.partial(jax.jit, static_argnames=("resolution",))
def discretize_segments_vmap(segments, resolution):
    """
    Breaks segments into discrete `dl` vectors and midpoints for integration.
    IMPROVED: This function is now JIT-compiled and uses jax.vmap for
              significantly better performance on many segments.
    """
    if not segments:
        return jnp.empty((0, 3)), jnp.empty((0, 3))

    # Define the operation for a single segment
    def process_segment(segment):
        start, end = segment
        segment_length = jnp.linalg.norm(end - start)
        num_points = jnp.maximum(2, jnp.ceil(segment_length / resolution).astype(int))
        points = jnp.linspace(start, end, num_points)
        dl_vectors = jnp.diff(points, axis=0)
        midpoints = points[:-1] + dl_vectors / 2.0
        return dl_vectors, midpoints

    # Use vmap to apply the function over all segments in a batch.
    # This is much faster than a Python for-loop.
    all_dl_vectors, all_midpoints = jax.vmap(process_segment)(jnp.array(segments))
    
    # vmap adds a leading batch dimension, so we concatenate the results.
    return jnp.concatenate(all_dl_vectors), jnp.concatenate(all_midpoints)


# --- Core Physics Solver ---

@jax.jit
def biot_savart_kernel(grid_points, dl_vectors, midpoints, current):
    """
    Computes the magnetic field (B-field) using the Biot-Savart law.
    This JIT-compiled kernel is the performance-critical heart of the simulation.
    """
    assert grid_points.shape[-1] == 3
    assert dl_vectors.shape[-1] == 3
    
    points_reshaped = grid_points[:, :, jnp.newaxis, :]
    midpoints_reshaped = midpoints[jnp.newaxis, jnp.newaxis, :, :]
    dl_reshaped = dl_vectors[jnp.newaxis, jnp.newaxis, :, :]

    r_vec = points_reshaped - midpoints_reshaped
    r_mag_sq = jnp.sum(r_vec**2, axis=-1)
    
    cross_product = jnp.cross(dl_reshaped, r_vec)
    
    denominator = r_mag_sq**(1.5)
    denominator = jnp.maximum(denominator, config.SINGULARITY_EPSILON)
    
    scaling_factor = config.MU_0_OVER_4PI * current
    b_field_contributions = (scaling_factor * cross_product) / denominator[..., jnp.newaxis]
    
    return jnp.sum(b_field_contributions, axis=2)

# --- Standalone Test Harness ---
if __name__ == '__main__':
    # IMPROVED: Use argparse for a flexible command-line interface.
    parser = argparse.ArgumentParser(description="Standalone test harness for Ferrocella physics engine.")
    parser.add_argument("--paths", nargs='+', default=["A-B"], 
                        help="One or more path keys to energize (e.g., 'A-B' 'C-D').")
    parser.add_argument("--grid_size", type=int, default=200, 
                        help="Size of the test grid (e.g., 200 for a 200x200 visual).")
    parser.add_argument("--resolution", type=float, default=5.0, 
                        help="Discretization resolution for wire segments.")
    parser.add_argument("--plot_type", choices=['heatmap', 'vector'], default='heatmap',
                        help="Type of plot to generate ('heatmap' for Bz, 'vector' for Bxy).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="If provided, saves the B-field array to this .npy file.")
    args = parser.parse_args()

    print("--- Running Standalone Physics Test ---")
    print(f"JAX Backend: {xb.get_backend().platform}")

    topology = load_kepler_topology()
    print(f"Loaded topology for '{topology['metadata']['name']}'.")

    all_dl_vectors, all_midpoints = [], []
    all_segments = []
    
    # IMPROVED: Handle multiple paths
    for path_key in args.paths:
        print(f"Processing path: '{path_key}'...")
        segments = get_path_segments(topology, path_key)
        all_segments.extend(segments)
        dl_vectors, midpoints = discretize_segments_vmap(segments, resolution=args.resolution)
        all_dl_vectors.append(dl_vectors)
        all_midpoints.append(midpoints)

    final_dl = jnp.concatenate(all_dl_vectors)
    final_midpoints = jnp.concatenate(all_midpoints)
    print(f"Total paths discretized into {final_dl.shape[0]} vector segments.")

    x = y = jnp.linspace(0, config.GRID_SIZE, args.grid_size)
    xv, yv = jnp.meshgrid(x, y)
    zv = jnp.zeros_like(xv)
    grid_points = jnp.stack([xv, yv, zv], axis=-1)

    print("Running JAX JIT-compiled Biot-Savart kernel (first run may be slow)...")
    b_field = biot_savart_kernel(grid_points, final_dl, final_midpoints, config.KEPLER_GRID_CURRENT)
    # Block until the computation is actually finished to get accurate timing.
    b_field.block_until_ready()
    print("Kernel execution complete.")

    if args.output_file:
        print(f"Saving B-field data to '{args.output_file}'...")
        jnp.save(args.output_file, b_field)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_aspect('equal')
    ax.set_xlabel("Grid X-coordinate")
    ax.set_ylabel("Grid Y-coordinate")

    if args.plot_type == 'heatmap':
        ax.set_title(f"Magnetic Field (Bz Component) for Path(s): {', '.join(args.paths)}")
        bz_component = b_field[:, :, 2]
        im = ax.imshow(bz_component.T, origin='lower', cmap='seismic', 
                       extent=[0, config.GRID_SIZE, 0, config.GRID_SIZE])
        fig.colorbar(im, ax=ax, label="Magnetic Field Strength (Bz) in Tesla", shrink=0.8)

    elif args.plot_type == 'vector':
        ax.set_title(f"Magnetic Field (In-Plane Vector Field) for Path(s): {', '.join(args.paths)}")
        bx, by = b_field[:, :, 0], b_field[:, :, 2] # Swapped y and z for top-down view of in-plane field
        
        # Subsample the grid for a clearer quiver plot
        skip = max(1, args.grid_size // 25)
        ax.quiver(xv[::skip, ::skip], yv[::skip, ::skip], 
                  bx[::skip, ::skip], by[::skip, ::skip],
                  color='blue', scale=None, scale_units='xy')
    
    # Overlay the wire paths for context
    for start, end in all_segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', lw=2)

    print("Displaying visualization plot...")
    plt.show()
