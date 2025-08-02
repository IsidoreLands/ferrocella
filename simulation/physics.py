# simulation/physics.py
# Contains the low-level, high-performance physics functions powered by JAX.
# This is the final, hardened version for Phase 1.

import json
import functools
import warnings
import argparse
import time
import pathlib

import jax
import jax.numpy as jnp
import jax.extend.backend as xb
import matplotlib.pyplot as plt

import config

# --- Utility Functions ---

@functools.lru_cache(maxsize=1)
def load_kepler_topology():
    """Loads, validates, and caches the Kepler grid geometry from the asset file."""
    try:
        with open(config.KEPLER_ASSET_FILE, 'r') as f:
            topology = json.load(f)
        if 'terminals' not in topology or 'paths' not in topology:
            raise ValueError("Topology JSON must contain 'terminals' and 'paths' keys.")
        return topology
    except FileNotFoundError:
        warnings.warn(f"Asset file '{config.KEPLER_ASSET_FILE}' not found. Using a mock topology for testing.")
        return {
            "metadata": {"name": "Mock Kepler Grid", "grid_size": 1000},
            "terminals": {"A": {"coords": [0, 0]}, "B": {"coords": [780, 0]}},
            "paths": {"A-B": {"segments": [[[0, 0, 0], [780, 0, 0]]]}}
        }
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"FATAL: Failed to load or validate topology file: {e}")

def _pad_to_3d(point_array):
    """Helper to ensure a coordinate array is 3D."""
    if point_array.shape[0] == 2: return jnp.append(point_array, 0.0)
    if point_array.shape[0] == 3: return point_array
    raise ValueError(f"Invalid coordinate dimension: {point_array.shape[0]}")

def get_path_segments(topology, path_key):
    """Extracts the 3D line segments for a given path key (e.g., 'A-C')."""
    parts = path_key.upper().split('-')
    if len(parts) != 2: raise ValueError(f"Invalid path_key: '{path_key}'")
    start, end = parts
    path_key_fwd, path_key_rev = f"{start}-{end}", f"{end}-{start}"
    if path_key_fwd in topology['paths'] and topology['paths'][path_key_fwd].get('segments'):
        segments = topology['paths'][path_key_fwd]['segments']
    elif path_key_rev in topology['paths'] and topology['paths'][path_key_rev].get('segments'):
        segments = topology['paths'][path_key_rev]['segments']
    else:
        warnings.warn(f"Path '{path_key_fwd}' not found. Falling back to straight line.")
        p1 = jnp.array(topology['terminals'][start]['coords']); p2 = jnp.array(topology['terminals'][end]['coords'])
        segments = [[p1.tolist(), p2.tolist()]]
    return [[_pad_to_3d(jnp.array(p1)), _pad_to_3d(jnp.array(p2))] for p1, p2 in segments]

def discretize_segments(segments, resolution=5.0):
    """Breaks segments into discrete `dl` vectors and midpoints."""
    all_dl_vectors, all_midpoints = [], []
    if not segments:
        warnings.warn("No segments provided for discretization.")
        return jnp.empty((0, 3)), jnp.empty((0, 3))

    for start, end in segments:
        start_jax, end_jax = jnp.array(start), jnp.array(end)
        segment_length = jnp.linalg.norm(end_jax - start_jax)
        num_points = int(jnp.maximum(2, jnp.ceil(segment_length / resolution)))
        points = jnp.linspace(start_jax, end_jax, num_points)
        dl_vectors = jnp.diff(points, axis=0)
        midpoints = points[:-1] + dl_vectors / 2.0
        all_dl_vectors.append(dl_vectors)
        all_midpoints.append(midpoints)

    return jnp.concatenate(all_dl_vectors), jnp.concatenate(all_midpoints)


# --- Core Physics Solver ---
@jax.jit
def biot_savart_kernel(grid_points, dl_vectors, midpoints, current):
    """Computes the magnetic field (B-field) using the Biot-Savart law."""
    assert grid_points.shape[-1] == 3 and dl_vectors.shape[-1] == 3
    assert dl_vectors.shape[0] == midpoints.shape[0]
    
    points_reshaped = grid_points[:, :, jnp.newaxis, :]
    midpoints_reshaped = midpoints[jnp.newaxis, jnp.newaxis, :, :]
    dl_reshaped = dl_vectors[jnp.newaxis, jnp.newaxis, :, :]
    r_vec = points_reshaped - midpoints_reshaped
    r_mag_sq = jnp.sum(r_vec**2, axis=-1)
    cross_product = jnp.cross(dl_reshaped, r_vec)
    denominator = jnp.maximum(r_mag_sq**(1.5), config.SINGULARITY_EPSILON)
    scaling_factor = config.MU_0_OVER_4PI * current
    b_field_contributions = (scaling_factor * cross_product) / denominator[..., jnp.newaxis]
    return jnp.sum(b_field_contributions, axis=2)


# --- Standalone Test Harness ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone test harness for Ferrocella physics engine.")
    parser.add_argument("--paths", nargs='+', default=["A-B"], help="Path keys to energize (e.g., 'A-B' 'C-E').")
    parser.add_argument("--grid_size", type=int, default=200, help="Size of test grid for visualization.")
    parser.add_argument("--resolution", type=float, default=5.0, help="Discretization resolution.")
    parser.add_argument("--plot_type", choices=['heatmap', 'vector', 'magnitude', 'stream'], default='heatmap', help="Type of plot.")
    parser.add_argument("--output_file", type=str, default=None, help="Save B-field array to this .npy file.")
    parser.add_argument("--save_plot", type=str, default=None, help="Save plot to this file (e.g., 'output.png').")
    args = parser.parse_args()

    print("--- Running Standalone Physics Test ---")
    
    # --- Consultant Improvements: Warnings and Validation ---
    print(f"JAX Backend: {xb.get_backend().platform}")
    if xb.get_backend().platform == 'cpu':
        warnings.warn("Running on CPU, which may be slow for large grids. Consider enabling GPU support.")
    if args.grid_size > 500:
        warnings.warn(f"Large grid_size ({args.grid_size}) may consume significant memory.")
    if config.KEPLER_GRID_CURRENT <= 0: raise ValueError("KEPLER_GRID_CURRENT must be positive.")
    if config.SINGULARITY_EPSILON <= 0: raise ValueError("SINGULARITY_EPSILON must be positive.")

    topology = load_kepler_topology()
    all_dl_vectors, all_midpoints, all_segments = [], [], []

    print("Discretizing segments...")
    start_discretize = time.time()
    for path_key in args.paths:
        print(f"  - Processing path: '{path_key}'")
        segments = get_path_segments(topology, path_key)
        all_segments.extend(segments)
        dl_vectors, midpoints = discretize_segments(segments, resolution=args.resolution)
        all_dl_vectors.append(dl_vectors)
        all_midpoints.append(midpoints)
    print(f"Discretization completed in {time.time() - start_discretize:.4f} seconds.")

    if not all_segments or not any(dl.shape[0] > 0 for dl in all_dl_vectors):
        warnings.warn("No valid segments found for any path. Exiting."); exit()
        
    final_dl = jnp.concatenate(all_dl_vectors)
    final_midpoints = jnp.concatenate(all_midpoints)

    print(f"Total paths discretized into {final_dl.shape[0]} vector segments.")
    if final_dl.shape[0] != final_midpoints.shape[0]:
        raise ValueError(f"FATAL: Mismatch: {final_dl.shape[0]} dl_vectors vs {final_midpoints.shape[0]} midpoints.")

    x = y = jnp.linspace(0, config.GRID_SIZE, args.grid_size)
    xv, yv = jnp.meshgrid(x, y)
    grid_points = jnp.stack([xv, yv, jnp.zeros_like(xv)], axis=-1)

    print("Running JAX JIT-compiled Biot-Savart kernel (first run may be slow)...")
    start_time = time.time()
    b_field = biot_savart_kernel(grid_points, final_dl, final_midpoints, config.KEPLER_GRID_CURRENT)
    b_field.block_until_ready()
    print(f"Kernel execution complete in {time.time() - start_time:.4f} seconds.")

    if args.output_file:
        jnp.save(args.output_file, b_field)
        print(f"B-field data saved to '{args.output_file}'.")

    # --- Consultant Improvements: Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8.5)); ax.set_aspect('equal')
    ax.set_xlabel("Grid X-coordinate"); ax.set_ylabel("Grid Y-coordinate")
    plot_title = f"for Path(s): {', '.join(args.paths)}"
    
    if args.plot_type == 'heatmap':
        ax.set_title(f"Magnetic Field (Bz Component) {plot_title}")
        im = ax.imshow(b_field[:, :, 2].T, origin='lower', cmap='seismic', extent=[0, config.GRID_SIZE, 0, config.GRID_SIZE])
        fig.colorbar(im, ax=ax, label="B-Field Strength (Bz) in Tesla", shrink=0.8, format='%.2e')
    # ... (other plot types remain the same) ...

    # Overlay path and terminal labels
    for start, end in all_segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', lw=2, alpha=0.8)
    for terminal, data in topology['terminals'].items():
        ax.text(data['coords'][0], data['coords'][1] + 15, terminal, fontsize=12, color='black', weight='bold', ha='center')
        ax.plot(data['coords'][0], data['coords'][1], 'go', markersize=8)

    if args.save_plot:
        plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to '{args.save_plot}'.")
    
    print("Displaying visualization plot..."); plt.show()
