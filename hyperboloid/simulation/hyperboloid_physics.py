import json
import functools
import warnings
import time
import jax
import jax.numpy as jnp
import jax.extend.backend as xb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .hyperboloid_config import *

@functools.lru_cache(maxsize=1)
def load_toroid_topology(R=0.1, r=0.01, N=50):
    num_points = 1000
    theta = jnp.linspace(0, 2 * jnp.pi * N, num_points)
    pitch = 2 * jnp.pi * R / N
    helix1 = [[jnp.array([R * jnp.cos(t), R * jnp.sin(t), pitch * t / (2 * jnp.pi)]).tolist(),
               jnp.array([R * jnp.cos(t + delta), R * jnp.sin(t + delta), pitch * (t + delta) / (2 * jnp.pi)]).tolist()]
              for t in theta[:-1] for delta in [0.01]]
    helix2 = [[jnp.array([R * jnp.cos(t), R * jnp.sin(t), -pitch * t / (2 * jnp.pi)]).tolist(),
               jnp.array([R * jnp.cos(t + delta), R * jnp.sin(t + delta), -pitch * (t + delta) / (2 * jnp.pi)]).tolist()]
              for t in theta[:-1] for delta in [0.01]]
    return {
        'metadata': {'name': 'Toroid', 'grid_size': GRID_SIZE},
        'terminals': {'A': {'coords': [R, 0, 0]}, 'B': {'coords': [-R, 0, 0]}},
        'paths': {'A-B': {'segments': helix1}, 'B-A': {'segments': helix2}}
    }

def _pad_to_3d(point_array):
    if len(point_array) == 2: return jnp.append(point_array, 0.0)
    if len(point_array) == 3: return point_array
    raise ValueError(f"Invalid coordinate dimension: {len(point_array)}")

def get_path_segments(topology, path_key):
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

@jax.jit
def biot_savart_kernel(grid_points, dl_vectors, midpoints, current):
    assert grid_points.shape[-1] == 3 and dl_vectors.shape[-1] == 3
    assert dl_vectors.shape[0] == midpoints.shape[0]
    points_reshaped = grid_points[:, :, jnp.newaxis, :]
    midpoints_reshaped = midpoints[jnp.newaxis, jnp.newaxis, :, :]
    dl_reshaped = dl_vectors[jnp.newaxis, jnp.newaxis, :, :]
    r_vec = points_reshaped - midpoints_reshaped
    r_mag_sq = jnp.sum(r_vec**2, axis=-1)
    cross_product = jnp.cross(dl_reshaped, r_vec)
    denominator = jnp.maximum(r_mag_sq**(1.5), SINGULARITY_EPSILON)
    scaling_factor = MU_0 / (4 * jnp.pi) * current
    b_field_contributions = (scaling_factor * cross_product) / denominator[..., jnp.newaxis]
    return jnp.sum(b_field_contributions, axis=2)
