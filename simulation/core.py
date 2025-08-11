# simulation/core.py
# This file contains the main simulation controller class.
# VERSION 2: Now includes the Thermal Layer.

import jax
import jax.numpy as jnp

from . import physics
import config

# We'll use a simple Gaussian kernel for heat diffusion
# This is pre-calculated once for performance.
_kernel_size = 5
_sigma = 1.5
_x = jnp.arange(_kernel_size) - (_kernel_size - 1) / 2
_gaussian_kernel_1d = jnp.exp(-(_x**2) / (2 * _sigma**2))
DIFFUSION_KERNEL = jnp.outer(_gaussian_kernel_1d, _gaussian_kernel_1d)
DIFFUSION_KERNEL /= jnp.sum(DIFFUSION_KERNEL)


class MultiphysicsFerrocella:
    """
    The main controller for the Ferrocella simulation.
    VERSION 2: Now includes a thermal simulation.
    """

    def __init__(self):
        print("Initializing Multiphysics Ferrocella Engine (v2)...")
        self.size = config.GRID_SIZE
        
        # --- State Grids ---
        self.b_field = jnp.zeros((self.size, self.size, 3), dtype=config.PRECISION)
        
        # NEW: The temperature grid holds the heat value at each point
        self.temperature_grid = jnp.zeros((self.size, self.size), dtype=config.PRECISION)
        
        # --- Control State ---
        self.active_paths = []
        self.led_state = {'color': 'black', 'brightness': 0}
        
        # --- Pre-computation ---
        self.topology = physics.load_kepler_topology()
        x = y = jnp.linspace(0, self.size, self.size)
        xv, yv = jnp.meshgrid(x, y)
        self.grid_points = jnp.stack([xv, yv, jnp.zeros_like(xv)], axis=-1)

        print("Engine Initialized.")

    @jax.jit
    def _update_thermal_step(self, temp_grid, heat_to_add):
        """A JIT-compiled function for one step of the heat equation."""
        # 1. Add heat from the LED source
        updated_grid = temp_grid + heat_to_add
        
        # 2. Diffuse the heat (spread it out) using convolution
        # We use 'same' padding to keep the grid size constant.
        diffused_grid = jax.scipy.signal.convolve2d(
            updated_grid, DIFFUSION_KERNEL, mode='same'
        )
        
        # 3. Cool the grid (lose heat to the environment)
        cooled_grid = diffused_grid * (1.0 - config.COOLING_RATE)
        
        # Clamp the temperature to a reasonable range to prevent instability
        return jnp.clip(cooled_grid, 0, 5.0)


    def update_timestep(self):
        """
        Executes one full step of the multi-physics simulation.
        """
        # --- 1. Thermal Simulation ---
        # Calculate how much heat the current LED state adds to the entire grid
        heat_source_strength = config.ABSORPTION_COEFFICIENTS.get(self.led_state['color'], 0.0)
        heat_per_step = heat_source_strength * (self.led_state['brightness'] / 255.0)
        
        # Run one step of the heat equation
        self.temperature_grid = self._update_thermal_step(self.temperature_grid, heat_per_step)
        

        # --- 2. Electromagnetism ---
        # Discretize the currently energized wire paths
        all_segments = []
        for path_key_tuple in self.active_paths:
            path_key = f"{path_key_tuple[0]}-{path_key_tuple[1]}"
            segments = physics.get_path_segments(self.topology, path_key)
            all_segments.extend(segments)

        if all_segments:
            dl_vectors, midpoints = physics.discretize_segments(all_segments)
            # Calculate the raw B-field from the wires
            raw_b_field = physics.biot_savart_kernel(
                self.grid_points, dl_vectors, midpoints, config.KEPLER_GRID_CURRENT
            )
        else:
            raw_b_field = jnp.zeros_like(self.b_field)
            
        # THE "CROOKES RADIOMETER" EFFECT:
        # Modulate the magnetic field based on the temperature.
        # Higher temperature reduces the fluid's magnetic susceptibility.
        susceptibility_grid = 1.0 + (self.temperature_grid * config.TEMP_SUSCEPTIBILITY_FACTOR)
        np.clip(susceptibility_grid, 0.01, 1.0, out=susceptibility_grid)
        
        # The final, effective B-field is the raw field weakened by heat
        self.b_field = raw_b_field * susceptibility_grid[..., jnp.newaxis]


    def get_render_image(self):
        """Calculates the final visual image based on the current state."""
        b_x, b_y, b_z = self.b_field[:,:,0], self.b_field[:,:,1], self.b_field[:,:,2]
        b_perp_mag = jnp.sqrt(b_x**2 + b_y**2)
        b_total_mag = jnp.linalg.norm(self.b_field, axis=-1)
        brightness_map = jnp.divide(b_perp_mag, b_total_mag + 1e-9)
        return brightness_map # Return the raw, unnormalized data

    # --- Methods to Control the Simulation ---
    def energize_paths(self, paths_to_activate: list):
        self.active_paths = [tuple(p.upper().split('-')) for p in paths_to_activate]

    def set_leds(self, color: str, brightness: int):
        self.led_state = {'color': color.lower(), 'brightness': brightness or 0}
