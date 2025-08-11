# simulation/core.py
# This file contains the main simulation controller class.

import jax.numpy as jnp

from . import physics  # Use a relative import within the package
import config

class MultiphysicsFerrocella:
    """
    The main controller for the Ferrocella simulation.
    This class holds the complete state of the simulation (magnetic, thermal, etc.)
    and orchestrates the physics updates for each time step.
    """

    def __init__(self):
        print("Initializing Multiphysics Ferrocella Engine...")
        self.size = config.GRID_SIZE
        
        # --- State Grids ---
        # Magnetic Field (Bx, By, Bz components)
        self.b_field = jnp.zeros((self.size, self.size, 3), dtype=config.PRECISION)
        
        # --- (Future) State Grids for Phase 2 ---
        # self.temperature_grid = jnp.zeros((self.size, self.size), dtype=config.PRECISION)
        # self.velocity_u = jnp.zeros((self.size, self.size), dtype=config.PRECISION) # Fluid vel X
        # self.velocity_v = jnp.zeros((self.size, self.size), dtype=config.PRECISION) # Fluid vel Y
        # self.pressure = jnp.zeros((self.size, self.size), dtype=config.PRECISION)
        
        # --- Control State ---
        self.active_paths = [] # List of energized Kepler paths, e.g., [('A', 'B')]
        self.led_state = {'color': 'black', 'brightness': 0}
        
        # --- Pre-computation ---
        # Load topology once
        self.topology = physics.load_kepler_topology()
        # Create the grid coordinate system once for calculations
        x = y = jnp.linspace(0, self.size, self.size)
        xv, yv = jnp.meshgrid(x, y)
        self.grid_points = jnp.stack([xv, yv, jnp.zeros_like(xv)], axis=-1)

        print("Engine Initialized.")

    def update_timestep(self):
        """
        Executes one full step of the multi-physics simulation.
        This method will become the heart of our simulation loop.
        """
        # --- 1. Electromagnetism ---
        # Get the geometry of the currently energized wires
        all_segments = []
        for path_key_tuple in self.active_paths:
            path_key = f"{path_key_tuple[0]}-{path_key_tuple[1]}"
            segments = physics.get_path_segments(self.topology, path_key)
            all_segments.extend(segments)

        # Discretize the geometry
        if all_segments:
            dl_vectors, midpoints = physics.discretize_segments(all_segments)
        else:
            # If no paths are active, the B-field from wires is zero
            self.b_field = jnp.zeros_like(self.b_field)
            return # End of this timestep

        # Calculate the B-field from the energized wires
        self.b_field = physics.biot_savart_kernel(
            self.grid_points, dl_vectors, midpoints, config.KEPLER_GRID_CURRENT
        )

        # --- 2. (Future) Thermal Simulation ---
        # Here we will add the code to update the temperature_grid
        
        # --- 3. (Future) Fluid Dynamics (MHD) ---
        # Here we will add the code to update the fluid velocity and pressure

    def get_render_image(self):
        """
        Calculates the final visual image based on the current state.
        For now, this is just the optical effect of the B-field.
        """
        # The optical effect: brightness is high when the field is perpendicular to view (XY plane)
        b_x, b_y, b_z = self.b_field[:,:,0], self.b_field[:,:,1], self.b_field[:,:,2]
        
        b_perp_mag = jnp.sqrt(b_x**2 + b_y**2)
        b_total_mag = jnp.linalg.norm(self.b_field, axis=-1)
        
        brightness_map = jnp.divide(b_perp_mag, b_total_mag + 1e-9) # Add epsilon for safety
        
        # Normalize to 0-1 range
        max_val = jnp.max(brightness_map)
        if max_val > 0:
            brightness_map /= max_val
            
        return brightness_map

    # --- Methods to Control the Simulation ---
    def energize_paths(self, paths_to_activate: list):
        """Sets which Kepler Grid paths are currently active."""
        self.active_paths = [tuple(p.upper().split('-')) for p in paths_to_activate]

    def set_leds(self, color: str, brightness: int):
        """Sets the state of the LED array."""
        self.led_state = {'color': color, 'brightness': brightness}
