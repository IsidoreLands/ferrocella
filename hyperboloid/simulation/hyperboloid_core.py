import jax
import jax.numpy as jnp
import numpy as np
from .hyperboloid_physics import load_toroid_topology, get_path_segments, discretize_segments, biot_savart_kernel
from .hyperboloid_config import *

# Gaussian kernel for thermal diffusion
_kernel_size = 5
_sigma = 1.5
_x = jnp.arange(_kernel_size) - (_kernel_size - 1) / 2
_gaussian_kernel_1d = jnp.exp(-(_x**2) / (2 * _sigma**2))
DIFFUSION_KERNEL = jnp.outer(_gaussian_kernel_1d, _gaussian_kernel_1d)
DIFFUSION_KERNEL /= jnp.sum(DIFFUSION_KERNEL)

class MultiphysicsFerrocella:
    def __init__(self, mode='standard'):
        print(f"Initializing Toroidal Ferrocella Engine ({mode} mode)...")
        self.mode = mode
        self.size = GRID_SIZE
        self.R = R
        self.r = R_MINOR if mode == 'aether' else 0.01
        self.N = N_TURNS
        self.chi = 5
        self.alpha0 = 0.3 if mode == 'standard' else 0.3 / PHI
        self.k = 0.1
        self.L_tube = 2 * self.r
        self.slots = 4
        self.symbol_rate = 25e6
        self.a_d = A_D

        # State grids
        self.b_field = jnp.zeros((self.size, self.size, 3), dtype=PRECISION)
        self.temperature_grid = jnp.zeros((self.size, self.size), dtype=PRECISION)
        self.laser_intensity = {'A': 0, 'B': 0}
        self.lcr_reading = {'A': 0, 'B': 0}
        self.hall_reading = {'A': 0, 'B': 0}
        self.baseline = {'A': {'hall': 0, 'lcr': {'inductance': 0, 'capacitance': 0, 'resistance': 0}, 'laser': 0},
                         'B': {'hall': 0, 'lcr': {'inductance': 0, 'capacitance': 0, 'resistance': 0}, 'laser': 0}}

        # Toroid grid
        theta = jnp.linspace(0, 2 * jnp.pi, self.size)
        rho = jnp.linspace(0, self.r, self.size)
        tv, rv = jnp.meshgrid(theta, rho)
        self.grid_points = jnp.stack([
            (self.R + rv * jnp.cos(tv)) * jnp.cos(theta),
            (self.R + rv * jnp.cos(tv)) * jnp.sin(theta),
            rv * jnp.sin(tv)
        ], axis=-1)

        self.topology = load_toroid_topology(self.R, self.r, self.N)
        self.active_paths = {'A': [], 'B': []}
        self.laser_state = {'A': {'pulse': 0, 'color': 'blue'}, 'B': {'pulse': 0, 'color': 'blue'}}
        self.led_state = {'color': 'black', 'brightness': 0}
        print("Engine Initialized.")

    @jax.jit
    def _generate_sl_ppm(self, bits, symbol_time):
        slope_sign = 1 if bits[0] else -1
        position = int(''.join(map(str, bits[1:])), 2)
        t = jnp.linspace(0, symbol_time, 10)
        pulse = slope_sign * (t - symbol_time / 2) / (symbol_time / 2)
        pulse = jnp.roll(pulse, position * (len(t) // self.slots))
        if self.mode == 'aether':
            pulse *= PHI
        return pulse

    @jax.jit
    def _demod_sl_ppm(self, received_pulse):
        slope = received_pulse[5] - received_pulse[0]
        msb = 1 if slope > 0 else 0
        position = jnp.argmax(jnp.abs(received_pulse)) // (len(received_pulse) // self.slots)
        bits = [msb] + [int(b) for b in bin(position)[2:].zfill(2)]
        return bits

    @jax.jit
    def _update_thermal_step(self, temp_grid, heat_to_add):
        updated_grid = temp_grid + heat_to_add
        diffused_grid = jax.scipy.signal.convolve2d(updated_grid, DIFFUSION_KERNEL, mode='same')
        cooled_grid = diffused_grid * (1.0 - COOLING_RATE)
        return jnp.clip(cooled_grid, 0, 5.0)

    @jax.jit
    def _laser_transmission(self, I_in, B_mag, pulse_shape):
        M = self.chi * (B_mag / MU_0)
        alpha = self.alpha0 + self.k * jnp.abs(M) + 0.2 * jnp.random.normal(0, 1)
        if self.mode == 'aether':
            alpha /= PHI
            alpha += 0.1 / (1 + jnp.exp(-B_mag / PHI**3))
        I_out = I_in * jnp.exp(-alpha * self.L_tube)
        return I_out * pulse_shape

    def set_baseline(self, side, baseline):
        self.baseline[side] = baseline

    def update_timestep(self, data_a_to_b=None, data_b_to_a=None):
        heat_per_step = sum(self.laser_state[side]['pulse'] * ABSORPTION_COEFFICIENTS.get(self.laser_state[side]['color'], 0.1)
                            for side in ['A', 'B'])
        self.temperature_grid = self._update_thermal_step(self.temperature_grid, heat_per_step)

        for side, other, data in [('A', 'B', data_a_to_b), ('B', 'A', data_b_to_a)]:
            if data:
                symbols = [data[i:i+3] for i in range(0, len(data), 3)]
                for bits in symbols:
                    pulse = self._generate_sl_ppm(bits, 1/self.symbol_rate)
                    self.laser_state[side]['pulse'] = pulse
                    self.energize_paths(side, [side + '-' + other])

            segments = []
            for path_key in self.active_paths[side]:
                segments.extend(get_path_segments(self.topology, path_key))
            if segments:
                dl_vectors, midpoints = discretize_segments(segments)
                raw_b_field = biot_savart_kernel(
                    self.grid_points, dl_vectors, midpoints, KEPLER_GRID_CURRENT
                )
                susceptibility_grid = 1.0 + (self.temperature_grid * TEMP_SUSCEPTIBILITY_FACTOR)
                susceptibility_grid = jnp.clip(susceptibility_grid, 0.01, 1.0)
                self.b_field = raw_b_field * susceptibility_grid[..., jnp.newaxis]
            else:
                self.b_field = jnp.zeros_like(self.b_field)

            B_mag = jnp.linalg.norm(self.b_field, axis=-1)
            mu_eff = MU_0 * (1 + self.chi * (1 + jnp.tanh(B_mag / MU_0)))
            A = jnp.pi * self.r**2
            l = 2 * jnp.pi * self.R
            self.lcr_reading[other] = mu_eff * self.N**2 * A / l - self.baseline[other]['lcr']['inductance']
            sensor_point = self.grid_points[0, 0] if side == 'A' else self.grid_points[-1, 0]
            self.hall_reading[other] = jnp.linalg.norm(self.b_field[0 if side == 'A' else -1, 0]) + jnp.random.normal(0, 0.001) - self.baseline[other]['hall']
            pulse = self.laser_state[side]['pulse']
            I_out = self._laser_transmission(1.0, B_mag, pulse)
            self.laser_intensity[other] = I_out - self.baseline[other]['laser']

    def energize_paths(self, side, paths):
        self.active_paths[side] = paths

    def set_laser(self, side, pulse, color='blue'):
        self.laser_state[side] = {'pulse': pulse, 'color': color}

    def get_readings(self, side):
        return {
            'lcr': {'inductance': float(self.lcr_reading[side]), 'capacitance': float(self.baseline[side]['lcr']['capacitance']),
                    'resistance': float(self.baseline[side]['lcr']['resistance'])},
            'hall': float(self.hall_reading[side]),
            'laser': float(self.laser_intensity[side])
        }

    def get_render_image(self):
        b_perp_mag = jnp.sqrt(self.b_field[:,:,0]**2 + self.b_field[:,:,1]**2)
        b_total_mag = jnp.linalg.norm(self.b_field, axis=-1)
        brightness_map = jnp.divide(b_perp_mag, b_total_mag + 1e-9)
        if self.mode == 'aether':
            brightness_map *= PHI
        return brightness_map
