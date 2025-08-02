# config.py
# The single source of truth for all physical constants and simulation parameters
# for the Ferrocella project.

# --- Grid & Simulation Parameters ---
GRID_SIZE = 1000         # The simulation resolution matches the physical design units.
TIME_STEP = 0.05         # Duration of a single physics update step (in seconds).
PRECISION = 'float32'    # Data type for grids ('float32' is good for GPUs).

# --- NEW: Configurable threshold for singularity handling in Biot-Savart Law ---
SINGULARITY_EPSILON = 1e-9 # A small number to prevent division by zero on the wire.

# --- Physical Constants (based on an 18" x 18" design) ---
PHYSICAL_SIZE_M = 0.4572  # 18 inches in meters.
FLUID_VISCOSITY = 0.08    # A property of the ferrofluid (Pa·s). Tunable.
FLUID_DENSITY = 1200.0    # kg/m^3. Typical for oil-based ferrofluid.
FLUID_MAGNETIZATION = 2.0 # How strongly the fluid reacts to field gradients. Tunable.

# --- Kepler Control Grid Parameters ---
KEPLER_ASSET_FILE = 'assets/kepler_grid_topology.json'
MU_0_OVER_4PI = 1e-7      # The magnetic constant (T·m/A).
KEPLER_GRID_CURRENT = 1.0 # The electric current (Amperes). Tunable.

# --- LED Array & Thermal Parameters ---
LED_BRIGHTNESS = 150  # Default brightness (0-255).
ABSORPTION_COEFFICIENTS = {
    'red': 0.8, 'green': 0.4, 'blue': 0.2,
    'white': 0.5, 'black': 0.0
}
TEMP_SUSCEPTIBILITY_FACTOR = -0.15 # Tunable.
THERMAL_DIFFUSIVITY = 0.1     # How quickly heat spreads through the fluid.
COOLING_RATE = 0.02           # How quickly the fluid loses heat to the environment.

# --- API & Server Parameters ---
SERVER_HOST = '0.0.0.0'       # Host on all network interfaces.
SERVER_PORT = 5000            # Port for the API server.
MAX_FRAMERATE = 20            # Max updates per second to send to web clients.
