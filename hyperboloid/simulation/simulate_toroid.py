import numpy as np
import matplotlib.pyplot as plt
from .hyperboloid_core import MultiphysicsFerrocella
from hyperboloid_sensor_hook import FerrocellSensor
from .hyperboloid_config import *

def run_simulation(mode='standard', data='1011010110'*10, turbidity_ntu=TURBIDITY_NTU):
    sim = MultiphysicsFerrocella(mode=mode)
    sensor = FerrocellSensor(mock_mode=True)
    sim.set_baseline('A', sensor.get_sextet('A'))
    sim.set_baseline('B', sensor.get_sextet('B'))
    data_bits = list(map(int, data))
    sim.update_timestep(data_a_to_b=data_bits, data_b_to_a=data_bits[::-1])
    received_bits_a = []
    received_bits_b = []
    for _ in range(len(data) // 3):
        readings_a = sim.get_readings('A')
        readings_b = sim.get_readings('B')
        pulse_a = readings_a['laser']
        pulse_b = readings_b['laser']
        bits_a = sim._demod_sl_ppm(np.array([pulse_a] * 10))
        bits_b = sim._demod_sl_ppm(np.array([pulse_b] * 10))
        received_bits_a.extend(bits_a)
        received_bits_b.extend(bits_b)
    ber_a = np.mean(np.abs(np.array(data_bits) - np.array(received_bits_a[:len(data_bits)])))
    ber_b = np.mean(np.abs(np.array(data_bits[::-1]) - np.array(received_bits_b[:len(data_bits)])))
    print(f"Mode: {mode}, BER A->B: {ber_a:.2e}, B->A: {ber_b:.2e}")
    img = sim.get_render_image()
    plt.imshow(img, cmap='viridis')
    plt.title(f"Toroid Field ({mode} mode, NTU={turbidity_ntu})")
    plt.savefig(f'toroid_render_{mode}.png')
    plt.close()

if __name__ == "__main__":
    run_simulation('standard')
    run_simulation('aether')
