import unittest
import numpy as np
from simulation.hyperboloid_core import MultiphysicsFerrocella
from hyperboloid_sensor_hook import FerrocellSensor
from hyperboloid_aether_os import Contextus
from simulation.hyperboloid_config import *
import requests
import time
import asyncio

class TestHyperboloidSystem(unittest.TestCase):
    def setUp(self):
        self.sim = MultiphysicsFerrocella(mode='standard')
        self.sensor = FerrocellSensor(mock_mode=True)
        self.context = Contextus()
        self.sim.set_baseline('A', self.sensor.get_sextet('A'))
        self.sim.set_baseline('B', self.sensor.get_sextet('B'))

    def test_simulation_standard(self):
        data = list(map(int, '1011010110' * 10))
        self.sim.update_timestep(data_a_to_b=data, data_b_to_a=data[::-1])
        readings_a = self.sim.get_readings('A')
        readings_b = self.sim.get_readings('B')
        received_bits_a = []
        received_bits_b = []
        for _ in range(len(data) // 3):
            bits_a = self.sim._demod_sl_ppm(np.array([readings_a['laser']] * 10))
            bits_b = self.sim._demod_sl_ppm(np.array([readings_b['laser']] * 10))
            received_bits_a.extend(bits_a)
            received_bits_b.extend(bits_b)
        ber_a = np.mean(np.abs(np.array(data) - np.array(received_bits_a[:len(data)])))
        ber_b = np.mean(np.abs(np.array(data[::-1]) - np.array(received_bits_b[:len(data)])))
        self.assertLess(ber_a, 0.01, "Standard mode BER A->B too high")
        self.assertLess(ber_b, 0.01, "Standard mode BER B->A too high")

    def test_simulation_aether(self):
        self.sim.mode = 'aether'
        data = list(map(int, '1011010110' * 10))
        self.sim.update_timestep(data_a_to_b=data, data_b_to_a=data[::-1])
        readings_a = self.sim.get_readings('A')
        readings_b = self.sim.get_readings('B')
        received_bits_a = []
        received_bits_b = []
        for _ in range(len(data) // 3):
            bits_a = self.sim._demod_sl_ppm(np.array([readings_a['laser']] * 10))
            bits_b = self.sim._demod_sl_ppm(np.array([readings_b['laser']] * 10))
            received_bits_a.extend(bits_a)
            received_bits_b.extend(bits_b)
        ber_a = np.mean(np.abs(np.array(data) - np.array(received_bits_a[:len(data)])))
        ber_b = np.mean(np.abs(np.array(data[::-1]) - np.array(received_bits_b[:len(data)])))
        self.assertLess(ber_a, 0.01, "Aether mode BER A->B too high")
        self.assertLess(ber_b, 0.01, "Aether mode BER B->A too high")

    def test_aetheros_commands(self):
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(self.context.execute_command("CREO 'SIDE_A'"))
        self.assertIn("CREO MATERIAM 'SIDE_A'", response)
        response = loop.run_until_complete(self.context.execute_command("CREO 'SIDE_B'"))
        self.assertIn("CREO MATERIAM 'SIDE_B'", response)
        response = loop.run_until_complete(self.context.execute_command("SET_LASER SIDE 'A' PULSE 128"))
        self.assertIn("SET_LASER A PULSE 128 COMPLETE", response)
        response = loop.run_until_complete(self.context.execute_command("TOROID '1011010110'"))
        self.assertIn("TOROID standard", response)
        response = loop.run_until_complete(self.context.execute_command("READ_BER AETHER"))
        self.assertIn("READ_BER aether", response)

    def test_servers(self):
        time.sleep(1)
        response = requests.post(
            f"http://{SERVER_HOST}:{SERVER_PORT}/api/get_state",
            json={'paths': ['A-B'], 'mode': 'standard'}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('grid', data)
        self.assertIn('data_b64', data['grid'])
        response = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/api/readings/A")
        self.assertEqual(response.status_code, 200)
        self.assertIn('lcr', response.json())
        response = requests.post(
            f"http://{SERVER_HOST}:{SERVER_PORT}/api/set_laser/A",
            json={'pulse': 128}
        )
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
