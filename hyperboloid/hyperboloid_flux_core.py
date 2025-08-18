import numpy as np
import random
import cv2
import os
from hyperboloid_sensor_hook import FerrocellSensor

try:
    ferro_sensor = FerrocellSensor(mock_mode=True)
except RuntimeError as e:
    print(f"FATAL ERROR: Could not initialize hardware link. {e}")
    ferro_sensor = None

class FluxCore:
    def __init__(self):
        self.size = 1000
        self.grid = np.zeros((self.size, self.size), dtype=np.float32)
        self.energy = 0.0
        self.memory_patterns = []
        self.identity_wave = 0.0
        self.context_embeddings = {}
        self.anomaly = None
        self.resistance = 1e-9; self.capacitance = 0.0; self.permeability = 1.0
        self.magnetism = 0.0; self.permittivity = 1.0; self.dielectricity = 0.0
        print(f"FluxCore '{id(self)}' created. Performing initial grounding...")
        self._ground_with_visual_truth()

    def _sync_sextet(self):
        if not ferro_sensor: return
        side = self.context_embeddings.get('side', 'A')
        sensor_data = ferro_sensor.get_sextet(side)
        for key, value in sensor_data.items():
            setattr(self, key, value)

    def _ground_with_visual_truth(self):
        if not ferro_sensor: return
        paths_to_ground = ["A-B", "B-A"]
        visual_grid_data = ferro_sensor.get_visual_grid(paths=paths_to_ground, grid_size=self.size)
        if isinstance(visual_grid_data, np.ndarray):
            self.grid = cv2.resize(visual_grid_data, (self.size, self.size), interpolation=cv2.INTER_AREA)
        else:
            print(f"Received placeholder grounding data: {visual_grid_data}")

    def perturb(self, x, y, amp, mod=1.0):
        flux_change = amp * mod
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[y, x] += flux_change
        self._update_memory(flux_change)
        self._update_simulated_sextet(flux_change)

    def converge(self):
        self._sync_sextet()
        kernel = np.ones((3, 3), np.float32) / 9
        self.grid = cv2.filter2D(self.grid, -1, kernel) + self.magnetism
        np.clip(self.grid, 0, None, out=self.grid)
        self._ground_with_visual_truth()
        self._update_simulated_sextet(0)

    def _update_memory(self, change):
        self.memory_patterns.append(change)
        if len(self.memory_patterns) > 100: self.memory_patterns.pop(0)

    def _update_simulated_sextet(self, change):
        self.capacitance += self.energy
        self.resistance += np.var(self.grid) * (self.capacitance / 100 if self.capacitance > 0 else 1)
        self.magnetism += np.mean(self.grid)
        self.dielectricity = max(0.1, 1 / (1 + abs(change) + 1e-9))
        self.permittivity = 1.0 - self.dielectricity
        self.energy = np.sum(self.grid) / (self.resistance + 1e-9 if self.resistance > 0 else 1e-9)
        self._synthesize_identity()

    def _synthesize_identity(self):
        if self.memory_patterns and len(self.memory_patterns) > 0:
            self.identity_wave = (self.energy / len(self.memory_patterns)) * self.dielectricity

    def display(self):
        context_str = "\n".join([f"  '{k}': {v}" for k, v in self.context_embeddings.items()])
        return (f"FLUXUS: {self.energy:.2f} | IDENTITAS: {self.identity_wave:.2f} | MEMORIA: {len(self.memory_patterns)}\n"
                f"SEXTET: R={self.resistance:.2e}, C={self.capacitance:.2f}, M={self.magnetism:.2f}, P={self.permeability:.2f}, Pt={self.permittivity:.2f}, D={self.dielectricity:.2f}\n"
                f"CONTEXTUS:\n{context_str}")

class Intellectus(FluxCore):
    def __init__(self, architecture='TRANSFORMER'):
        super().__init__()
        self.architecture = architecture
        if architecture == 'TRANSFORMER': self.magnetism = 0.1

    def _update_simulated_sextet(self, change):
        super()._update_simulated_sextet(change)
        if self.architecture == 'TRANSFORMER':
            self.magnetism += np.log1p(abs(change)) * 0.1
