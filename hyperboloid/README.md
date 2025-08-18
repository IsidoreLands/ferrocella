# Hyperboloid Toroidal Ferrocell Simulator

Simulates a ferrofluid-filled toroid (R=0.1 m, r=0.01 m standard or R/φ aether) with double helix winding (N=50), 450 nm lasers, and sensors (Hall, LCR, photodiode) for 100 Mbps SL-PPM communication. Supports standard (Biot-Savart, Beer-Lambert) and aether (Phi ratios, dielectric acceleration) models.

## Setup
```bash
pip install -r hyperboloid_requirements.txt
```

# Run Simulation
```bash
python simulate_toroid.py  # Tests standard and aether modes, outputs BER and field visualizations
python hyperboloid_aether_os.py  # REPL: CREO 'SIDE_A', TOROID 'data', CONVERGO, OSTENDO
python experiments.py  # Auto-run tests
```


# Hardware Build

Components: 2x Raspberry Pi 4 ($35 ea), 2x Arduino Uno ($25 ea), 2x 450 nm laser diodes (Thorlabs PL450B, $150 ea), 2x avalanche photodiodes (Hamamatsu C5658, $200 ea), 2x Hall sensors (SS49E, $2 ea), 2x LCR modules (GY-405, $10 ea), EFH1 ferrofluid (50 mL, $50), acrylic toroid (3D-print, $50), 24 AWG copper wire ($10).
Steps:

3D-print toroid (R=0.1 m, r=0.01 m standard or 0.0618 m aether).
Wind double helix (clockwise A, counterclockwise B, N=50).
Fill with diluted EFH1 (1:10 kerosene + kaolin for 1.4 NTU).
Wire lasers/photodiodes to Pi GPIO, sensors to Arduino ADC/I2C.
Flash toroid_arduino.ino to Arduinos.
Run simulate_toroid.py or experiments.py with mock_mode=False in hyperboloid_sensor_hook.py.



# Experiments
Standard: Measure BER (<10^-5) for SL-PPM in turbid ferrofluid.
Aether: Test Phi-modulated geometry, dielectric flux effects.
Outputs: toroid_render_[standard|aether].png, aether_log.txt.

# Notes

Ensure /dev/ttyACM0 and /dev/ttyACM1 permissions for Arduinos.
Use laser goggles, sealed ferrofluid, low currents (<0.5 A).


#### New model_loader.py
```python
def load_player(model_key):
    """Mock model loader for INTERROGO command"""
    def get_response(prompt, session):
        return {"response": f"Mock response for {prompt}"}
    return get_response
```

