#!/usr/bin/env python3
import threading
import time
import random
import numpy as np
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
try:
    import cv2
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

class FerrocellSensor:
    def __init__(self, mock_mode=True, resolution=(128, 128)):
        self.mock_mode = mock_mode
        self.resolution = resolution
        self.calibration_interval = 3600
        self.last_calibration_time = 0
        self.sides = {'A': {'port': '/dev/ttyACM0'}, 'B': {'port': '/dev/ttyACM1'}}

        # Hardware Initialization
        if not mock_mode:
            if SERIAL_AVAILABLE:
                try:
                    for side in self.sides:
                        self.sides[side]['ser'] = serial.Serial(self.sides[side]['port'], 9600, timeout=1)
                        print(f"INFO: Connected to {side}: {self.sides[side]['port']}")
                except Exception as e:
                    print(f"WARN: Serial connection failed: {e}")
            if CAMERA_AVAILABLE:
                try:
                    self.camera = PiCamera()
                    self.camera.resolution = resolution
                    self.raw_capture = PiRGBArray(self.camera, size=resolution)
                    time.sleep(2)
                    print(f"INFO: PiCamera initialized at {resolution}")
                except Exception as e:
                    print(f"WARN: PiCamera failed: {e}")
                    if hasattr(self, 'camera') and self.camera: self.camera.close()
                    self.camera = None
            else:
                print("WARN: 'picamera' or 'opencv-python' not found.")

        # Data Attributes
        self.sextet = {'A': {'resistance': 1e-9, 'capacitance': 0.0, 'permeability': 1.0, 'magnetism': 0.0, 'permittivity': 1.0, 'dielectricity': 0.0, 'laser': 0.0},
                       'B': {'resistance': 1e-9, 'capacitance': 0.0, 'permeability': 1.0, 'magnetism': 0.0, 'permittivity': 1.0, 'dielectricity': 0.0, 'laser': 0.0}}
        self.visual_grid = np.zeros(resolution, dtype=np.float32)
        self._capture_baselines()
        self.thread = threading.Thread(target=self._poll_sensors, daemon=True)
        self.thread.start()
        print("INFO: FerrocellSensor thread started.")

    def _capture_baselines(self):
        print("INFO: Capturing baselines...")
        for side in ['A', 'B']:
            self.sextet[side].update(self._read_arduino(side))
        if self.camera:
            try:
                self.camera.capture(self.raw_capture, format="bgr", use_video_port=True)
                self.visual_grid = cv2.cvtColor(self.raw_capture.array, cv2.COLOR_BGR2GRAY) / 255.0
                self.raw_capture.truncate(0)
            except Exception as e:
                print(f"ERROR: Camera baseline failed: {e}")
        else:
            self.visual_grid = np.random.uniform(0, 0.15, self.resolution)
        self.last_calibration_time = time.time()

    def _read_arduino(self, side):
        if self.mock_mode or not self.sides[side].get('ser'):
            t = time.time()
            return {'resistance': 1e-9, 'capacitance': 0.0, 'permeability': 0.5 + np.sin(t * 0.1) * 0.5,
                    'magnetism': random.uniform(0, 0.2), 'permittivity': 1.0, 'dielectricity': 0.0, 'laser': random.uniform(0, 1)}
        ser = self.sides[side]['ser']
        ser.write(b'READ_SEXTET\n')
        data = ser.readline().decode().strip()
        if data and len(data.split(',')) == 7:
            values = list(map(float, data.split(',')))
            return dict(zip(['resistance', 'capacitance', 'permeability', 'magnetism', 'permittivity', 'dielectricity', 'laser'], values))
        return self.sextet[side]

    def _poll_sensors(self):
        while True:
            if time.time() - self.last_calibration_time > self.calibration_interval:
                self._capture_baselines()
            for side in ['A', 'B']:
                self.sextet[side].update(self._read_arduino(side))
            if self.camera:
                try:
                    self.camera.capture(self.raw_capture, format="bgr", use_video_port=True)
                    self.visual_grid = cv2.cvtColor(self.raw_capture.array, cv2.COLOR_BGR2GRAY) / 255.0
                    self.raw_capture.truncate(0)
                except Exception as e:
                    print(f"ERROR: Camera capture failed: {e}")
            else:
                x = np.linspace(-np.pi, np.pi, self.resolution[1])
                y = np.linspace(-np.pi, np.pi, self.resolution[0])
                xx, yy = np.meshgrid(x, y)
                t = time.time()
                self.visual_grid = 0.5 * (1 + np.sin(xx * 2 + t) * np.cos(yy * 2 + t))
            time.sleep(0.1)

    def get_sextet(self, side='A'):
        return self.sextet[side].copy()

    def get_visual_grid(self, paths=['A-B'], grid_size=1000):
        return self.visual_grid.copy()

    def set_laser(self, side, pulse):
        if not self.mock_mode and self.sides[side].get('ser'):
            self.sides[side]['ser'].write(f"L{pulse}".encode())
