// web/static/dashboard.js
import * as THREE from 'three';

// --- Scene Setup ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
camera.position.z = 100;

// --- Particle System ---
const PARTICLE_COUNT = 50000;
const DOWNSAMPLE_SIZE = 64; // Must match server
const positions = new Float32Array(PARTICLE_COUNT * 3);
const velocities = new Float32Array(PARTICLE_COUNT * 3);
let forceField = null;

for (let i = 0; i < PARTICLE_COUNT; i++) {
    positions[i * 3 + 0] = (Math.random() - 0.5) * 200; // x
    positions[i * 3 + 1] = (Math.random() - 0.5) * 200; // y
    positions[i * 3 + 2] = (Math.random() - 0.5) * 20;  // z
}
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const material = new THREE.PointsMaterial({ color: 0x00ff00, size: 0.5, blending: THREE.AdditiveBlending, transparent: true, opacity: 0.7 });
const particles = new THREE.Points(geometry, material);
scene.add(particles);

// --- Socket.IO Connection ---
const socket = io();
socket.on('force_field_update', (data) => {
    forceField = data.field;
});

// --- Animation Loop ---
function animate() {
    requestAnimationFrame(animate);

    if (forceField) {
        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const x = positions[i * 3 + 0];
            const y = positions[i * 3 + 1];

            // Map particle position to force field grid
            const gridX = Math.floor(((x / 200) + 0.5) * DOWNSAMPLE_SIZE);
            const gridY = Math.floor(((y / 200) + 0.5) * DOWNSAMPLE_SIZE);

            if (gridX >= 0 && gridX < DOWNSAMPLE_SIZE && gridY >= 0 && gridY < DOWNSAMPLE_SIZE) {
                const forceIndex = (gridY * DOWNSAMPLE_SIZE + gridX) * 2;
                const forceX = forceField[forceIndex];
                const forceY = forceField[forceIndex + 1];

                // Apply force to velocity
                velocities[i * 3 + 0] += forceX * 0.05;
                velocities[i * 3 + 1] += forceY * 0.05;
            }

            // Apply drag/friction
            velocities[i * 3 + 0] *= 0.95;
            velocities[i * 3 + 1] *= 0.95;
            
            // Update position with velocity
            positions[i * 3 + 0] += velocities[i * 3 + 0];
            positions[i * 3 + 1] += velocities[i * 3 + 1];

            // Boundary checks
            if (positions[i*3+0] > 100 || positions[i*3+0] < -100) velocities[i*3+0] *= -1;
            if (positions[i*3+1] > 100 || positions[i*3+1] < -100) velocities[i*3+1] *= -1;
        }
        geometry.attributes.position.needsUpdate = true;
    }

    renderer.render(scene, camera);
}
animate();
