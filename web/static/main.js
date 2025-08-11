// web/static/main.js (Upgraded with LED control function)

document.addEventListener('DOMContentLoaded', () => {
    // ... (all the code at the top is the same) ...
    const canvas = document.getElementById('ferrocell-canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const statsDisplay = document.getElementById('stats_display');
    const socket = io();
    
    socket.on('connect', () => { /* ... same ... */ });
    socket.on('simulation_update', (data) => { /* ... same ... */ });
    socket.on('disconnect', () => { /* ... same ... */ });

    // --- Control functions that send commands back to the server ---
    window.setPaths = function(paths) {
        console.log('Sending command to activate paths:', paths);
        socket.emit('set_active_paths', { paths: paths });
    };

    // --- NEW: Function to send LED control commands ---
    window.setLEDs = function(color, brightness) {
        console.log(`Sending command to set LEDs: ${color} at brightness ${brightness}`);
        socket.emit('set_led_state', { color: color, brightness: brightness });
    };
});
