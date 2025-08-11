// web/static/main.js

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('ferrocell-canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const statsDisplay = document.getElementById('stats_display');
    const socket = io();

    let lastFrameTime = 0;
    let frameCount = 0;
    let totalFrameTime = 0;
    
    // Set a fixed internal resolution for the canvas
    const resolution = 512;
    canvas.width = resolution;
    canvas.height = resolution;

    socket.on('connect', () => {
        console.log('Connected to Ferrocella server!');
        statsDisplay.textContent = 'Connected. Waiting for data...';
    });

    // This is the main event listener for real-time updates from the server
    socket.on('simulation_update', (data) => {
        // Performance stats
        const now = performance.now();
        if (lastFrameTime > 0) {
            const frameTime = now - lastFrameTime;
            totalFrameTime += frameTime;
            frameCount++;
            if (frameCount >= 30) { // Update stats every 30 frames
                const avgFrameTime = totalFrameTime / frameCount;
                const fps = 1000 / avgFrameTime;
                statsDisplay.textContent = `FPS: ${fps.toFixed(1)} | Frame Time: ${avgFrameTime.toFixed(2)}ms`;
                frameCount = 0;
                totalFrameTime = 0;
            }
        }
        lastFrameTime = now;

        // The server sends the raw grid data as a Base64 encoded string.
        // We decode it and render it to the canvas.
        const byteArray = new Uint8ClampedArray(atob(data.grid).split('').map(c => c.charCodeAt(0)));
        const imageData = new ImageData(byteArray, data.width, data.height);
        
        // Use createImageBitmap for faster rendering, then draw it
        createImageBitmap(imageData).then(bitmap => {
            ctx.drawImage(bitmap, 0, 0, resolution, resolution);
        });
    });

    socket.on('disconnect', () => {
        statsDisplay.textContent = 'Disconnected from server.';
    });

    // --- Control functions that send commands back to the server ---
    window.setPaths = function(paths) {
        console.log('Sending command to activate paths:', paths);
        socket.emit('set_active_paths', { paths: paths });
    };
});
