// const express = require('express');
// const http = require('http');
// const socketIo = require('socket.io');
// const cors = require('cors');
// const fs = require('fs');
// const path = require('path');

// const app = express();
// const server = http.createServer(app);
// const io = socketIo(server, {
//   cors: {
//     origin: "*",
//     methods: ["GET", "POST"]
//   },
//   maxHttpBufferSize: 1e8 // 100MB max buffer size for video chunks
// });

// // Add frame counter to track streaming
// let frameCount = 0;

// // Middleware
// app.use(cors());
// app.use(express.json());
// app.use(express.static(path.join(__dirname, 'public')));

// // Routes
// app.get('/', (req, res) => {
//   res.send('Video streaming server is running');
// });

// // Add status endpoint to check streaming statistics
// app.get('/status', (req, res) => {
//   const stats = {
//     activeConnections: io.engine.clientsCount,
//     totalFramesReceived: frameCount,
//     serverUptime: process.uptime(),
//     timestamp: new Date().toISOString()
//   };
  
//   res.json(stats);
// });

// // Add monitoring page route
// app.get('/monitor', (req, res) => {
//   res.sendFile(path.join(__dirname, 'public', 'monitor.html'));
// });

// // Socket.io connection handling
// io.on('connection', (socket) => {
//   console.log('New client connected:', socket.id);
  
//   // Handle incoming video frames
//   socket.on('videoFrame', (frameData) => {
//     // Increment frame counter
//     frameCount++;
    
//     // Log every 30 frames (about once per second at 30fps)
//     if (frameCount % 30 === 0) {
//       console.log(`Total frames received: ${frameCount} - Latest size: ${frameData.byteLength} bytes`);
//     }
    
//     // Broadcast to all other connected clients
//     socket.broadcast.emit('videoStream', frameData);
//   });
  
//   // Handle disconnection
//   socket.on('disconnect', () => {
//     console.log('Client disconnected:', socket.id);
//   });
// });

// const PORT = process.env.PORT || 5000;
// server.listen(PORT, () => {
//   console.log(`Server running on port ${PORT}`);
// });


const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  },
  maxHttpBufferSize: 1e8 // 100MB max buffer size for video chunks
});

// Add frame counter to track streaming
let frameCount = 0;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Routes
app.get('/', (req, res) => {
  res.send('Video streaming server is running');
});

// Add endpoint to get the server's port
app.get('/server-info', (req, res) => {
  const port = server.address().port;
  res.json({ port });
});

// Add status endpoint to check streaming statistics
app.get('/status', (req, res) => {
  const stats = {
    activeConnections: io.engine.clientsCount,
    totalFramesReceived: frameCount,
    serverUptime: process.uptime(),
    timestamp: new Date().toISOString()
  };
  
  res.json(stats);
});

// Add monitoring page route
app.get('/monitor', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'monitor.html'));
});

// Socket.io connection handling
io.on('connection', (socket) => {
  console.log('New client connected:', socket.id);
  
  // Handle incoming video frames
  socket.on('videoFrame', (frameData) => {
    // Increment frame counter
    frameCount++;
    
    // Log every 30 frames (about once per second at 30fps)
    if (frameCount % 30 === 0) {
      console.log(`Total frames received: ${frameCount} - Latest size: ${frameData.byteLength} bytes`);
    }
    
    // Broadcast to all other connected clients
    socket.broadcast.emit('videoStream', frameData);
  });
  
  // Handle disconnection
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Use port 0 to let the OS assign an available port
const PORT = process.env.PORT || 0;
server.listen(PORT, () => {
  const actualPort = server.address().port;
  console.log(`Server running on port ${actualPort}`);
  console.log(`Server URL: http://localhost:${actualPort}`);
  console.log(`Monitor URL: http://localhost:${actualPort}/monitor`);
});