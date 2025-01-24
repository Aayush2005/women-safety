from flask import Flask, Response, jsonify, redirect
import cv2
import model
import threading
import atexit
import time

app = Flask(__name__)

# Camera System Class
class CameraSystem:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.active = True
        atexit.register(self.cleanup)
        self._initialize_camera()

    def _initialize_camera(self):
        # Use video file instead of webcam
        self.cap = cv2.VideoCapture('test.mp4')
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video file")

    def get_frame(self):
        with self.lock:
            if self.cap and self.cap.isOpened():
                success, frame = self.cap.read()
                if success:
                    return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return False, None
            return False, None

    def cleanup(self):
        #Release resources on shutdown
        with self.lock:
            self.active = False
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            print("🛑 Camera resources released")

# Initialize Components
camera_system = CameraSystem()
latest_data = {
    "people_count": 0,
    "men": 0,
    "women": 0,
    "lone_women": 0
}


# Video Feed Generator
def generate_frames():
    while camera_system.active:
        success, frame = camera_system.get_frame()
        if not success:
            print("⚠️ Camera read failed, retrying...")
            time.sleep(0.1)
            continue

        try:
            # Process frame through model
            processed_frame, json_data = model.process_frame(frame)
            latest_data.update(json_data)

            # Convert back to BGR for JPEG encoding
            encoded_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', encoded_frame, [
                int(cv2.IMWRITE_JPEG_QUALITY), 
                85  # Quality balance between size and clarity
            ])
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"🚨 Frame processing error: {str(e)}")
            time.sleep(0.5)

# Flask Routes
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Safety Monitor</title>
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <style>
            body { margin: 0; padding: 20px; background: #1a1a1a; color: white; }
            #container { 
                position: relative; 
                width: 640px;
                margin: 20px auto;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 0 20px rgba(0,0,0,0.3);
            }
            #video-feed {
                display: block;
                width: 100%;
                height: auto;
            }
            #stats {
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(0,0,0,0.7);
                padding: 15px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                backdrop-filter: blur(5px);
            }
            .alert { color: #ff4444; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1 style="text-align: center; margin-bottom: 30px;">SAFETY MONITOR DASHBOARD</h1>
        <div id="container">
            <img id="video-feed" src="/video_feed">
            <div id="stats">
                <div>👥 People: <span id="people">0</span></div>
                <div>👨 Men: <span id="men">0</span></div>
                <div>👩 Women: <span id="women">0</span></div>
                <div class="alert">🚨 Lone Women: <span id="lone_women">0</span></div>
            </div>
        </div>

        <script>
            // Video feed management
            const videoElement = document.getElementById('video-feed');
            let connectionRetries = 0;

            function refreshVideo() {
                videoElement.src = '/video_feed?nocache=' + Date.now();
                connectionRetries = 0;
            }

            videoElement.onerror = () => {
                if(connectionRetries < 5) {
                    console.log(`Reconnecting... (Attempt ${connectionRetries + 1}/5)`);
                    setTimeout(refreshVideo, 1000);
                    connectionRetries++;
                } else {
                    console.error('Maximum connection attempts reached');
                    videoElement.style.backgroundColor = '#ff000020';
                }
            };

            // Data updates
            function updateAnalytics() {
                fetch('/get_data')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('people').textContent = data.people_count;
                        document.getElementById('men').textContent = data.men;
                        document.getElementById('women').textContent = data.women;
                        document.getElementById('lone_women').textContent = data.lone_women;
                        
                        if(data.lone_women > 0) {
                            document.querySelector('.alert').style.animation = 'blink 1s infinite';
                        }
                    })
                    .catch(error => console.error('Data fetch error:', error));
            }

            // Initial setup
            refreshVideo();
            setInterval(updateAnalytics, 500);
            updateAnalytics();

            // Visual alert animation
            const style = document.createElement('style');
            style.textContent = `
                @keyframes blink {
                    0% { opacity: 1; }
                    50% { opacity: 0.3; }
                    100% { opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/get_data')
def get_data():
    """JSON analytics endpoint"""
    return jsonify(latest_data)

# ======================
# Application Bootstrap
# ======================
if __name__ == "__main__":
    try:
        print("🚀 Starting safety monitoring system...")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    finally:
        camera_system.cleanup()
        print("✅ System shutdown complete")