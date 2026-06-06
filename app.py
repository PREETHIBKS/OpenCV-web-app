import os
import cv2
import numpy as np
import time
import threading
from flask import Flask, render_template, Response, request, jsonify
from vision_pipeline import VisionPipeline

app = Flask(__name__)

# Initialize the modular computer vision processor
processor = VisionPipeline()

class WebcamStream:
    """Thread-safe background webcam frame Grabber."""
    def __init__(self, src=0):
        self.src = src
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.started = False
        self.read_lock = threading.Lock()
        self.active_clients = 0
        self.client_lock = threading.Lock()
        self.timer = None

    def start(self):
        with self.read_lock:
            if self.started:
                return
            
            # Close previous if exists
            if self.stream is not None:
                self.stream.release()
                
            self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
            
            # Optimize camera resolution for high FPS stream
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stream.set(cv2.CAP_PROP_FPS, 30)
            
            (self.grabbed, self.frame) = self.stream.read()
            self.started = True
            
        self.thread = threading.Thread(target=self.update, name="WebcamGrabber", daemon=True)
        self.thread.start()

    def update(self):
        while True:
            {
                # Check if we should terminate
            }
            with self.read_lock:
                if not self.started:
                    break
                if self.stream is not None:
                    (grabbed, frame) = self.stream.read()
                    if grabbed:
                        self.grabbed = grabbed
                        self.frame = frame
            time.sleep(0.015) # Cap loop to run approx 60 FPS max, saving CPU

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        with self.read_lock:
            self.started = False
            if self.stream is not None:
                self.stream.release()
                self.stream = None
            self.frame = None
            self.grabbed = False

    def add_client(self):
        with self.client_lock:
            self.active_clients += 1
            # Camera is always running, no need to start dynamically

    def remove_client(self):
        with self.client_lock:
            self.active_clients = max(0, self.active_clients - 1)
            # Keep camera active for instant reuse

# Global webcam manager instance (start instantly and keep active)
webcam = WebcamStream(src=0)
webcam.start()

@app.route('/')
def index():
    """Renders the main glassmorphic HTML dashboard."""
    return render_template('index.html')

def video_stream_generator(mode):
    """Feeds processed frames to Flask client under MJPEG standards."""
    webcam.add_client()
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0.0
    
    try:
        while True:
            frame = webcam.read()
            if frame is None:
                time.sleep(0.03)
                continue
            
            # FPS Calculation
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                
            # Run matching CV filters based on request parameter
            if mode == 'moving_object':
                processed_frame = processor.process_motion_detection(frame, fps)
            elif mode == 'face_detection':
                processed_frame = processor.process_face_detection(frame, fps)
            elif mode == 'color_tracking':
                processed_frame = processor.process_color_tracking(frame, fps)
            elif mode == 'face_recognition':
                processed_frame = processor.process_face_recognition(frame, fps)
            elif mode == 'emotion_detection':
                processed_frame = processor.process_emotion_detection(frame, fps)
            else:
                processed_frame = frame
                
            # Compress processed frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Enforce small sleep to allow other thread switches
            time.sleep(0.02)
            
    finally:
        webcam.remove_client()

@app.route('/video_feed/<mode>')
def video_feed(mode):
    """Streaming route. The source attribute of the HTML <img> tag binds here."""
    return Response(video_stream_generator(mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/register_face', methods=['POST'])
def api_register_face():
    """Captures and stores facial template matching under a provided identifier."""
    data = request.json or {}
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({'success': False, 'message': 'Invalid name provided.'}), 400
        
    frame = webcam.read()
    if frame is None:
        return jsonify({'success': False, 'message': 'Camera not streaming yet. Please try again.'}), 400
        
    success, message = processor.register_face(frame, name)
    return jsonify({'success': success, 'message': message})

@app.route('/api/set_color_bounds', methods=['POST'])
def api_set_color_bounds():
    """Dynamically updates HSV thresholds for the color tracking suite."""
    data = request.json or {}
    lower = data.get('lower', [])
    upper = data.get('upper', [])
    hex_color = data.get('hex', '#00d2ff')
    
    if len(lower) == 3 and len(upper) == 3:
        processor.hsv_lower = np.array(lower)
        processor.hsv_upper = np.array(upper)
        
        # Parse hex to BGR for OpenCV drawing overlays
        hex_clean = hex_color.lstrip('#')
        rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        processor.target_bgr = (rgb[2], rgb[1], rgb[0]) # BGR order
        
        return jsonify({'success': True, 'message': 'Color bounds updated successfully.'})
        
    return jsonify({'success': False, 'message': 'Invalid color bounds schema.'}), 400

@app.route('/api/set_motion_threshold', methods=['POST'])
def api_set_motion_threshold():
    """Dynamically updates the motion detection threshold."""
    data = request.json or {}
    val = data.get('threshold')
    if val is not None:
        processor.motion_threshold = int(val)
        return jsonify({'success': True, 'message': 'Motion threshold updated successfully.'})
    return jsonify({'success': False, 'message': 'Invalid threshold value.'}), 400

@app.route('/api/status')
def api_status():
    """Gets status coordinates for active camera feed."""
    return jsonify({
        'camera_active': webcam.started,
        'active_clients': webcam.active_clients,
        'registered_profiles': list(processor.registered_faces.keys())
    })

if __name__ == "__main__":

    # Render provides its own PORT
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )