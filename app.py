from flask import Flask, render_template_string, Response
from deepface import DeepFace
from deepface.modules.verification import find_distance
import cv2
import pickle
import cvzone
import os
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables with thread locks
detected_faces = []
model_name = "Facenet512"
metrics = [{"cosine": 0.30}, {"euclidean": 20.0}, {"euclidean_l2": 0.78}]
embs = {}
frame_count = 0
face_lock = threading.Lock()
emb_lock = threading.Lock()

# Load embeddings on startup
def load_embeddings():
    global embs
    emb_path = "./embeddings/embs_facenet512.pkl"
    try:
        if os.path.exists(emb_path):
            with emb_lock:
                with open(emb_path, "rb") as file:
                    embs = pickle.load(file)
                    logger.info(f"Loaded embeddings for {len(embs)} faces")
        else:
            logger.warning("No embeddings file found. Face recognition will show 'Unknown' for all faces.")
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")

# CLAHE contrast enhancement
def clahe(image):
    try:
        clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe_filter.apply(image)
    except Exception as e:
        logger.error(f"CLAHE processing error: {e}")
        return image

def generate_frames():
    global detected_faces, frame_count
    
    # Initialize camera with reduced resolution
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame, reinitializing camera...")
                cap.release()
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.error("Could not reinitialize webcam")
                    break
                continue
            
            # Process every 5th frame for face detection
            if frame_count % 5 == 0:
                temp_faces = []
                try:
                    results = DeepFace.extract_faces(
                        frame, detector_backend="yolov8", enforce_detection=False
                    )
                    
                    for result in results:
                        if result["confidence"] >= 0.5:
                            area = result["facial_area"]
                            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                            x1, y1, x2, y2 = x, y, x + w, y + h
                            
                            cropped_face = frame[y:y + h, x:x + w]
                            try:
                                resized = cv2.resize(cropped_face, (224, 224))
                                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                                enhanced = clahe(gray)
                                prepared = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                                
                                if embs:  # Only do recognition if we have embeddings
                                    emb = DeepFace.represent(
                                        prepared,
                                        model_name=model_name,
                                        enforce_detection=False,
                                        detector_backend="skip",
                                    )[0]["embedding"]
                                    
                                    min_dist = float("inf")
                                    match_name = None
                                    
                                    with emb_lock:  # Thread-safe access to embeddings
                                        for name, emb2 in embs.items():
                                            dst = find_distance(emb, emb2, list(metrics[2].keys())[0])
                                            if dst < min_dist:
                                                min_dist = dst
                                                match_name = name
                                    
                                    threshold = list(metrics[2].values())[0]
                                    color = (0, 255, 0) if min_dist < threshold else (0, 0, 255)
                                    label = match_name if min_dist < threshold else "Unknown"
                                else:
                                    color = (0, 0, 255)
                                    label = "Unknown"
                                    min_dist = 0.0
                                
                                temp_faces.append((x1, y1, x2, y2, label, min_dist, color))
                            except Exception as e:
                                logger.error(f"Face processing error: {e}")
                                continue
                    
                    with face_lock:
                        detected_faces = temp_faces
                except Exception as e:
                    logger.error(f"Face detection error: {e}")
            
            # Draw rectangles and labels for detected faces
            with face_lock:
                for x1, y1, x2, y2, name, min_dist, color in detected_faces:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cvzone.putTextRect(
                        frame,
                        f"{name} {min_dist:.2f}",
                        (x1 + 10, y1 - 12),
                        scale=1,
                        thickness=1,
                        colorR=color,
                    )
            
            frame_count += 1
            
            # Encode frame to JPEG with lower quality for better streaming
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                logger.error("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    except KeyboardInterrupt:
        logger.info("Shutting down video feed")
    finally:
        cap.release()
        logger.info("Camera released")

@app.route('/')
def index():
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Recognition Camera</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                background-color: #000000;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                font-family: Arial, sans-serif;
            }
            
            .container {
                text-align: center;
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
            }
            
            .camera-feed {
                border: 2px solid #333;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
                max-width: 100%;
                height: auto;
            }
            
            .title {
                color: #ffffff;
                margin-bottom: 20px;
                font-size: 24px;
                font-weight: bold;
            }
            
            .info {
                color: #cccccc;
                margin-top: 15px;
                font-size: 14px;
            }
            
            .loading {
                color: #ffffff;
                margin: 20px;
                font-size: 18px;
            }
        </style>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const videoFeed = document.querySelector('.camera-feed');
                const loadingMsg = document.createElement('div');
                loadingMsg.className = 'loading';
                loadingMsg.textContent = 'Loading video feed...';
                videoFeed.parentNode.insertBefore(loadingMsg, videoFeed);
                
                videoFeed.onload = function() {
                    loadingMsg.style.display = 'none';
                };
                
                videoFeed.onerror = function() {
                    loadingMsg.textContent = 'Failed to load video feed. Please refresh.';
                };
            });
        </script>
    </head>
    <body>
        <div class="container">
            <div class="title">Face Recognition Camera</div>
            <img src="{{ url_for('video_feed') }}" class="camera-feed" alt="Camera Feed">
            <div class="info">
                Green box: Recognized face | Red box: Unknown face
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    })

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    # Load embeddings on startup
    load_embeddings()
    
    logger.info("Starting Flask Face Recognition App...")
    logger.info("Open your browser and go to: http://192.168.1.3:5000")
    logger.info("Press Ctrl+C to stop the application")
    
    # Use Waitress production server with explicit IP binding
    from waitress import serve
    try:
        serve(app, host='192.168.1.3', port=5000, threads=4)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")