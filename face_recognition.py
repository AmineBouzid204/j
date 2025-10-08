from deepface import DeepFace
from deepface.modules.verification import find_distance
import cv2
import time
import sys
import pickle
import cvzone
import os

# Use webcam instead of video file
cap = cv2.VideoCapture(0)  # 0 is the default camera index

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Read the first frame to get dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read from webcam.")
    sys.exit(1)

frame_width = frame.shape[1]
frame_height = frame.shape[0]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))

start_time = time.time()
frame_count = 0
detected_faces = []
model_name = "Facenet512"
metrics = [{"cosine": 0.30}, {"euclidean": 20.0}, {"euclidean_l2": 0.78}]

# Load embeddings
emb_path = "./embeddings/embs_facenet512.pkl"
if not os.path.exists(emb_path):
    print("No existing embeddings file found. Check out your path.")
    sys.exit(1)

with open(emb_path, "rb") as file:
    embs = pickle.load(file)
    print("Existing embeddings file loaded successfully.")

# FPS helper
def calculate_fps(start_time):
    current_time = time.time()
    fps = 1.0 / (current_time - start_time)
    return fps, current_time

# CLAHE contrast enhancement
def clahe(image):
    clahe_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_filter.apply(image)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from webcam.")
        break

    fps, start_time = calculate_fps(start_time)

    if frame_count % 5 == 0:
        detected_faces = []
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
                except:
                    continue  # Skip faces that can't be processed

                emb = DeepFace.represent(
                    prepared,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )[0]["embedding"]

                min_dist = float("inf")
                match_name = None
                for name, emb2 in embs.items():
                    dst = find_distance(emb, emb2, list(metrics[2].keys())[0])
                    if dst < min_dist:
                        min_dist = dst
                        match_name = name

                threshold = list(metrics[2].values())[0]
                color = (0, 255, 0) if min_dist < threshold else (0, 0, 255)
                label = match_name if min_dist < threshold else "Inconnu"
                detected_faces.append((x1, y1, x2, y2, label, min_dist, color))
                print(f"Detected: {label} | Distance: {min_dist:.2f}")

    for x1, y1, x2, y2, name, min_dist, color in detected_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cvzone.putTextRect(
            frame,
            f"{name} {min_dist:.2f}",
            (x1 + 10, y1 - 12),
            scale=1.5,
            thickness=2,
            colorR=color,
        )

    out.write(frame)
    cv2.imshow("Frame", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
