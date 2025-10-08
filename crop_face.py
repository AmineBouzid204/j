from deepface import DeepFace
import os
from tqdm import tqdm
import cv2
import numpy as np

def crop(input_dir, output_dir, detector_backend="mtcnn"):
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_file)
        img_name = os.path.splitext(img_file)[0]
        
        try:
            # Extract faces
            face_objs = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                enforce_detection=True,
                grayscale=False
            )
            
            for i, face_obj in enumerate(face_objs):
                face = face_obj["face"]
                
                # Convert to uint8 if needed
                if face.dtype != np.uint8:
                    face = (face * 255).astype(np.uint8)
                
                # Resize and save
                face = cv2.resize(face, (224, 224))
                output_path = os.path.join(output_dir, f"{img_name}_{i}.jpg")
                cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue

crop("./data", "./cropped_faces")