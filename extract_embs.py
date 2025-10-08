from deepface import DeepFace
import os
import pickle
from tqdm import tqdm

def extract_embeddings(input_dir, output_path, model_name="Facenet512"):
    embeddings = {}
    
    for img_file in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_file)
        name = os.path.splitext(img_file)[0]
        
        try:
            # Get embedding
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend="skip",  # Skip detection since we already have cropped faces
                enforce_detection=False
            )
            
            embeddings[name] = embedding[0]["embedding"]
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    # Save embeddings
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    
    return embeddings

embs = extract_embeddings("./cropped_faces", "./embeddings/embs_facenet512.pkl")
print(f"Extracted embeddings for {len(embs)} faces")