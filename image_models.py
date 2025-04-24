from typing import List, Dict, Set
import re
from sentence_transformers import SentenceTransformer
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from bson.binary import Binary
from datetime import datetime
from pymongo import MongoClient
import os
import numpy as np
from deepface import DeepFace
import cv2

# In search_model.py
THRESHOLD = 0.6  # Typical face recognition threshold
# Initialize MongoDB client
client = MongoClient("mongodb://localhost:27017/")
db = client["image_gallery"]
collection = db["images"]
verified_faces = db["verified"]
unverified_faces = db["unverified"]
db2= client["face_gallery"]
faces=db2['faces']

# Initialize models
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device, select_largest=False, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def process_sets(set1: List[Dict], set2: List[List[str]]) -> tuple:
    """
    Extract image paths from both sets and return them as separate lists.
    
    Args:
        set1: List of dictionaries containing image information (including 'image_path')
        set2: List of lists containing image paths
        
    Returns:
        tuple: (list_path1, list_path2) containing paths from set1 and flattened paths from set2
    """
    # Extract paths from set1
    list_path1 = [item['image_path'] for item in set1]
    
    # Flatten set2 and remove empty lists
    list_path2 = [path for sublist in set2 for path in sublist if sublist]
    
    return list_path1, list_path2

def wide_search(paths1: List[str]) -> Set[str]:
    return set(paths1)

def narrow_search(paths1: List[str], paths2: List[str]) -> Set[str]:
    return set(paths1).intersection(set(paths2))

def search_faces(name_query):
    pattern = re.compile(f".*{re.escape(name_query)}.*", re.IGNORECASE)
    matching_names = [
        face["name"] 
        for face in faces.find({}, {"name": 1})
        if pattern.match(face["name"])
    ]
    
    image_paths = []
    for name in matching_names:
        records = verified_faces.find({"name": name})
        for record in records:
            if "sample_image" in record and os.path.exists(record["sample_image"]):
                image_paths.append(record["sample_image"])
            if "related_images" in record:
                image_paths.extend([
                    img for img in record["related_images"] 
                    if os.path.exists(img)
                ])
    return list(set(image_paths))

def image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def search_and_show_images(query, top_k=5):
    try:
        query_embedding = embedder.encode(query, convert_to_tensor=False)
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
    except Exception as e:
        print(f"Error generating query embedding: {str(e)}")
        return []
    
    all_images = list(collection.find({}))
    results = []
    
    for img in all_images:
        try:
            if 'embedding' not in img:
                metadata_text = (
                    f"Caption: {img['metadata']['caption']}. "
                    f"Objects: {img['metadata']['objects_count']}. "
                    f"Colors: {img['metadata']['objects_colors']}. "
                    f"Actions: {img['metadata']['actions']}"
                )
                img_embedding = embedder.encode(metadata_text, convert_to_tensor=False)
                collection.update_one(
                    {'_id': img['_id']},
                    {'$set': {'embedding': img_embedding.tolist()}}
                )
            else:
                img_embedding = np.array(img['embedding'])
            
            query_embedding = np.array(query_embedding).reshape(1, -1)
            img_embedding = np.array(img_embedding).reshape(1, -1)
            sim_score = cosine_similarity(query_embedding, img_embedding)[0][0]
            
            results.append({
                'score': sim_score,
                'image_path': img['image_path'],
                'metadata': img['metadata']
            })
        except Exception as e:
            print(f"Error processing image {img.get('image_path', 'unknown')}: {str(e)}")
            continue
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def preprocess_image(img_path):
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading {img_path}: {str(e)}")
        return None

def get_face_embedding(img_path):
    img = preprocess_image(img_path)
    if img is None:
        return None
    
    try:
        faces = mtcnn(img)
        if faces is None:
            print(f"No faces detected in {os.path.basename(img_path)}")
            return None
            
        if faces.dim() == 4:
            faces = [faces[i] for i in range(faces.shape[0])]
        else:
            faces = [faces]
        
        largest_face = max(faces, key=lambda f: f.shape[1]*f.shape[2])
        largest_face = largest_face.unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = resnet(largest_face)[0].cpu().numpy()
        
        return embedding
    except Exception as e:
        print(f"Error processing {os.path.basename(img_path)}: {str(e)}")
        return None

def find_similar_faces(embedding, threshold=0.7):
    matches = []
    
    for face in verified_faces.find({}):
        try:
            stored_embedding = pickle.loads(face["embedding"])
            similarity = 1 - np.linalg.norm(embedding - stored_embedding)
            if similarity > threshold:
                matches.append({
                    "type": "verified",
                    "id": face["_id"],
                    "name": face["name"],
                    "similarity": similarity,
                    "image_path": face["sample_image"]
                })
        except Exception as e:
            print(f"Error loading face from DB: {str(e)}")
            continue
    
    for face in unverified_faces.find({}):
        try:
            stored_embedding = pickle.loads(face["embedding"])
            similarity = 1 - np.linalg.norm(embedding - stored_embedding)
            if similarity > threshold:
                matches.append({
                    "type": "unverified",
                    "id": face["_id"],
                    "similarity": similarity,
                    "image_path": face["image_path"]
                })
        except Exception as e:
            print(f"Error loading unverified face from DB: {str(e)}")
            continue
    
    return sorted(matches, key=lambda x: x["similarity"], reverse=True)

def register_new_face(img_path, embedding, matches, name=None):
    if not name:
        return False
    
    verified_faces.insert_one({
        "name": name,
        "embedding": Binary(pickle.dumps(embedding)),
        "sample_image": img_path,
        "related_images": [img_path] + [m["image_path"] for m in matches],
        "registration_date": datetime.now(),
        "count": len(matches) + 1
    })
    
    for match in matches:
        if match["type"] == "unverified":
            verified_faces.update_one(
                {"name": name},
                {"$push": {"related_images": match["image_path"]}}
            )
            unverified_faces.delete_one({"_id": match["id"]})
    
    return True

def analyze_face(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Image not found at {image_path}")
            return None

        # Analyze all faces in the image
        results = DeepFace.analyze(
            img_path=image,
            actions=["emotion", "age", "gender"],
            enforce_detection=True,
            detector_backend="opencv"
        )

        # Handle single or multiple faces (results can be a dict or a list)
        if not isinstance(results, list):
            results = [results]

        analysis_list = []
        for i, face_data in enumerate(results):
            analysis = {
                "face_index": i,
                "age": face_data['age'],
                "gender": max(face_data['gender'], key=face_data['gender'].get),
                "emotion": face_data['dominant_emotion']
            }
            analysis_list.append(analysis)

        return analysis_list

    except ValueError as e:
        if "Face could not be detected" in str(e):
            print(f"No face detected in {image_path}")
        else:
            print(f"DeepFace error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    


