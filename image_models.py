from transformers import  AutoTokenizer,VisionEncoderDecoderModel,ViTImageProcessor
from PIL import Image
import torch
from deepface import DeepFace
import json
import cv2
from ultralytics import YOLO
import torchvision
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from PIL import Image
import matplotlib.pyplot as plt
import os

#cap
model_name="nlpconnect/vit-gpt2-image-captioning"
model1=VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extracter=ViTImageProcessor.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)


#obj
model2 = YOLO("yolov5s.pt")


#rbg
model3 = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
"""Run DeepLabV3 to get person mask"""
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
             ])
#knn
knn = KNeighborsClassifier(n_neighbors=3)


def load_image(img_path):
    return Image.open(img_path).convert("RGB")

def captioning_model(image):
    pixel_values=feature_extracter(images=image, return_tensors="pt").pixel_values
    #print("pv",pixel_values)
    with torch.no_grad():
     output_ids = model1.generate(pixel_values, max_length=30, num_beams=5,repetition_penalty=2.0,early_stopping=True)
     #print("oid",output_ids)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #print("oid1",output_ids[0])
    return caption

def analyze_face(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Image not found at {image_path}")
            return None

        result = DeepFace.analyze(
            img_path=image,
            actions=["emotion", "age", "gender"],
            enforce_detection=True,
            detector_backend="opencv"
        )

        return {
            "age": result[0]['age'],
            "gender": max(result[0]['gender'], key=result[0]['gender'].get),
            "emotion": result[0]['dominant_emotion']
        }

    except ValueError as e:
        if "Face could not be detected" in str(e):
            print(f"No face detected in {image_path}")
        else:
            print(f"DeepFace error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def objects_with_count(image_path):
   results = model2(image_path)
   result = results[0]
   objects={}

   for box in result.boxes:
      cls_id = int(box.cls[0])                           # Class ID
      label = model2.names[cls_id]                        # Class label (like 'person', 'car', etc.)
      conf = float(box.conf[0])                          # Confidence score
      xyxy = box.xyxy[0].tolist()
      if label in objects:
        objects.update({label:objects[label]+1})                      
      else:
        objects[label]=1

   return objects

def get_person_mask(image):
    
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model3(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

def extract_colors(image, mask, n_colors=3):
    """Extract dominant colors from clothes region"""
    person_pixels = image[mask == 15]  # Class 15 is 'person' in COCO
    if len(person_pixels) == 0:
        return None

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(person_pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def plot_colors(colors):
    """Plot color boxes"""
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(colors):
        plt.subplot(1, len(colors), i+1)
        plt.axis("off")
        plt.imshow([[color / 255]])
    plt.show()


def euclidean_distance(color1, color2):
    # Compute the Euclidean distance between two RGB colors
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

def get_color_name(rgb):
    COLOR_NAMES = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'brown': (165, 42, 42),
    'pink': (255, 192, 203),
    'gray': (169, 169, 169),
    'black': (0, 0, 0),
    'white': (255, 255, 255)
        }
    min_distance = float('inf')
    closest_name = None
    for name, color in COLOR_NAMES.items():
        distance = euclidean_distance(rgb, color)
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name



def color(image_path):
  cols=[]
  image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
  mask = get_person_mask(image)
  colors = extract_colors(image, mask)
  #plot_colors(colors)
  for col in colors:
      cols.append(get_color_name(col))
  return cols


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
    
def load_image(img_path):
    return Image.open(img_path).convert("RGB")

