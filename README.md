title:Image captioning with face recognition and Smart image retrieval by semantic searching

Introduction:

This project was built as a pre-final year project during our 6th semester. The basic idea of our project is "smart" or "speed" searching that means enabling faster search using custom captions in our gallery

This is not yet a full fledged project like a complete app or cloud service, but we plan to develop it into one. In this project, we extract captions and metadata from images using machine learning and DL such as Blip, vit-gpt2, Deepface facenet, YOLO and K-Nearest Neighbors. The extracted data is then stored in a DB

For searching, we use sematic search powered by the all-MiniLM-L6-v2 model . This allows us to search both:
captions from images
images from captions
   
Models:
BLIP + ViT + GPT-2
Input: Image → Output: Generated Caption

DeepFace (FaceNet)
Input: Image → Output: Face Embedding

YOLO (Ultralytics)
Input: Image → Output: Detected Objects with Bounding Boxes

K-Nearest Neighbors (KNN)
Input: Feature Vector → Output: Predicted Class

KMeans (Optional)
Input: Feature Vectors → Output: Cluster Labels

SentenceTransformer (all-MiniLM-L6-v2)
Input: Text or Caption → Output: Semantic Embedding Vector

MTCNN + InceptionResnetV1 (FaceNet via facenet-pytorch)
Input: Image → Output: Aligned Face & 512-D Embedding


torch>=2.0.0
Core PyTorch library for deep learning and tensor computations.

transformers>=4.25.0
Hugging Face library for using pre-trained transformer models like GPT-2, BERT, etc.

sentence-transformers>=2.2.2
Enables semantic similarity and embedding generation for sentences using transformer models.

facenet-pytorch>=2.5.2
Face detection and recognition models (MTCNN + InceptionResnetV1) in PyTorch.

deepface>=0.0.79
Lightweight face recognition and facial attribute analysis framework.

ultralytics>=8.0.20
Official YOLOv5/YOLOv8 implementation for real-time object detection.

opencv-python>=4.7.0.72
OpenCV library for image and video processing tasks.

Pillow>=9.5.0
Python Imaging Library (PIL fork) for image manipulation and file I/O.

torchvision>=0.15.0
Image datasets, pre-trained models, and transforms for PyTorch.

matplotlib>=3.7.1
Visualization library for plotting graphs, images, and data.

scikit-learn>=1.2.2
Machine learning library for classification, clustering, and more.

numpy>=1.24.2
Core scientific computing library for numerical operations and arrays.

pymongo>=4.3.3
Python driver for MongoDB database interaction.

bson>=0.5.10
Binary JSON encoding/decoding used with MongoDB (for storing images, etc).




