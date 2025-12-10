import os
import uuid
import glob
import shutil
import logging
import random
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.neural_network import MLPClassifier
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")

# --- Model Initialization ---

# 1. YOLO Model
# Using yolov8n for speed. It will download automatically on first run.
yolo_model = YOLO('yolov8n.pt')

# 2. Feature Extractor (ResNet18)
# We remove the last fully connected layer to get embeddings.
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # Remove the last fc layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        return x.flatten(start_dim=1)

feature_extractor = FeatureExtractor()

# Preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# 3. Online Learnable Classifier (MLP + Replay Buffer)

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def add(self, embedding, label):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0) # Remove oldest
        self.buffer.append((embedding, label))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return self.buffer # Return all if not enough
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(capacity=2000)

# MLPClassifier supports online learning via partial_fit.
# Using a slightly larger network for stability.
# Note: warm_start=True is strictly for fit(), but partial_fit() handles incremental learning internally.
# Setting warm_start=True with partial_fit causes strict class checking issues if a batch misses a class.
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=False, random_state=42)

# Initialize the classifier with dummy data so it's ready to predict
# Classes: 0 = False Positive (Not Human), 1 = True Positive (Human)
dummy_X = np.zeros((2, 512)) # ResNet18 output dim is 512
dummy_y = np.array([0, 1])
clf.partial_fit(dummy_X, dummy_y, classes=[0, 1])
logger.info("Classifier initialized.")

# --- Helper Functions ---

def get_embedding(img_crop: Image.Image) -> np.ndarray:
    """Extract feature vector from an image crop."""
    img_tensor = transform(img_crop).unsqueeze(0) # Add batch dimension
    # Output of ResNet is float32, but sklearn MLPClassifier initialized with float64 expects float64
    embedding = feature_extractor(img_tensor).detach().numpy().astype(np.float64)
    return embedding

# --- API Endpoints ---

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Uploads an image, runs YOLO detection, and scores detections with the online classifier.
    """
    file_id = str(uuid.uuid4())
    # Sanitize filename: Just use UUID + original extension
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".jpg" # Default to jpg if no extension
    filename = f"{file_id}{ext}"
    filepath = os.path.join("app/uploads", filename)
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    # Read image for YOLO
    # YOLO accepts paths, PIL images, numpy arrays.
    # We'll use the path.
    results = yolo_model(filepath)
    
    # Process detections
    detections = []
    
    # Open image with PIL for cropping
    original_img = Image.open(filepath).convert("RGB")
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            # Check if it's a person (class 0 in COCO)
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Crop and extract features
                # Ensure coordinates are within bounds
                w, h = original_img.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = original_img.crop((x1, y1, x2, y2))
                    embedding = get_embedding(crop)
                    
                    # Predict using online classifier
                    # [0] is prob of class 0 (FP), [1] is prob of class 1 (TP/Human)
                    # We want probability that it IS a human.
                    human_prob = clf.predict_proba(embedding)[0][1]
                    
                    detections.append({
                        "id": str(uuid.uuid4()), # Unique ID for this detection
                        "bbox": [x1, y1, x2, y2],
                        "yolo_conf": conf,
                        "human_prob": float(human_prob)
                    })
    
    return JSONResponse({
        "filename": filename,
        "url": f"/uploads/{filename}",
        "detections": detections
    })

class FeedbackRequest(BaseModel):
    filename: str
    bbox: List[int]
    is_human: bool

@app.post("/feedback")
async def submit_feedback(data: FeedbackRequest):
    """
    Receives feedback for a detection and updates the online classifier.
    """
    filepath = os.path.join("app/uploads", data.filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
        
    original_img = Image.open(filepath).convert("RGB")
    x1, y1, x2, y2 = data.bbox
    
    # Validation
    w, h = original_img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
         raise HTTPException(status_code=400, detail="Invalid bbox")

    crop = original_img.crop((x1, y1, x2, y2))
    embedding = get_embedding(crop)
    
    # Label 1 if human, 0 if not
    y_label = 1 if data.is_human else 0
    
    # Add to replay buffer
    replay_buffer.add(embedding, y_label)
    
    # Sample a batch for training
    batch_size = 32
    samples = replay_buffer.sample(batch_size)
    
    # Prepare batch data
    # samples is a list of tuples (embedding, label)
    # embedding shape is (1, 512)
    
    X_batch = np.vstack([s[0] for s in samples])
    y_batch = np.array([s[1] for s in samples])
    
    # We must pass classes=[0, 1] to partial_fit every time to avoid "new classes" errors
    # or errors when a batch only contains one class.
    clf.partial_fit(X_batch, y_batch, classes=[0, 1])
    
    logger.info(f"Updated classifier with label {y_label}. Buffer size: {len(replay_buffer)}")
    
    return {"status": "success", "message": "Classifier updated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
