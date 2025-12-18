import os
import uuid
import glob
import shutil
import logging
import random
import threading
import time
import queue
from typing import List, Optional, Dict

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

from transformers import CLIPProcessor, CLIPModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")

# --- Model Initialization ---

# 1. YOLO Model
yolo_model = YOLO('yolov8n.pt')

# 2. ResNet18 (Feature Extractor for MLP)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
        return x.flatten(start_dim=1)

feature_extractor = FeatureExtractor()

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# 3. CLIP Model (Feature Extractor for Filtering)
logger.info("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
logger.info("CLIP model loaded.")

# --- Storage Classes ---

# MLP Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def add(self, embedding, label):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((embedding, label))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return self.buffer
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(capacity=2000)
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=False, random_state=42)

# Initialize MLP with dummy data
dummy_X = np.zeros((2, 512))
dummy_y = np.array([0, 1])
clf.partial_fit(dummy_X, dummy_y, classes=[0, 1])
logger.info("MLP Classifier initialized.")

# CLIP Negative Store
class CLIPNegativeStore:
    def __init__(self):
        self.negatives = [] # List of {id, embedding, image_path}

    def add(self, embedding: torch.Tensor, image_path: str):
        # embedding shape: (1, 512)
        item = {
            "id": str(uuid.uuid4()),
            "embedding": embedding,
            "image_url": f"/uploads/{os.path.basename(image_path)}"
        }
        self.negatives.append(item)
        logger.info(f"Added negative sample to CLIP store. Count: {len(self.negatives)}")

    def delete(self, item_id: str):
        self.negatives = [item for item in self.negatives if item["id"] != item_id]

    def get_all(self):
        # Return serializable list
        return [{"id": x["id"], "image_url": x["image_url"]} for x in self.negatives]

    def is_similar(self, embedding: torch.Tensor, threshold=0.98) -> bool:
        if not self.negatives:
            return False

        # Stack stored embeddings
        stored_embs = torch.cat([x["embedding"] for x in self.negatives], dim=0) # (N, 512)

        # Normalize
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        stored_embs = stored_embs / stored_embs.norm(dim=-1, keepdim=True)

        # Cosine similarity
        # embedding: (1, 512)
        sims = (embedding @ stored_embs.T).squeeze(0) # (N,)

        if len(sims.shape) == 0: # Case if only 1 negative
             max_sim = sims.item()
        else:
             max_sim = sims.max().item()

        logger.info(f"Max CLIP similarity: {max_sim}")
        return max_sim > threshold

clip_store = CLIPNegativeStore()

# --- RTSP Monitor ---

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

class RtspMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
        self.url = ""
        self.events = [] # List of pending events {id, image_path, image_url, bbox}
        self.lock = threading.Lock()

        # Cool-down tracking
        # List of {"bbox": [x1,y1,x2,y2], "timestamp": t}
        self.recent_detections = []

    def start(self, url):
        if self.running:
            self.stop()
        self.url = url
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        logger.info(f"RTSP Monitor started for {url}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread = None
        logger.info("RTSP Monitor stopped")

    def loop(self):
        cap = cv2.VideoCapture(self.url)

        # Motion Detection Setup
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        last_frame = None

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            # 1. Motion Detection Check (Optimization)
            # Use small resize for speed
            small_frame = cv2.resize(frame, (640, 480))
            fgmask = fgbg.apply(small_frame)
            motion_ratio = np.count_nonzero(fgmask) / (small_frame.shape[0] * small_frame.shape[1])

            # If motion is very low, skip YOLO
            if motion_ratio < 0.01:
                time.sleep(0.1)
                continue

            # 2. Cleanup old cool-down records
            now = time.time()
            self.recent_detections = [d for d in self.recent_detections if now - d["timestamp"] < 5.0]

            # 3. Run YOLO
            results = yolo_model(frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0: # Person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        current_box = [x1, y1, x2, y2]

                        # Check cool-down
                        is_cooled_down = False
                        for prev in self.recent_detections:
                            if compute_iou(current_box, prev["bbox"]) > 0.3: # Threshold 0.3 IoU
                                is_cooled_down = True
                                break

                        if is_cooled_down:
                            continue

                        # If passed cool-down, process it
                        h, w, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]

                            # Save crop
                            event_id = str(uuid.uuid4())
                            filename = f"rtsp_{event_id}.jpg"
                            filepath = os.path.join("app/uploads", filename)
                            cv2.imwrite(filepath, crop)

                            with self.lock:
                                self.events.append({
                                    "id": event_id,
                                    "image_path": filepath,
                                    "image_url": f"/uploads/{filename}",
                                    "timestamp": now
                                })
                                if len(self.events) > 50:
                                    self.events.pop(0)

                            # Add to recent detections
                            self.recent_detections.append({
                                "bbox": current_box,
                                "timestamp": now
                            })

            time.sleep(0.5)
        cap.release()

    def get_events(self):
        with self.lock:
            return list(self.events)

    def remove_event(self, event_id):
        with self.lock:
            self.events = [e for e in self.events if e["id"] != event_id]

rtsp_monitor = RtspMonitor()

# --- Helper Functions ---

def get_resnet_embedding(img_crop: Image.Image) -> np.ndarray:
    img_tensor = resnet_transform(img_crop).unsqueeze(0)
    embedding = feature_extractor(img_tensor).detach().numpy().astype(np.float64)
    return embedding

def get_clip_embedding(img_crop: Image.Image) -> torch.Tensor:
    # CLIPProcessor takes images (PIL or numpy)
    inputs = clip_processor(images=img_crop, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    # Return (1, 512) tensor
    return outputs

# --- API Endpoints ---

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), mode: str = Form("mlp")):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{file_id}{ext}"
    filepath = os.path.join("app/uploads", filename)
    
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    results = yolo_model(filepath)
    detections = []
    original_img = Image.open(filepath).convert("RGB")
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls[0]) == 0: # Person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                w, h = original_img.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = original_img.crop((x1, y1, x2, y2))
                    
                    human_prob = 1.0 # Default if passed filtering
                    status = "accepted"

                    if mode == "clip":
                        # Extract CLIP feature
                        clip_emb = get_clip_embedding(crop)
                        # Filter
                        if clip_store.is_similar(clip_emb, threshold=0.98):
                            status = "filtered_clip"
                        else:
                            human_prob = 1.0

                    else:
                        # MLP Mode
                        embedding = get_resnet_embedding(crop)
                        human_prob = clf.predict_proba(embedding)[0][1]
                        status = "processed_mlp"
                    
                    detections.append({
                        "id": str(uuid.uuid4()),
                        "bbox": [x1, y1, x2, y2],
                        "yolo_conf": conf,
                        "human_prob": float(human_prob),
                        "status": status
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
    mode: str = "mlp"

@app.post("/feedback")
async def submit_feedback(data: FeedbackRequest):
    filepath = os.path.join("app/uploads", data.filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
        
    original_img = Image.open(filepath).convert("RGB")
    x1, y1, x2, y2 = data.bbox
    crop = original_img.crop((x1, y1, x2, y2))
    
    if data.mode == "clip":
        if not data.is_human:
            # User says "Not Human" -> Add to CLIP Negative Store
            clip_emb = get_clip_embedding(crop)
            # We save the crop as a separate file to display in history
            crop_id = str(uuid.uuid4())
            crop_filename = f"neg_{crop_id}.jpg"
            crop_path = os.path.join("app/uploads", crop_filename)
            crop.save(crop_path)

            clip_store.add(clip_emb, crop_path)
            return {"status": "success", "message": "Added to CLIP negatives"}
        else:
            # User says "Human". If it was previously filtered, it wouldn't be here.
            # If it passed filter and user confirms, we don't need to do anything for CLIP filtering.
            # (Unless we wanted a Positive store, but requirement only mentions filtering out)
            return {"status": "success", "message": "Feedback ignored for CLIP positive (only negatives stored)"}
    
    else:
        # MLP Mode
        embedding = get_resnet_embedding(crop)
        y_label = 1 if data.is_human else 0
        replay_buffer.add(embedding, y_label)

        batch_size = 32
        samples = replay_buffer.sample(batch_size)
        X_batch = np.vstack([s[0] for s in samples])
        y_batch = np.array([s[1] for s in samples])
        clf.partial_fit(X_batch, y_batch, classes=[0, 1])

        return {"status": "success", "message": "Classifier updated"}

# --- History Endpoints ---

@app.get("/history")
async def get_history():
    return clip_store.get_all()

@app.delete("/history/{item_id}")
async def delete_history_item(item_id: str):
    clip_store.delete(item_id)
    return {"status": "success"}

# --- RTSP Endpoints ---

class RtspUrl(BaseModel):
    url: str

@app.post("/rtsp/start")
async def start_rtsp(data: RtspUrl):
    rtsp_monitor.start(data.url)
    return {"status": "started"}

@app.post("/rtsp/stop")
async def stop_rtsp():
    rtsp_monitor.stop()
    return {"status": "stopped"}

@app.get("/rtsp/events")
async def get_rtsp_events():
    return rtsp_monitor.get_events()

class RtspLabel(BaseModel):
    event_id: str
    is_human: bool

@app.post("/rtsp/label")
async def label_rtsp_event(data: RtspLabel):
    events = rtsp_monitor.get_events()
    target = next((e for e in events if e["id"] == data.event_id), None)
    
    if target:
        # If labeled, we treat it as feedback logic
        # If "Not Human", add to CLIP store (since we want to filter these out in future monitoring?)
        # The prompt says: "Real-time monitoring... wait for user labeling... maintain a learning history... in clip mode allow delete".
        # This implies RTSP labeling should also feed into the system.

        # Load the image from disk
        crop_path = target["image_path"]
        if os.path.exists(crop_path):
            img = Image.open(crop_path).convert("RGB")

            # If Not Human, add to CLIP store so we don't show similar things again?
            # Or just add to MLP?
            # Prompt implies CLIP mode is the primary enhancement.
            # Let's assume RTSP labeling feeds the CLIP negative store if "Not Human".

            if not data.is_human:
                clip_emb = get_clip_embedding(img)
                clip_store.add(clip_emb, crop_path)

            # Remove from pending list
            rtsp_monitor.remove_event(data.event_id)
            return {"status": "success"}
    
    return {"status": "error", "message": "Event not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
