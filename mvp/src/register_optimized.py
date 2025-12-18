import os
import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# -----------------------------
# Paths (adjusted to your setup)
# -----------------------------
data_folder = r"D:\AnomalyVision/mvp/database/train"
yolo_model_path = r"D:\AnomalyVision/mvp/yolov8n-face.pt"
embeddings_file = r"D:\AnomalyVision/mvp/database/embeddings_db.npz"

# -----------------------------
# Load YOLO model (face detection)
# -----------------------------
yolo_model = YOLO(yolo_model_path)

# -----------------------------
# Load InsightFace (embedding)
# -----------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)  # CPU

# -----------------------------
# Check dataset folder
# -----------------------------
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Dataset folder not found: {data_folder}")

# -----------------------------
# Process each person folder
# -----------------------------
embeddings_db = {}

for person in os.listdir(data_folder):
    person_folder = os.path.join(data_folder, person)
    if not os.path.isdir(person_folder):
        continue

    person_embeddings = []

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # YOLO detects face bounding boxes
        results = yolo_model.predict(img, imgsz=640)
        if len(results[0].boxes) == 0:
            continue  # skip images with no face

        # Take the first detected face
        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        # Add padding (25%)
        h, w, _ = img.shape
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        face_img = img[y1:y2, x1:x2]

        # Generate embedding
        faces = face_app.get(face_img)
        if len(faces) == 0:
            continue
        embedding = faces[0].embedding
        person_embeddings.append(embedding)

    if len(person_embeddings) > 0:
        embeddings_db[person] = np.vstack(person_embeddings)
        print(f"Processed {person}, {len(person_embeddings)} embeddings.")

# -----------------------------
# Save embeddings
# -----------------------------
np.savez(embeddings_file, **embeddings_db)
print(f"Saved embeddings to {embeddings_file}")

