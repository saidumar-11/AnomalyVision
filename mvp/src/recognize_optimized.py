import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# -----------------------------
# Paths
# -----------------------------
yolo_model_path = r"D:\AnomalyVision/mvp/yolov8n-face.pt"
embeddings_file = r"D:\AnomalyVision/mvp/database/embeddings_db.npz"

# -----------------------------
# Load YOLO and InsightFace
# -----------------------------
yolo_model = YOLO(yolo_model_path)
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)  # CPU

# -----------------------------
# Load embeddings
# -----------------------------
data = np.load(embeddings_file, allow_pickle=True)
embeddings_db = {k: data[k] for k in data.keys()}

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(frame, imgsz=640)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)

        # Add padding for better InsightFace detection
        h, w, _ = frame.shape
        pad_x = int((x2 - x1) * 0.25)  # 25% padding (increased from 10%)
        pad_y = int((y2 - y1) * 0.25)
        
        x1_p = max(0, x1 - pad_x)
        y1_p = max(0, y1 - pad_y)
        x2_p = min(w, x2 + pad_x)
        y2_p = min(h, y2 + pad_y)

        face_img = frame[y1_p:y2_p, x1_p:x2_p]

        # -----------------------------
        # Draw bounding box (YOLO) - original box
        # -----------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        faces = face_app.get(face_img)
        if len(faces) == 0:
            cv2.putText(frame, "No Face (Insight)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue
        embedding = faces[0].embedding

        # -----------------------------
        # Compare embedding with database
        # -----------------------------
        min_dist = float("inf")
        identity = "Unknown"

        for person, emb_array in embeddings_db.items():
            for db_emb in emb_array:
                dist = cosine(embedding, db_emb)
                if dist < min_dist:
                    min_dist = dist
                    identity = person

        # -----------------------------
        # Draw Label
        # -----------------------------
        label_color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
        cv2.putText(frame, f"{identity} ({min_dist:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
