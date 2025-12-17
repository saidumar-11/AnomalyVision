import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# --- Load YOLOv8n (optimized for speed) ---
yolo_model = YOLO("runs/detect/face_detection/weights/best.pt")

# --- Load InsightFace ---
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0)

# --- Load embeddings database ---
db_file = np.load("embeddings_db.npz", allow_pickle=True)
embeddings_db = {key: db_file[key] for key in db_file.files}

# --- Recognition function ---
def recognize_face(embedding, embeddings_db, threshold=0.5):
    """
    embedding: np.array of detected face
    embeddings_db: dict {person_name: [embeddings]}
    threshold: cosine similarity threshold
    """
    best_person = "Unknown"
    best_score = -1
    for person, emb_list in embeddings_db.items():
        sims = cosine_similarity([embedding], emb_list)
        sim = sims.max()
        if sim > best_score and sim > threshold:
            best_score = sim
            best_person = person
    return best_person

# --- Real-time webcam ---
cap = cv2.VideoCapture(0)
imgsz = 320  # small size for speed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo_model.predict(frame, imgsz=imgsz, conf=0.5)
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        face_crop = frame[y1:y2, x1:x2]

        # InsightFace embedding
        faces = face_app.get(face_crop)
        if len(faces) == 0:
            continue
        embedding = faces[0].embedding

        # Recognize person
        label = recognize_face(embedding, embeddings_db, threshold=0.5)

        # Draw results
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
