import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

THRESHOLD = 0.6

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load DB
with open("data/faces/employees.pkl", "rb") as f:
    db = pickle.load(f)

# Init model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        embedding = face.embedding
        name = "UNKNOWN"
        best_score = 0

        for person, db_emb in db.items():
            score = cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                name = person

        if best_score < THRESHOLD:
            name = "UNKNOWN"

        box = face.bbox.astype(int)
        cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 2)
        cv2.putText(frame, name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("AnomalyVision MVP", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
