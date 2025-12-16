import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

frame_count = 0
SKIP_FRAMES = 5


THRESHOLD = 0.6

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load DB
with open("data/faces/employees.pkl", "rb") as f:
    db = pickle.load(f)

# Init model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))


rtsp_url = "rtsp://admin:488679102+@192.168.0.191/streaming/channels/101"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        cv2.imshow("AnomalyVision MVP", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    faces = app.get(frame)

    for face in faces[:1]:
        embedding = face.embedding

        best_score = 0
        name = "UNKNOWN"

        for person, db_emb in db.items():
            score = cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                name = person

        if best_score < THRESHOLD:
            name = "UNKNOWN"

        box = face.bbox.astype(int)
        cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("AnomalyVision MVP", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
