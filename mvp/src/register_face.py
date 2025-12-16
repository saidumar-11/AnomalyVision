import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import os

# Init model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

DB_PATH = "data/faces/employees.pkl"

# Load or create DB
db = {}
if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)


name = input("Enter person name: ")

cap = cv2.VideoCapture(0)
print("Press SPACE to capture face")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 2)

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1)
    if key == 32 and faces:  # SPACE
        db[name] = faces[0].embedding
        break

cap.release()
cv2.destroyAllWindows()

with open(DB_PATH, "wb") as f:
    pickle.dump(db, f)

print(f"Face for {name} saved.")
