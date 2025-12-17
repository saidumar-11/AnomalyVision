# prepare_embeddings.py
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

face_app = FaceAnalysis()
face_app.prepare(ctx_id=0)

data_folder = r"D:\AnomalyVision\mvp\database\train"
embeddings_db = {}

for person in os.listdir(data_folder):
    person_folder = os.path.join(data_folder, person)
    if not os.path.isdir(person_folder):
        continue
    embeddings = []
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        img = cv2.imread(img_path)
        faces = face_app.get(img)
        if len(faces) > 0:
            embeddings.append(faces[0].embedding)
    if embeddings:
        embeddings_db[person] = np.array(embeddings)

# Save database
np.savez("embeddings_db.npz", **embeddings_db)
print(f"Saved embeddings for {len(embeddings_db)} people")
