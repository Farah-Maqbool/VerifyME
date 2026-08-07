import cv2
import numpy as np
import os
from insightface.model_zoo import model_zoo

MODEL_PATH = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")

_recognition_model = model_zoo.get_model(MODEL_PATH)
_recognition_model.prepare(ctx_id=0)

EMBEDDING_INPUT_SIZE = 112

def get_embedding(crop):
    if crop is None or crop.size == 0:
        return None

    # crop should already be 112x112 if it came from crop_full_face_aligned()
    embedding = _recognition_model.get_feat([crop])[0]
    return embedding


def compare_embeddings(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None

    a = np.asarray(embedding1).flatten()
    b = np.asarray(embedding2).flatten()

    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return None

    similarity = float(np.dot(a, b) / denom)
    return similarity