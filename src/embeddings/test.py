import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'crop'))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from crop_utils import crop_full_face_aligned
from embedder import get_embedding, compare_embeddings

BaseOptions = mp_python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(
        os.path.dirname(__file__), '..', 'landmarks', 'face_landmarker.task'
    )),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)
landmarker = FaceLandmarker.create_from_options(options)


def get_aligned_embedding(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read {image_path}")
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print(f"No face detected in {image_path}")
        return None

    landmarks = result.face_landmarks[0]
    aligned_crop = crop_full_face_aligned(frame, landmarks)
    embedding = get_embedding(aligned_crop)
    return embedding


emb1 = get_aligned_embedding('S_1.jpeg')
emb2 = get_aligned_embedding('d_2.jpeg')

if emb1 is not None and emb2 is not None:
    print('Embedding1 shape:', emb1.shape)
    print('Embedding2 shape:', emb2.shape)
    print('Similarity:', compare_embeddings(emb1, emb2))