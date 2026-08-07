import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'crop'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from crop_utils import crop_full_face_aligned, crop_periocular_aligned
from embedder import get_embedding
from database import enroll_employee

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

frame = cv2.imread("../embeddings/S_1.jpeg")  # your enrollment photo
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
result = landmarker.detect(mp_image)

landmarks = result.face_landmarks[0]

full_crop = crop_full_face_aligned(frame, landmarks)
peri_crop = crop_periocular_aligned(frame, landmarks)

full_emb = get_embedding(full_crop)
peri_emb = get_embedding(peri_crop)

inserted_id = enroll_employee("EMP001", "Test Employee", full_emb, peri_emb)
print("Enrolled with ID:", inserted_id)