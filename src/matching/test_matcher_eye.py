import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'crop'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from crop_utils import crop_periocular_aligned
from embedder import get_embedding
from matching import find_best_match

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

# covered-face photo — same person as EMP001, or a different person to test impostor case
frame = cv2.imread("../embeddings/d_2_c.jpeg")
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
result = landmarker.detect(mp_image)

if not result.face_landmarks:
    print("No face/eyes detected in image")
else:
    landmarks = result.face_landmarks[0]
    peri_crop = crop_periocular_aligned(frame, landmarks)
    live_embedding = get_embedding(peri_crop)

    match_result = find_best_match(live_embedding, "eyes_only")
    print(match_result)