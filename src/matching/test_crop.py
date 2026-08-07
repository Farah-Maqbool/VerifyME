import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'crop'))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from crop_utils import crop_full_face_aligned, crop_periocular_aligned

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


def show_crops(image_path, label):
    frame = cv2.imread(image_path)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print(f"No face detected in {image_path}")
        return

    landmarks = result.face_landmarks[0]
    full_crop = crop_full_face_aligned(frame, landmarks)
    peri_crop = crop_periocular_aligned(frame, landmarks)

    if full_crop is not None:
        cv2.imshow(f"{label} - Full Face", full_crop)
    if peri_crop is not None:
        cv2.imshow(f"{label} - Periocular", peri_crop)


# your enrollment photo (uncovered)
show_crops("../embeddings/S_1.jpeg", "Enrollment")

# your covered verification photo
show_crops("../embeddings/s_1_c.jpeg", "Verification")

cv2.waitKey(0)
cv2.destroyAllWindows()