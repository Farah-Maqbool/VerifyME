import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'landmarks'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'crop'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'db'))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from crop_utils import crop_full_face_aligned, crop_periocular_aligned
from embedder import get_embedding
from database import enroll_employee, get_employee_by_id

router = APIRouter()

BaseOptions = mp_python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'landmarks', 'face_landmarker.task')

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)
landmarker = FaceLandmarker.create_from_options(options)


@router.post("/")
async def enroll_new_employee(
    employee_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...)
):
    if get_employee_by_id(employee_id) is not None:
        raise HTTPException(status_code=400, detail="Employee ID already enrolled")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise HTTPException(status_code=400, detail="No face detected in enrollment photo")

    landmarks = result.face_landmarks[0]

    full_crop = crop_full_face_aligned(frame, landmarks)
    peri_crop = crop_periocular_aligned(frame, landmarks)

    if full_crop is None or peri_crop is None:
        raise HTTPException(status_code=400, detail="Could not generate crops from photo")

    full_embedding = get_embedding(full_crop)
    peri_embedding = get_embedding(peri_crop)

    inserted_id = enroll_employee(employee_id, name, full_embedding, peri_embedding)

    return {"status": "enrolled", "employee_id": employee_id, "db_id": str(inserted_id)}