import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'landmarks'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'crop'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'matching'))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from fastapi import APIRouter, UploadFile, File, HTTPException

from occlusion import classify_occlusion
from crop_utils import crop_full_face_aligned, crop_periocular_aligned
from embedder import get_embedding
from matcher import find_best_match

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
async def verify_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {"verified": False, "reason": "No face detected"}

    landmarks = result.face_landmarks[0]

    occlusion_status = classify_occlusion(landmarks, frame)

    if occlusion_status == "full_face":
        crop = crop_full_face_aligned(frame, landmarks)
    elif occlusion_status == "eyes_only":
        crop = crop_periocular_aligned(frame, landmarks)
    else:
        return {"verified": False, "reason": f"Occlusion status: {occlusion_status}"}

    if crop is None:
        return {"verified": False, "reason": "Could not crop face region"}

    embedding = get_embedding(crop)
    match_result = find_best_match(embedding, occlusion_status)

    return {
        "verified": match_result["matched"],
        "employee_id": match_result["employee_id"],
        "name": match_result["name"],
        "score": match_result["score"],
        "occlusion_type": occlusion_status,
    }