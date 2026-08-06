import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'crop'))


import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from occlusion import classify_occlusion, get_full_face_box
from crop_utils import crop_full_face, crop_periocular

BaseOptions = mp_python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

# --- Stability tracking settings ---
STABILITY_FRAMES_REQUIRED = 20
MOVEMENT_THRESHOLD_PX = 8

stable_frame_count = 0
prev_eye_center = None
captured = False
final_result = None

REFERENCE_LANDMARKS = [33, 263]  # left eye corner, right eye corner


def get_eye_center(landmarks, w, h):
    xs = [landmarks[i].x * w for i in REFERENCE_LANDMARKS]
    ys = [landmarks[i].y * h for i in REFERENCE_LANDMARKS]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


print("Position your face in frame and hold still. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    status_text = "No face detected"

    if result.face_landmarks and not captured:
        landmarks = result.face_landmarks[0]

        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Diagnostic box now matches what's actually classified — the full face
        face_box = get_full_face_box(landmarks, w, h, padding=20)
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 255), 2)

        eye_center = get_eye_center(landmarks, w, h)

        if prev_eye_center is not None:
            movement = math.dist(eye_center, prev_eye_center)
            if movement < MOVEMENT_THRESHOLD_PX:
                stable_frame_count += 1
            else:
                stable_frame_count = 0
        prev_eye_center = eye_center

        progress = min(stable_frame_count, STABILITY_FRAMES_REQUIRED)
        status_text = f"Hold still... {progress}/{STABILITY_FRAMES_REQUIRED}"

        if stable_frame_count >= STABILITY_FRAMES_REQUIRED:
            final_result = classify_occlusion(landmarks, frame, debug=True)
            if final_result == "full_face":
                crop = crop_full_face(frame, landmarks)
            elif final_result == "eyes_only":
                crop = crop_periocular(frame, landmarks)
            else:
                crop = None

            if crop is not None:
                cv2.imshow("Cropped Region", crop)
            captured = True
            status_text = f"CAPTURED - Status: {final_result}"

    elif captured:
        status_text = f"CAPTURED - Status: {final_result} (press 'r' to retry)"

    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("Face Landmarker + Occlusion Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        captured = False
        stable_frame_count = 0
        prev_eye_center = None
        final_result = None

cap.release()
cv2.destroyAllWindows()