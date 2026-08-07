from insightface.utils import face_align
import numpy as np

# MediaPipe landmark indices approximating the 5 standard ArcFace alignment points
LEFT_EYE_CENTER_IDX = [33, 133, 160, 159, 158, 144, 153, 154]
RIGHT_EYE_CENTER_IDX = [362, 263, 387, 386, 385, 373, 380, 381]
NOSE_TIP_IDX = 1
LEFT_MOUTH_CORNER_IDX = 61
RIGHT_MOUTH_CORNER_IDX = 291


def get_5_point_landmarks(landmarks, w, h):
    def avg_point(indices):
        xs = [landmarks[i].x * w for i in indices]
        ys = [landmarks[i].y * h for i in indices]
        return [sum(xs) / len(xs), sum(ys) / len(ys)]

    left_eye = avg_point(LEFT_EYE_CENTER_IDX)
    right_eye = avg_point(RIGHT_EYE_CENTER_IDX)
    nose = [landmarks[NOSE_TIP_IDX].x * w, landmarks[NOSE_TIP_IDX].y * h]
    left_mouth = [landmarks[LEFT_MOUTH_CORNER_IDX].x * w, landmarks[LEFT_MOUTH_CORNER_IDX].y * h]
    right_mouth = [landmarks[RIGHT_MOUTH_CORNER_IDX].x * w, landmarks[RIGHT_MOUTH_CORNER_IDX].y * h]

    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)


def crop_full_face_aligned(frame, landmarks, output_size=112):
    """
    Produces a properly aligned 112x112 face crop, matching the exact
    preprocessing ArcFace-based models (like our w600k_r50 recognition
    model) were trained on. Use this instead of crop_full_face() when
    the output is going into get_embedding().
    """
    h, w = frame.shape[:2]
    kps = get_5_point_landmarks(landmarks, w, h)
    aligned = face_align.norm_crop(frame, kps, image_size=output_size)
    return aligned



import cv2

def crop_periocular_aligned(frame, landmarks, output_size=112):
    h, w = frame.shape[:2]

    def avg_point(indices):
        xs = [landmarks[i].x * w for i in indices]
        ys = [landmarks[i].y * h for i in indices]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    left_eye = avg_point(LEFT_EYE_CENTER_IDX)
    right_eye = avg_point(RIGHT_EYE_CENTER_IDX)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(frame, rot_matrix, (w, h))

    inter_eye_dist = np.hypot(dx, dy)

    # Tightened: much closer crop around just the eye region
    half_width = inter_eye_dist * 1.0     # was 1.8 — too wide
    half_height_top = inter_eye_dist * 0.6   # room above for eyebrows
    half_height_bottom = inter_eye_dist * 0.5  # room below for upper cheek/under-eye

    x_min = max(int(eye_center[0] - half_width), 0)
    x_max = min(int(eye_center[0] + half_width), w)
    y_min = max(int(eye_center[1] - half_height_top), 0)
    y_max = min(int(eye_center[1] + half_height_bottom), h)

    if x_max <= x_min or y_max <= y_min:
        return None

    cropped = rotated[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return None

    return cv2.resize(cropped, (output_size, output_size))