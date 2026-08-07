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
    """
    Produces an aligned periocular (eyes-only) crop, using ONLY eye
    landmarks for alignment — since nose/mouth landmarks are unreliable
    or covered when the face is occluded (e.g. niqab). Rotates the crop
    so the eyes are level, then scales/crops to a consistent size.
    """
    h, w = frame.shape[:2]

    def avg_point(indices):
        xs = [landmarks[i].x * w for i in indices]
        ys = [landmarks[i].y * h for i in indices]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    left_eye = avg_point(LEFT_EYE_CENTER_IDX)
    right_eye = avg_point(RIGHT_EYE_CENTER_IDX)

    # Angle to rotate so the eye line is horizontal
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(frame, rot_matrix, (w, h))

    # Crop around the eye region post-rotation
    inter_eye_dist = np.hypot(dx, dy)
    box_half = inter_eye_dist * 1.8  # margin around the eyes

    x_min = max(int(eye_center[0] - box_half), 0)
    x_max = min(int(eye_center[0] + box_half), w)
    y_min = max(int(eye_center[1] - box_half * 0.7), 0)
    y_max = min(int(eye_center[1] + box_half * 0.7), h)

    if x_max <= x_min or y_max <= y_min:
        return None

    cropped = rotated[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return None

    return cv2.resize(cropped, (output_size, output_size))