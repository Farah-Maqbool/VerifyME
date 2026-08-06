import cv2
import numpy as np

# Landmark index groups (MediaPipe Face Landmarker, 468/478-point model)
LEFT_EYE = [33, 133, 160, 159, 158, 144, 153, 154]
RIGHT_EYE = [362, 263, 387, 386, 385, 373, 380, 381]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295]

# Eye corner landmarks, used as a stable reference for alignment/scale
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def _get_bounding_box(landmarks, indices, frame_width, frame_height, padding=20):
    """
    Returns a pixel-space bounding box (x_min, y_min, x_max, y_max) around the
    given landmark indices, expanded by `padding` pixels on each side and
    clamped to the frame boundaries.
    """
    xs = [landmarks[i].x * frame_width for i in indices]
    ys = [landmarks[i].y * frame_height for i in indices]

    x_min = max(int(min(xs)) - padding, 0)
    x_max = min(int(max(xs)) + padding, frame_width)
    y_min = max(int(min(ys)) - padding, 0)
    y_max = min(int(max(ys)) + padding, frame_height)

    return x_min, y_min, x_max, y_max


def _safe_crop(frame, box):
    """
    Crops `frame` to `box` (x_min, y_min, x_max, y_max). Returns None if the
    resulting region is empty (e.g. a degenerate/zero-size box).
    """
    x_min, y_min, x_max, y_max = box
    if x_max <= x_min or y_max <= y_min:
        return None
    region = frame[y_min:y_max, x_min:x_max]
    if region.size == 0:
        return None
    return region


def crop_full_face(frame, landmarks, output_size=224, padding=20):
    """
    Crops the whole face region from `frame` using all detected landmarks,
    and resizes it to a fixed (output_size x output_size) square image.

    frame: the raw BGR camera frame (numpy array)
    landmarks: list of landmark objects from result.face_landmarks[0]
               (MediaPipe Tasks API format, each with .x / .y in [0, 1])
    output_size: side length (pixels) of the returned square crop
    padding: extra margin (pixels) added around the tightest bounding box

    Returns a resized BGR image (numpy array), or None if cropping failed.
    """
    h, w = frame.shape[:2]
    all_indices = list(range(len(landmarks)))
    box = _get_bounding_box(landmarks, all_indices, w, h, padding=padding)

    region = _safe_crop(frame, box)
    if region is None:
        return None

    resized = cv2.resize(region, (output_size, output_size))
    return resized


def crop_periocular(frame, landmarks, output_size=224, padding=15):
    """
    Crops just the eye region (both eyes + eyebrows + immediate surrounding
    skin) from `frame`, and resizes it to a fixed (output_size x output_size)
    square image. Used when the lower face is covered (e.g. niqab) and only
    the eyes are available for verification.

    frame: the raw BGR camera frame (numpy array)
    landmarks: list of landmark objects from result.face_landmarks[0]
    output_size: side length (pixels) of the returned square crop
    padding: extra margin (pixels) added around the tightest bounding box

    Returns a resized BGR image (numpy array), or None if cropping failed.
    """
    h, w = frame.shape[:2]
    indices = LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW
    box = _get_bounding_box(landmarks, indices, w, h, padding=padding)

    region = _safe_crop(frame, box)
    if region is None:
        return None

    resized = cv2.resize(region, (output_size, output_size))
    return resized


def get_eye_alignment_angle(landmarks, w, h):
    """
    Returns the rotation angle (in degrees) of the line connecting the two
    outer eye corners, relative to horizontal. Useful later for rotating a
    crop so the eyes are level before generating an embedding — a tilted
    head can otherwise hurt embedding consistency.

    landmarks: list of landmark objects from result.face_landmarks[0]
    """
    left = landmarks[LEFT_EYE_OUTER]
    right = landmarks[RIGHT_EYE_OUTER]

    left_x, left_y = left.x * w, left.y * h
    right_x, right_y = right.x * w, right.y * h

    dx = right_x - left_x
    dy = right_y - left_y
    angle = np.degrees(np.arctan2(dy, dx))
    return angle