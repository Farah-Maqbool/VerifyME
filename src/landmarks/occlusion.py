import cv2
import numpy as np
import tensorflow as tf
import tf_keras

LEFT_EYE = [33, 133, 160, 159, 158, 144, 153, 154]
RIGHT_EYE = [362, 263, 387, 386, 385, 373, 380, 381]
MOUTH_REGION = [61, 291, 13, 14, 78, 308, 17, 0]

mask_net = tf_keras.models.load_model("mask_detector.model")

CONFIDENCE_FLOOR = 70.0  # percent — below this, treat as uncertain


def get_bounding_box(landmarks, indices, frame_width, frame_height, padding=20):
    xs = [landmarks[i].x * frame_width for i in indices]
    ys = [landmarks[i].y * frame_height for i in indices]

    x_min = max(int(min(xs)) - padding, 0)
    x_max = min(int(max(xs)) + padding, frame_width)
    y_min = max(int(min(ys)) - padding, 0)
    y_max = min(int(max(ys)) + padding, frame_height)

    return x_min, y_min, x_max, y_max


def get_full_face_box(landmarks, frame_width, frame_height, padding=20):
    all_indices = list(range(len(landmarks)))
    return get_bounding_box(landmarks, all_indices, frame_width, frame_height, padding)


def classify_occlusion(landmarks, frame, debug=False):
    """
    landmarks: list of landmark objects (from result.face_landmarks[0], Tasks API format)
    Returns one of: 'full_face', 'eyes_only', 'uncertain', 'insufficient'
    """
    h, w = frame.shape[:2]

    eye_indices = LEFT_EYE + RIGHT_EYE
    eye_box = get_bounding_box(landmarks, eye_indices, w, h)
    if eye_box[2] <= eye_box[0] or eye_box[3] <= eye_box[1]:
        return "insufficient"

    # Use the WHOLE face region — matches how the pretrained classifier was trained
    face_box = get_full_face_box(landmarks, w, h, padding=20)
    x_min, y_min, x_max, y_max = face_box
    region = frame[y_min:y_max, x_min:x_max]

    if region.size == 0:
        return "insufficient"

    if debug:
        print(f"Face crop shape: {region.shape}")

    face_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    face_input = tf_keras.applications.mobilenet_v2.preprocess_input(face_resized.astype("float32"))
    face_input = np.expand_dims(face_input, axis=0)

    (mask, without_mask) = mask_net.predict(face_input, verbose=0)[0]
    label = "eyes_only" if mask > without_mask else "full_face"
    confidence_pct = max(mask, without_mask) * 100

    if debug:
        print(f"Label: {label} | Confidence: {confidence_pct:.2f}%")

    if confidence_pct < CONFIDENCE_FLOOR:
        return "uncertain"

    return label