import numpy as np

# Eye landmark indices for EAR calculation (MediaPipe Face Landmarker, 468/478-point model)
# Each list: [left_corner, right_corner, top1, top2, bottom1, bottom2]
# Using vertical pairs + horizontal corners, standard 6-point EAR formulation
LEFT_EYE_EAR_POINTS = [33, 133, 159, 145, 158, 153]
RIGHT_EYE_EAR_POINTS = [362, 263, 386, 374, 385, 380]

# EAR drops below this during a blink; above this = eyes open
EAR_BLINK_THRESHOLD = 0.21

# How many consecutive low-EAR frames count as a genuine blink
# (avoids counting a single noisy frame as a blink)
BLINK_CONSEC_FRAMES = 2


def _euclidean(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(landmarks, eye_points, w, h):
    """
    Computes the Eye Aspect Ratio for one eye.
    eye_points: [left_corner_idx, right_corner_idx, top1_idx, bottom1_idx, top2_idx, bottom2_idx]
    """
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    left_corner, right_corner, top1, bottom1, top2, bottom2 = coords

    horizontal = _euclidean(left_corner, right_corner)
    vertical1 = _euclidean(top1, bottom1)
    vertical2 = _euclidean(top2, bottom2)

    if horizontal == 0:
        return 0.0

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


def get_average_ear(landmarks, w, h):
    """
    Returns the average EAR across both eyes for a single frame.
    """
    left_ear = calculate_ear(landmarks, LEFT_EYE_EAR_POINTS, w, h)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_EAR_POINTS, w, h)
    return (left_ear + right_ear) / 2.0


class BlinkDetector:
    """
    Stateful blink counter — call update() once per frame during the
    stability-hold period. Tracks consecutive low-EAR frames and counts
    a completed blink once EAR rises back above threshold afterward.
    """

    def __init__(self, ear_threshold=EAR_BLINK_THRESHOLD, consec_frames=BLINK_CONSEC_FRAMES):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.low_ear_count = 0
        self.blink_count = 0

    def update(self, ear_value):
        """
        Call once per frame with the current EAR value.
        Returns True the exact frame a blink is confirmed complete.
        """
        blink_just_completed = False

        if ear_value < self.ear_threshold:
            self.low_ear_count += 1
        else:
            if self.low_ear_count >= self.consec_frames:
                self.blink_count += 1
                blink_just_completed = True
            self.low_ear_count = 0

        return blink_just_completed

    def has_blinked(self):
        return self.blink_count > 0

    def reset(self):
        self.low_ear_count = 0
        self.blink_count = 0