import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'db'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embeddings'))

from database import get_all_employees
from embedder import compare_embeddings

# Starting thresholds — will be tuned empirically using real enrollment/test
# data (see note below). Based on our own test pair results (0.87 genuine,
# 0.34 impostor), a value in this range gives comfortable margin on both sides.
FULL_FACE_THRESHOLD = 0.55
PERIOCULAR_THRESHOLD = 0.50  # periocular matching is inherently harder/noisier,
                              # so a slightly lower bar is reasonable — tune with real data


def find_best_match(live_embedding, occlusion_type):
    """
    Compares a live embedding against all enrolled employees, using the
    embedding field that matches `occlusion_type` ('full_face' or 'eyes_only').

    Returns a dict: {
        'matched': bool,
        'employee_id': str or None,
        'name': str or None,
        'score': float or None
    }
    """
    if occlusion_type == "full_face":
        field = "full_face_embedding"
        threshold = FULL_FACE_THRESHOLD
    elif occlusion_type == "eyes_only":
        field = "periocular_embedding"
        threshold = PERIOCULAR_THRESHOLD
    else:
        return {"matched": False, "employee_id": None, "name": None, "score": None}

    employees = get_all_employees()

    best_score = -1.0
    best_employee = None

    for emp in employees:
        stored_embedding = emp.get(field)
        if stored_embedding is None:
            continue

        score = compare_embeddings(live_embedding, stored_embedding)
        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_employee = emp

    if best_employee is not None and best_score >= threshold:
        return {
            "matched": True,
            "employee_id": best_employee["employee_id"],
            "name": best_employee["name"],
            "score": best_score,
        }

    return {
        "matched": False,
        "employee_id": None,
        "name": None,
        "score": best_score if best_employee is not None else None,
    }