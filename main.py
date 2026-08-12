import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'landmarks'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'crop'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'embeddings'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'matching'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'db'))

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from occlusion import classify_occlusion
from crop_utils import crop_full_face_aligned, crop_periocular_aligned
from embedder import get_embedding
from matching import find_best_match
from database import enroll_employee, get_employee_by_id


# --- Page config ---
st.set_page_config(page_title="VerifyME", page_icon="🔒", layout="centered")


# --- Load MongoDB URI from secrets into environment (database.py reads from os.getenv) ---
os.environ["MONGO_URI"] = st.secrets.get("MONGO_URI", os.environ.get("MONGO_URI", ""))


# --- Load Face Landmarker once, cached across reruns ---
@st.cache_resource
def load_landmarker():
    BaseOptions = mp_python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    model_path = os.path.join(os.path.dirname(__file__), 'src', 'landmarks', 'face_landmarker.task')

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)


landmarker = load_landmarker()


def streamlit_image_to_cv2(uploaded_file):
    """Converts a Streamlit camera_input/file_uploader object into an OpenCV BGR image."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return frame


def process_frame(frame):
    """
    Runs the full pipeline on a single frame:
    landmarks -> occlusion classification -> crop+align.
    Returns (occlusion_status, crop) or (None, None) if no face detected.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None

    landmarks = result.face_landmarks[0]
    occlusion_status = classify_occlusion(landmarks, frame)

    if occlusion_status == "full_face":
        crop = crop_full_face_aligned(frame, landmarks)
    elif occlusion_status == "eyes_only":
        crop = crop_periocular_aligned(frame, landmarks)
    else:
        crop = None

    return occlusion_status, crop


# --- Sidebar navigation ---
page = st.sidebar.radio("Navigate", ["Verify", "Enroll (Admin)"])


# ============================================================
# VERIFY PAGE
# ============================================================
if page == "Verify":
    st.title("🔒 VerifyME — Identity Verification")
    st.write("Position your face in frame and take a photo.")

    photo = st.camera_input("Verification photo")

    if photo is not None:
        frame = streamlit_image_to_cv2(photo)

        with st.spinner("Verifying..."):
            occlusion_status, crop = process_frame(frame)

            if occlusion_status is None:
                st.error("No face detected. Please try again.")
            elif crop is None:
                st.error(f"Could not process face region (status: {occlusion_status}).")
            else:
                embedding = get_embedding(crop)
                result = find_best_match(embedding, occlusion_status)

                if result["matched"]:
                    st.success(
                        f"✅ Verified: **{result['name']}** ({result['employee_id']})  \n"
                        f"Score: {result['score']:.3f} | Type: {occlusion_status}"
                    )
                else:
                    score_text = f"{result['score']:.3f}" if result["score"] is not None else "N/A"
                    st.error(
                        f"❌ Not verified.  \n"
                        f"Best score: {score_text} | Type: {occlusion_status}"
                    )


# ============================================================
# ENROLL PAGE (admin-protected)
# ============================================================
elif page == "Enroll (Admin)":
    st.title("🔒 VerifyME — Enroll New Employee (Admin Only)")

    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        password = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            if password == st.secrets.get("ADMIN_PASSWORD", ""):
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")

    else:
        st.success("Logged in as admin.")
        if st.button("Log out"):
            st.session_state.admin_authenticated = False
            st.rerun()

        st.divider()

        employee_id = st.text_input("Employee ID")
        employee_name = st.text_input("Full Name")
        photo = st.camera_input("Enrollment photo (full face, uncovered)")

        if st.button("Enroll Employee"):
            if not employee_id or not employee_name:
                st.error("Please enter both Employee ID and Name.")
            elif photo is None:
                st.error("Please take a photo first.")
            elif get_employee_by_id(employee_id) is not None:
                st.error(f"Employee ID '{employee_id}' is already enrolled.")
            else:
                frame = streamlit_image_to_cv2(photo)

                with st.spinner("Processing enrollment..."):
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = landmarker.detect(mp_image)

                    if not result.face_landmarks:
                        st.error("No face detected in the photo. Please retake with a clear, uncovered face.")
                    else:
                        landmarks = result.face_landmarks[0]
                        full_crop = crop_full_face_aligned(frame, landmarks)
                        peri_crop = crop_periocular_aligned(frame, landmarks)

                        if full_crop is None or peri_crop is None:
                            st.error("Could not generate crops from this photo. Please try again.")
                        else:
                            full_embedding = get_embedding(full_crop)
                            peri_embedding = get_embedding(peri_crop)

                            inserted_id = enroll_employee(
                                employee_id, employee_name, full_embedding, peri_embedding
                            )

                            st.success(f"✅ Enrolled successfully: {employee_name} ({employee_id})")