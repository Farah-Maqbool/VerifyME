# VerifyME 

### Dynamic Face & Periocular Identity Verification System

**The problem:** Normal face recognition needs your whole face visible to work. That's a problem for niqab-wearing women — they usually have to uncover their face just to get verified, which defeats the purpose of wearing it in the first place.

**The solution:** VerifyME looks at whatever *is* visible and picks the right method automatically. Full face showing? It verifies using the full face. Only eyes showing (niqab)? It switches to verifying using just the eyes. No one ever has to uncover anything. It runs on a normal laptop and webcam — no special camera or hardware needed.

---

## Key Features

- **Dynamic occlusion detection** — automatically determines whether a face is fully visible or covered
- **Dual-mode identity verification** — full-face recognition and periocular (eyes-only) recognition, using the same enrollment photo
- **Liveness detection** — blink-based check to prevent spoofing via static photos
- **Real-time processing** — built on lightweight, efficient models suitable for CPU-only inference
- **Web-based interface** — accessible via browser, no software installation needed for end users
- **Cloud-hosted database** — employee records and embeddings stored securely via MongoDB Atlas

---

## Architecture Overview

```
                         ┌─────────────────────┐
                         │   Webcam / Camera    │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  MediaPipe Landmark   │
                         │      Detection        │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │   Occlusion Classification      │
                    │  (Full Face vs. Eyes-Only)      │
                    └───────────────┬────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
   ┌─────────────────────┐                     ┌──────────────────────┐
   │  Full-Face Crop +    │                     │  Periocular Crop +   │
   │     Alignment        │                     │     Alignment        │
   └──────────┬───────────┘                     └───────────┬──────────┘
              │                                              │
              └───────────────────┬──────────────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  ArcFace Embedding     │
                         │     Generation         │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │  Cosine Similarity      │
                         │  Matching vs. Database  │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │  Verified / Denied      │
                         └────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Face & landmark detection | MediaPipe Face Landmarker |
| Occlusion classification | MobileNetV2-based mask classifier (TensorFlow/Keras) |
| Face embedding generation | ArcFace (InsightFace — `w600k_r50`) |
| Liveness detection | Eye Aspect Ratio (EAR) blink detection |
| Backend framework | FastAPI / Streamlit |
| Database | MongoDB Atlas |
| Frontend | HTML, CSS, JavaScript (FastAPI version) / Streamlit components |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.11 |

---

## Project Structure

```
VerifyME/
├── app.py / main.py            # Application entry point
├── requirements.txt            # Pinned dependency versions
├── runtime.txt                 # Python version pin for deployment
├── src/
│   ├── landmarks/               # MediaPipe landmark detection + occlusion classifier
│   ├── crop/                    # Face/periocular cropping & alignment utilities
│   ├── embeddings/               # ArcFace embedding generation
│   ├── matching/                 # Cosine similarity matching logic
│   ├── db/                       # MongoDB connection & queries
│   ├── liveness/                 # Blink detection (EAR-based)
│   └── api/                      # FastAPI routes (enroll / verify)
├── static/                     # CSS & JavaScript (FastAPI frontend)
├── templates/                  # HTML pages (FastAPI frontend)
└── docs/                       # Report, diagrams, evaluation results
```

---

## How It Works

### Enrollment (one-time, per employee)
1. A full-face enrollment photo is captured
2. MediaPipe detects facial landmarks
3. Two aligned crops are generated from the same photo: a full-face crop and a periocular (eyes-only) crop
4. Each crop is passed through the ArcFace model to generate a 512-dimensional embedding
5. Both embeddings are stored in MongoDB against the employee's ID and name

### Verification (every access attempt)
1. Live camera captures a frame
2. System confirms the face is stable and a natural blink has occurred (liveness check)
3. MediaPipe detects landmarks on the captured frame
4. The occlusion classifier determines whether the face is fully visible or covered
5. The corresponding crop type (full-face or periocular) is generated and aligned
6. An embedding is generated from that crop
7. The embedding is compared via cosine similarity against **all stored employees' embeddings of the same type**
8. If the best match exceeds the similarity threshold, the person is verified; otherwise, access is denied

---

## Setup & Installation

### Prerequisites
- Python 3.11
- A MongoDB Atlas account (or local MongoDB instance)

### Steps

```bash
git clone https://github.com/<your-username>/VerifyME.git
cd VerifyME

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file (or `.streamlit/secrets.toml` for the Streamlit version) with:

```
MONGO_URI=your-mongodb-connection-string
ADMIN_USERNAME=your-admin-username
ADMIN_PASSWORD=your-admin-password
```

---

## Running Locally

**Streamlit version:**
```bash
streamlit run app.py
```

**FastAPI version:**
```bash
uvicorn main:app --reload
```
Then open `http://127.0.0.1:8000` in your browser.

---

## Evaluation Results

| Match Type | Genuine Match Score | Impostor Score |
|---|---|---|
| Full-face | ~0.87 | ~0.34 |
| Periocular (eyes-only) | ~0.61 – 0.75 | ~0.34 – 0.49 |

Scores are cosine similarity values (range -1 to 1). A clear separation between genuine and impostor scores confirms the pipeline reliably distinguishes identities in both full-face and periocular modes.

---

## Limitations

- **Not true iris recognition** — this system uses periocular (eye-region) embedding matching via a standard webcam, not infrared iris pattern scanning, which would require specialized hardware
- **Reduced accuracy on extreme niqab styles** — face/eye detection is less reliable on niqab styles with a very narrow eye-slit and no visible forehead/eyebrows
- **Single-frame liveness in the deployed version** — the deployed Streamlit version captures a single photo per attempt rather than continuous video, limiting liveness checks compared to the local prototype
- **Occlusion classifier trained on general mask imagery** — the pretrained classifier was originally trained on surgical/cloth mask data, not niqab-specific imagery; it generalizes well in testing but was not fine-tuned specifically for this use case
- **Threshold tuned on a limited test set** — similarity thresholds were set using a small number of test subjects; a larger, more diverse enrollment/test dataset would allow more rigorous FAR/FRR-based threshold tuning

---

## Future Work

- Fine-tune the occlusion classifier specifically on niqab/covering imagery for improved robustness
- Expand the evaluation dataset across more subjects, lighting conditions, and covering styles
- Restore continuous multi-frame liveness detection in the deployed version
- Explore infrared camera support for true iris-level biometric accuracy
- Add an audit/logging dashboard for verification attempts

---

## Acknowledgements

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — face landmark detection
- [InsightFace](https://github.com/deepinsight/insightface) — ArcFace embedding model
- [Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) — pretrained mask classification model, repurposed for occlusion detection
