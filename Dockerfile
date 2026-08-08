FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV, MediaPipe, and TensorFlow on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching —
# this step only re-runs if requirements.txt changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Pre-downloaded InsightFace models — copied into the image so the
# container doesn't need to download them on first request
COPY insightface_models/buffalo_l /root/.insightface/models/buffalo_l

# Cloud Run injects the PORT environment variable at runtime
ENV PORT=8080
EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port $PORT