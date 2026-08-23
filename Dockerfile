FROM python:3.11-slim

WORKDIR /app

# System deps required by opencv/ultralytics/pillow at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY src/ ./src/
COPY serving/ ./serving/
COPY services/ ./services/
COPY detection_models/ ./detection_models/
COPY data/ ./data/

# Pre-download the champion model weights at build time so the
# container doesn't hit the network on cold start.
RUN python -c "from ultralytics import YOLO; import json, os; \
    reg = json.load(open('detection_models/latest_model.json')) if os.path.exists('detection_models/latest_model.json') else {'weights':'yolov8n.pt'}; \
    YOLO(reg.get('weights','yolov8n.pt'))"

EXPOSE 8000

CMD uvicorn serving.detection_api:app --host 0.0.0.0 --port ${PORT:-8000}
