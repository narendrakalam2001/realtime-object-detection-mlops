# ============================================================
# OBJECT DETECTION API — FastAPI Serving
# ============================================================

import os
import json
import time
import logging

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

from src.model_loader import load_latest_model
from services.prediction_service import predict_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-Time Object Detection API")

MODEL_DIR = "detection_models"

# ── Load champion model on startup ────────────────────────────
try:
    model, conf_threshold, iou_threshold = load_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model = None
    conf_threshold, iou_threshold = 0.25, 0.45


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Real-Time Object Detection API is live",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}


@app.get("/model_info")
def model_info():
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {"error": "Model registry not found"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    domain: str = Query(
        "security_intrusion",
        description="retail_shelf | warehouse_safety | security_intrusion",
    ),
):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    start = time.time()
    image_bytes = await file.read()

    try:
        result = predict_frame(model, image_bytes, conf_threshold, iou_threshold, domain=domain)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

    # ── Log prediction ────────────────────────────────────────
    log_record = {
        "timestamp":       time.time(),
        "filename":        file.filename,
        "domain":          domain,
        "detection_count": result["detection_count"],
        "person_count":    result["person_count"],
        "decision":        result["decision"],
        "rule_triggered":  result.get("rule_triggered"),
        "latency_ms":      result["latency_ms"],
    }

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/prediction_logs.csv"
    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["total_latency_seconds"] = round(time.time() - start, 4)
    return result
