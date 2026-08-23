# ============================================================
# PREDICTION SERVICE — Real-Time Object Detection ML System
# ============================================================

import io
import time
import logging

from PIL import Image

from src.data_loader import COCO_CLASSES
from src.detection_engine import score_frame

logger = logging.getLogger(__name__)


# ============================================================
# RUN INFERENCE — single image
# ============================================================

def run_detection(model, image: Image.Image, conf: float, iou: float) -> list:
    """
    Runs the loaded YOLO model on a single PIL image.
    Returns a list of detection dicts: class_name, confidence, box.
    """
    t0 = time.perf_counter()
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    r = results[0]
    detections = []
    for cls_id, conf_val, box in zip(
        r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()
    ):
        detections.append({
            "class_name": COCO_CLASSES.get(int(cls_id), f"class_{int(cls_id)}"),
            "confidence": round(float(conf_val), 4),
            "box_xyxy":   [round(v, 1) for v in box],
        })

    return detections, round(latency_ms, 2)


# ============================================================
# PREDICT — full flow for the API
# ============================================================

def predict_frame(
    model,
    image_bytes: bytes,
    conf: float,
    iou: float,
    domain: str = "security_intrusion",
) -> dict:
    """
    Full prediction flow for one uploaded frame:
      1. Decode image
      2. Run YOLO inference
      3. Score frame through the detection/alert engine
    Returns structured API response dict.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    detections, latency_ms = run_detection(model, image, conf, iou)
    decision = score_frame(detections, domain=domain)

    result = {
        "detections":       detections,
        "detection_count":  len(detections),
        "domain":           domain,
        "decision":         decision["decision"],
        "rule_triggered":   decision["rule_triggered"],
        "person_count":     decision["person_count"],
        "latency_ms":       latency_ms,
        "image_size":       list(image.size),
    }

    logger.info(
        "Prediction  |  objects=%d  decision=%s  latency=%.1fms",
        result["detection_count"], result["decision"], latency_ms
    )

    return result
