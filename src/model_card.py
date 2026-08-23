# ============================================================
# MODEL CARD — Real-Time Object Detection ML System
# ============================================================
# Structured model card (Google Model Cards standard) covering:
#   - Model identity + pretrained provenance
#   - Threshold configuration
#   - Weak-label + published-benchmark metrics
#   - Latency / throughput
#   - Business alert-decision summary
#   - Cost-sensitive evaluation
# ============================================================

import os
import json
import time
import logging

from typing import Optional

logger = logging.getLogger(__name__)


def build_model_card(
    variant:          str,
    weights:          str,
    conf_threshold:   float,
    iou_threshold:    float,
    metrics:          dict,     # map50_95, map50, precision, recall, mean_latency_ms, fps
    coco_benchmark:   dict,     # officially published map50_95 / map50 / params_m
    cost_result:      dict,
    decision_counts:  dict,
    val_set_size:     int,
    calibration_status: str = "unknown",
    n_calibration_points: int = 0,
    version:          str = "v1",
) -> dict:

    card = {
        # ── Identity ──────────────────────────────────────────
        "model_version": version,
        "model_name":    variant,
        "weights_file":  weights,
        "trained_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "project":       "Real-Time Object Detection System",
        "provenance":    "Pretrained on COCO train2017 by Ultralytics — no local training run",

        # ── Dataset info ──────────────────────────────────────
        "dataset": {
            "source":       "coco128 (Ultralytics) — a 128-image subset of real COCO train2017",
            "download":     "Auto-downloaded via ultralytics on first use — no manual download",
            "val_set_size": val_set_size,
            "note": (
                "Metrics below are computed by ultralytics' own validator "
                "(model.val()) against coco128's real ground-truth labels — "
                "genuine mAP/precision/recall, not a proxy. coco128 is a "
                "128-image sanity-check subset (not the full 5,000-image "
                "COCO val2017), so its mAP will differ from the officially "
                "published val2017 benchmark reported alongside it below."
            ),
        },

        # ── Threshold config ───────────────────────────────────
        "thresholds": {
            "confidence": round(float(conf_threshold), 4),
            "iou":        round(float(iou_threshold), 4),
        },

        # ── Real metrics on coco128 (this environment) ──────────
        "metrics": {
            "map50_95":        round(float(metrics.get("map50_95", 0)), 4),
            "map50":           round(float(metrics.get("map50", 0)), 4),
            "precision":       round(float(metrics.get("precision", 0)), 4),
            "recall":          round(float(metrics.get("recall", 0)), 4),
            "mean_latency_ms": round(float(metrics.get("mean_latency_ms", 0)), 2),
            "fps":             round(float(metrics.get("fps", 0)), 2),
        },

        # ── Published COCO val2017 benchmark (Ultralytics official) ─
        "published_coco_benchmark": coco_benchmark,

        # ── Business alert decisions ────────────────────────────
        "alert_decisions": decision_counts,

        # ── Cost evaluation ─────────────────────────────────────
        "cost_evaluation": cost_result,

        # ── Confidence calibration ───────────────────────────────
        "calibration": {
            "status":            calibration_status,
            "n_points":          n_calibration_points,
            "method":            "isotonic regression on IoU-matched detections",
        },
    }

    return card


def save_model_card(card: dict, model_dir: str, variant: str, version: str = "v1") -> str:
    os.makedirs(model_dir, exist_ok=True)
    card_path = os.path.join(model_dir, f"model_card_{variant}_{version}.json")

    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)

    logger.info("Model card saved -> %s", card_path)
    return card_path


def load_model_card(model_dir: str, variant: str, version: str = "v1") -> dict:
    card_path = os.path.join(model_dir, f"model_card_{variant}_{version}.json")

    if not os.path.exists(card_path):
        raise FileNotFoundError(f"Model card not found: {card_path}")

    with open(card_path) as f:
        card = json.load(f)

    logger.info("Model card loaded <- %s", card_path)
    return card
