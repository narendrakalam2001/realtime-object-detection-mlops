# ============================================================
# EVALUATION — Real-Time Object Detection ML System
# ============================================================

import time
import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data_loader import COCO_CLASSES
from src.metrics import psi, latency_stats, match_predictions_to_ground_truth

logger = logging.getLogger(__name__)


# ============================================================
# COST-SENSITIVE EVALUATION (business-dollar impact)
# ============================================================
# Aggregate version driven by real precision/recall from
# ultralytics val() + the real ground-truth instance count from
# coco128 — no per-frame simulation, no fabricated labels.
#
#   missed_detection_cost -> a real object the model failed to
#                             surface (false negative)
#   false_alarm_cost       -> a detection that fired but didn't
#                             match any ground-truth box (false
#                             positive) — analyst time reviewing noise

def cost_sensitive_evaluation(
    precision: float,
    recall:    float,
    total_ground_truth_instances: int,
    missed_detection_cost: float = 500.0,   # cost of a missed intrusion/defect event
    false_alarm_cost:      float = 20.0,    # analyst time reviewing a false alert
) -> dict:
    true_positives  = recall * total_ground_truth_instances
    false_negatives = total_ground_truth_instances - true_positives
    predicted_positives = true_positives / precision if precision > 0 else true_positives
    false_positives = max(0.0, predicted_positives - true_positives)

    missed_loss = false_negatives * missed_detection_cost
    false_cost  = false_positives * false_alarm_cost
    total       = missed_loss + false_cost

    result = {
        "total_ground_truth_instances": int(total_ground_truth_instances),
        "estimated_true_positives":     round(true_positives, 1),
        "estimated_false_negatives":    round(false_negatives, 1),
        "estimated_false_positives":    round(false_positives, 1),
        "estimated_missed_loss":        round(missed_loss, 2),
        "estimated_false_alarm_cost":   round(false_cost, 2),
        "total_estimated_cost":         round(total, 2),
    }

    logger.info(
        "Cost eval  |  FN~%.1f  FP~%.1f  missed_loss=%.2f  false_cost=%.2f  total=%.2f",
        false_negatives, false_positives, missed_loss, false_cost, total
    )
    return result


# ============================================================
# CONFIDENCE CALIBRATION — isotonic regression (real matching)
# ============================================================
# Runs the model over the val split, matches every prediction to
# real ground truth via IoU (src/metrics.match_predictions_to_ground_truth),
# and fits confidence -> P(correct) using genuine correct/incorrect
# labels — not a proxy.

def build_calibration_dataset(model, val_records: list, conf: float, iou_thr: float) -> tuple:
    """Returns (confidences, correctness_labels) across the whole val split."""
    confidences, correctness = [], []

    for record in val_records:
        results = model.predict(record["image_path"], conf=conf, iou=iou_thr, verbose=False)
        r = results[0]

        predictions = [
            {"class_id": int(c), "confidence": float(cf), "bbox_xyxy": list(map(float, box))}
            for c, cf, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
        ]
        ground_truth = [
            {"class_id": a["class_id"], "bbox_xyxy": a["bbox_xyxy"]}
            for a in record["annotations"]
        ]

        matched = match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold=0.5)
        for m in matched:
            confidences.append(m["confidence"])
            correctness.append(1 if m["is_correct"] else 0)

    return confidences, correctness


def calibrate_confidence_scores(raw_confidences: list, correctness_labels: list):
    if len(raw_confidences) < 20 or len(set(correctness_labels)) < 2:
        logger.warning(
            "Not enough signal to fit isotonic calibration "
            "(need >=20 points and both classes present) — skipping, "
            "raw confidences will be used uncalibrated."
        )
        return None

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_confidences, correctness_labels)

    logger.info("Isotonic confidence calibrator fitted on %d real matched detections", len(raw_confidences))
    return calibrator


def apply_calibration(calibrator, confidences: list) -> list:
    if calibrator is None:
        return list(confidences)
    return calibrator.predict(confidences).tolist()


# ============================================================
# DRIFT REPORT — confidence & detection-count PSI
# ============================================================

def generate_drift_report(
    reference_confidences: list,
    live_confidences:      list,
    reference_counts:      list,
    live_counts:            list,
) -> pd.DataFrame:
    rows = [
        {"feature": "confidence_score",  "drift_score": psi(reference_confidences, live_confidences)},
        {"feature": "objects_per_frame", "drift_score": psi(reference_counts, live_counts)},
    ]
    return pd.DataFrame(rows).sort_values("drift_score", ascending=False).reset_index(drop=True)


# ============================================================
# FULL MODEL EVALUATION — one variant, real coco128 val split
# ============================================================

def evaluate_variant(
    variant_name: str,
    model,
    val_records: list,
    conf: float,
    iou_thr: float,
    ultralytics_metrics: dict,
    domain: str = "security_intrusion",
) -> dict:
    """
    Assembles the full metrics bundle for the model card + champion-
    challenger comparison: real ultralytics mAP/precision/recall
    (already computed via model.val() in model_tuning.py — passed in
    as `ultralytics_metrics`), real cost-sensitive evaluation, real
    isotonic calibration, and per-frame alert-engine decision counts.
    """
    from src.detection_engine import score_frame, summarize_decisions

    total_gt_instances = sum(len(r["annotations"]) for r in val_records)

    cost_result = cost_sensitive_evaluation(
        precision=ultralytics_metrics["precision"],
        recall=ultralytics_metrics["recall"],
        total_ground_truth_instances=total_gt_instances,
    )

    # ── Confidence calibration on real matched detections ─────────
    confidences, correctness = build_calibration_dataset(model, val_records, conf, iou_thr)
    calibrator = calibrate_confidence_scores(confidences, correctness)
    calibration_status = "fitted" if calibrator is not None else "skipped_insufficient_data"

    # ── Alert-engine decisions over the val split ──────────────────
    decisions = []
    all_counts = []
    latencies = []

    for record in val_records:
        t0 = time.perf_counter()
        results = model.predict(record["image_path"], conf=conf, iou=iou_thr, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        r = results[0]
        detections = [
            {"class_name": COCO_CLASSES[int(c)], "confidence": float(cf)}
            for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())
        ]
        all_counts.append(len(detections))
        decisions.append(score_frame(detections, domain=domain))

    decision_counts = summarize_decisions(decisions)
    lat = latency_stats(latencies)

    metrics = {
        "map50_95":        ultralytics_metrics["map50_95"],
        "map50":           ultralytics_metrics["map50"],
        "precision":       ultralytics_metrics["precision"],
        "recall":          ultralytics_metrics["recall"],
        "mean_latency_ms": ultralytics_metrics["mean_latency_ms"],
        "fps":             ultralytics_metrics["fps"],
    }

    logger.info(
        "Evaluated %s | mAP50-95=%.4f  P=%.3f  R=%.3f  calibration=%s",
        variant_name, metrics["map50_95"], metrics["precision"], metrics["recall"], calibration_status
    )

    return {
        "metrics":            metrics,
        "cost_result":        cost_result,
        "decision_counts":    decision_counts,
        "confidences":        confidences,
        "counts_per_frame":   all_counts,
        "calibration_status": calibration_status,
        "n_calibration_points": len(confidences),
    }
