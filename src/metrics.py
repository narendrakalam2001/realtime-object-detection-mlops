# ============================================================
# METRICS — Real-Time Object Detection ML System
# ============================================================

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# PSI — Population Stability Index
# ============================================================
# Identical formulation to the tabular projects: correct edge-based
# implementation (bin edges computed on the reference distribution,
# then applied unchanged to the new distribution). Here it is used
# to monitor drift in the *confidence-score* distribution and the
# *objects-per-frame* distribution between the reference (benchmark)
# batch and live production traffic.

def psi(expected, actual, buckets: int = 10) -> float:
    """
    PSI < 0.1   → stable
    PSI 0.1–0.2 → moderate drift, monitor
    PSI > 0.2   → major drift, investigate camera/scene change
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual, dtype=float)

        if len(expected) == 0 or len(actual) == 0:
            return float("nan")

        quantiles = np.linspace(0, 100, buckets + 1)
        bin_edges = np.unique(np.percentile(expected, quantiles))

        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        return float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# IoU — Intersection over Union
# ============================================================

def iou(box_a, box_b) -> float:
    """boxes as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def duplicate_box_rate(boxes: list, iou_threshold: float = 0.7) -> float:
    """
    Fraction of detections that overlap another detection above
    `iou_threshold` — a proxy for insufficient NMS / IoU-threshold
    tuning producing redundant boxes on the same object.
    """
    if len(boxes) < 2:
        return 0.0

    n_dupes, n_pairs = 0, 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            n_pairs += 1
            if iou(boxes[i], boxes[j]) >= iou_threshold:
                n_dupes += 1

    return float(n_dupes / n_pairs) if n_pairs else 0.0


# ============================================================
# PREDICTION <-> GROUND-TRUTH MATCHING (real, IoU-based)
# ============================================================
# Standard greedy IoU-matching used to turn raw detections + real
# coco128 ground-truth boxes into per-detection correct/incorrect
# labels — the basis for genuine confidence calibration and cost
# evaluation (as opposed to a class-presence-only proxy).

def match_predictions_to_ground_truth(
    predictions: list,    # [{"class_id", "confidence", "bbox_xyxy"}]
    ground_truth: list,   # [{"class_id", "bbox_xyxy"}]
    iou_threshold: float = 0.5,
) -> list:
    """
    Greedy matching: predictions sorted by confidence descending,
    each matched to the highest-IoU unmatched ground-truth box of
    the same class. Returns predictions annotated with `is_correct`
    (bool) and `matched_iou` (float).
    """
    preds_sorted = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
    gt_matched = [False] * len(ground_truth)
    results = []

    for pred in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(ground_truth):
            if gt_matched[i] or gt["class_id"] != pred["class_id"]:
                continue
            score = iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if score > best_iou:
                best_iou, best_idx = score, i

        is_correct = best_iou >= iou_threshold
        if is_correct:
            gt_matched[best_idx] = True

        results.append({**pred, "is_correct": is_correct, "matched_iou": round(best_iou, 4)})

    return results


# ============================================================
# LATENCY STATISTICS
# ============================================================

def latency_stats(latencies_ms: list) -> dict:
    arr = np.asarray(latencies_ms, dtype=float)
    if len(arr) == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "fps": 0.0}

    mean_ms = float(arr.mean())
    return {
        "mean_ms": round(mean_ms, 2),
        "p50_ms":  round(float(np.percentile(arr, 50)), 2),
        "p95_ms":  round(float(np.percentile(arr, 95)), 2),
        "max_ms":  round(float(arr.max()), 2),
        "fps":     round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


# ============================================================
# DETECTION SUMMARY STATS
# ============================================================

def detection_rate(objects_per_image: list) -> float:
    """Fraction of images with at least one detection."""
    arr = np.asarray(objects_per_image)
    return float((arr > 0).mean()) if len(arr) else 0.0


def class_coverage(all_detected_classes: set, domain_classes: set) -> float:
    """Fraction of domain-relevant classes seen at least once."""
    if not domain_classes:
        return 0.0
    return float(len(all_detected_classes & domain_classes) / len(domain_classes))


def simple_drift_report(ref_values: pd.Series, new_values: pd.Series) -> float:
    """Simple mean-shift magnitude, relative to reference std."""
    ref_std = ref_values.std() or 1.0
    return float(abs(ref_values.mean() - new_values.mean()) / ref_std)
