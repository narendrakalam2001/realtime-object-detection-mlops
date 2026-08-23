# ============================================================
# DETECTION ENGINE — Real-Time Object Detection ML System
# ============================================================
# Fintech-style 3-tier decision system, applied to detections:
#   NORMAL   -> no action, frame logged only
#   ALERT    -> flagged for review (shelf low-stock, mild crowding)
#   CRITICAL -> immediate notification (intrusion, safety breach)
#
# Decision hierarchy:
#   1. Hard domain rules (person count, zone, low confidence ratio)
#   2. Falls through to NORMAL if nothing triggers
# ============================================================

import logging

from src.config import (
    INTRUSION_PERSON_THRESHOLD, INTRUSION_CRITICAL_COUNT,
    SHELF_EMPTY_MAX_ITEMS, WAREHOUSE_CROWD_THRESHOLD,
    LOW_CONFIDENCE_ALERT_RATIO,
)

logger = logging.getLogger(__name__)


# ============================================================
# ALERT SCORING — single frame
# ============================================================

def score_frame(detections: list, domain: str) -> dict:
    """
    detections: list of dicts {"class_name": str, "confidence": float}
    domain: one of "retail_shelf" | "warehouse_safety" | "security_intrusion"

    Returns structured decision output for a single frame.
    """
    class_names = [d["class_name"] for d in detections]
    person_count = class_names.count("person")

    confidences = [d["confidence"] for d in detections]
    low_conf_ratio = (
        sum(1 for c in confidences if c < 0.4) / len(confidences)
        if confidences else 0.0
    )

    decision       = "NORMAL"
    rule_triggered = None

    if domain == "security_intrusion":
        if person_count >= INTRUSION_CRITICAL_COUNT:
            decision, rule_triggered = "CRITICAL", "INTRUSION_MULTIPLE_PERSONS"
        elif person_count >= INTRUSION_PERSON_THRESHOLD:
            decision, rule_triggered = "ALERT", "INTRUSION_PERSON_DETECTED"

    elif domain == "warehouse_safety":
        if person_count >= WAREHOUSE_CROWD_THRESHOLD:
            decision, rule_triggered = "CRITICAL", "WAREHOUSE_CROWDING"
        elif person_count >= 1 and "truck" in class_names:
            decision, rule_triggered = "ALERT", "PERSON_NEAR_VEHICLE"

    elif domain == "retail_shelf":
        retail_items = [
            c for c in class_names
            if c in {"bottle", "wine glass", "cup", "bowl", "banana",
                      "apple", "sandwich", "orange", "book"}
        ]
        if len(retail_items) <= SHELF_EMPTY_MAX_ITEMS:
            decision, rule_triggered = "ALERT", "SHELF_LOW_STOCK"

    # ── Secondary rule: unreliable frame (independent of domain) ──
    if decision == "NORMAL" and low_conf_ratio >= LOW_CONFIDENCE_ALERT_RATIO and confidences:
        decision, rule_triggered = "ALERT", "LOW_CONFIDENCE_FRAME"

    return {
        "domain":          domain,
        "decision":        decision,
        "rule_triggered":  rule_triggered,
        "person_count":    person_count,
        "object_count":    len(detections),
        "low_conf_ratio":  round(low_conf_ratio, 4),
    }


# ============================================================
# BATCH DECISION COUNTS — for model card / dashboard KPIs
# ============================================================

def summarize_decisions(decisions: list) -> dict:
    counts = {"NORMAL": 0, "ALERT": 0, "CRITICAL": 0}
    for d in decisions:
        counts[d.get("decision", "NORMAL")] = counts.get(d.get("decision", "NORMAL"), 0) + 1
    return counts
