# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM — Object Detection ML System
# ============================================================
#
# CHALLENGER MODEL SYSTEM:
#   A newly tuned model variant is never blindly swapped into
#   production. It runs a formal Champion vs Challenger comparison:
#
#   Champion   = current production config (detection_models/latest_model.json)
#   Challenger = newly benchmarked variant + threshold config
#                (from src/model_tuning.py — real ultralytics val()
#                metrics against the auto-downloaded coco128 dataset)
#
#   Promotion logic — challenger promoted ONLY if it beats champion
#   on ALL 3 gates:
#     1. mAP50-95 improvement  >= MIN_MAP_IMPROVEMENT_PP
#     2. Precision              >= MIN_PRECISION_FLOOR
#     3. Mean latency (ms)      <= MAX_LATENCY_MS
#
#   If challenger loses -> champion stays, challenger archived.
#   Full comparison history saved to detection_models/challenger_log.json
# ============================================================

import os
import json
import logging
import time

from ultralytics import YOLO

from src.config import (
    MODEL_DIR, MIN_PRECISION_FLOOR, MAX_LATENCY_MS, EFFECTIVE_LATENCY_CEILING_MS,
)

logger = logging.getLogger(__name__)

CHALLENGER_LOG          = os.path.join(MODEL_DIR, "challenger_log.json")
MIN_MAP_IMPROVEMENT_PP  = 0.005   # smallest meaningful mAP50-95 gain (0.5pp) to justify a swap


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model():
    """
    Reads detection_models/latest_model.json -> loads the YOLO
    weights + active (conf, iou) thresholds.
    Returns: (model, conf_threshold, iou_threshold)
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. Run train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    weights = registry["weights"]
    model   = YOLO(weights)

    conf = float(registry.get("conf_threshold", 0.25))
    iou  = float(registry.get("iou_threshold", 0.45))

    logger.info(
        "Champion model loaded: %s (%s)  |  conf=%.2f  iou=%.2f",
        registry.get("variant", "unknown"), weights, conf, iou
    )

    return model, conf, iou


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================

def _load_champion_metrics() -> dict:
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    # model_card_path is stored directly in the registry — no
    # brittle filename string parsing needed downstream.
    card_path = registry.get("model_card_path", "")

    if not card_path or not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", {})

    return {
        "variant":         card.get("model_name", "unknown"),
        "map50_95":        float(metrics.get("map50_95", 0)),
        "precision":       float(metrics.get("precision", 0)),
        "mean_latency_ms": float(metrics.get("mean_latency_ms", 9999)),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC
# ============================================================

def run_challenger_comparison(
    challenger_variant:     str,
    challenger_weights:     str,
    challenger_map50_95:    float,
    challenger_precision:   float,
    challenger_latency_ms:  float,
    challenger_conf:        float,
    challenger_iou:         float,
    challenger_card_path:   str = "",
) -> dict:
    """
    Compares challenger vs current champion across 3 gates, using
    real ultralytics mAP/precision/latency (see src/model_tuning.py).
    Returns result dict with decision: 'PROMOTED' or 'REJECTED'.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    champion = _load_champion_metrics()

    # ── STALE CHAMPION CHECK ─────────────────────────────────────
    # A registered champion is only trustworthy if it *still* clears
    # today's hard gates. Gate thresholds can tighten over time (e.g.
    # the latency safety margin added after observing 40%+ run-to-run
    # variance on real hardware) — without this check, a model
    # promoted under an older, looser rule would sit as champion
    # forever, since the mAP-improvement comparison below only ever
    # asks "is the challenger better than the champion", never
    # "does the champion even qualify anymore". A stale/non-compliant
    # champion is treated exactly like "no champion" — any compliant
    # challenger replaces it regardless of mAP.
    if champion:
        champion_still_valid = (
            champion.get("precision", 0.0) >= MIN_PRECISION_FLOOR
            and champion.get("mean_latency_ms", float("inf")) <= EFFECTIVE_LATENCY_CEILING_MS
        )
        if not champion_still_valid:
            logger.warning(
                "!! Registered champion '%s' (precision=%.3f, latency=%.1fms) no "
                "longer meets current gates (precision>=%.2f, latency<=%.0fms "
                "effective ceiling) — treating as STALE. Any compliant challenger "
                "will replace it regardless of mAP comparison.",
                champion.get("variant", "unknown"), champion.get("precision", 0.0),
                champion.get("mean_latency_ms", float("inf")),
                MIN_PRECISION_FLOOR, EFFECTIVE_LATENCY_CEILING_MS,
            )
            champion = {}  # fall through to the "no champion" path below

    # ── If no champion exists (or it's stale) -> still enforce ─────
    # HARD gates. mAP-improvement is relative (nothing to compare
    # against for a first/replacement model, so that gate is skipped)
    # — but precision floor and latency ceiling are absolute product
    # requirements ("real-time" is a hard SLA), and apply even here.
    # Latency uses the safety-margined effective ceiling, not the raw
    # one — the same standard training_pipeline.py used to select
    # this candidate in the first place.
    if not champion:
        gate2_precision_floor = challenger_precision >= MIN_PRECISION_FLOOR
        gate3_latency_ceiling = challenger_latency_ms <= EFFECTIVE_LATENCY_CEILING_MS

        if gate2_precision_floor and gate3_latency_ceiling:
            logger.info("No valid champion found — challenger meets hard SLA gates, auto-promoted")
            _update_registry(
                challenger_variant, challenger_weights,
                challenger_conf, challenger_iou, challenger_card_path
            )
            decision = "PROMOTED"
            reason = "No valid existing champion — model met precision/latency SLA, auto-promoted"
        else:
            failed = []
            if not gate2_precision_floor:
                failed.append(f"precision {challenger_precision:.4f} < {MIN_PRECISION_FLOOR}")
            if not gate3_latency_ceiling:
                failed.append(f"latency {challenger_latency_ms:.1f}ms > {EFFECTIVE_LATENCY_CEILING_MS:.0f}ms (effective ceiling)")
            decision = "REJECTED"
            reason = "REJECTED — fails hard SLA gate(s): " + " | ".join(failed)
            logger.warning(reason)

        result = {
            "decision":              decision,
            "reason":                reason,
            "challenger_variant":    challenger_variant,
            "challenger_map50_95":   round(challenger_map50_95, 4),
            "challenger_precision":  round(challenger_precision, 4),
            "challenger_latency_ms": round(challenger_latency_ms, 2),
            "champion_variant":      None,
            "champion_map50_95":     None,
            "gates": {
                "map_improvement_passed":  None,  # not applicable — no prior valid champion
                "precision_floor_passed":  gate2_precision_floor,
                "latency_ceiling_passed":  gate3_latency_ceiling,
            },
            "evaluated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(result)
        return result

    champion_map50_95 = champion.get("map50_95", 0.0)
    champion_variant   = champion.get("variant", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-10s mAP50-95=%.4f", champion_variant, champion_map50_95)
    logger.info("  Challenger: %-10s mAP50-95=%.4f", challenger_variant, challenger_map50_95)
    logger.info("=" * 55)

    # ── Promotion gates ────────────────────────────────────────
    gate1_map_improvement  = (challenger_map50_95 - champion_map50_95) >= MIN_MAP_IMPROVEMENT_PP
    gate2_precision_floor  = challenger_precision >= MIN_PRECISION_FLOOR
    gate3_latency_ceiling  = challenger_latency_ms <= EFFECTIVE_LATENCY_CEILING_MS

    gates_passed = gate1_map_improvement and gate2_precision_floor and gate3_latency_ceiling

    if gates_passed:
        decision = "PROMOTED"
        reason = (
            f"Challenger beats champion: "
            f"mAP50-95 {champion_map50_95:.4f} -> {challenger_map50_95:.4f} "
            f"(+{challenger_map50_95 - champion_map50_95:.4f})"
        )
        logger.info("CHALLENGER PROMOTED -> new champion: %s", challenger_variant)
        _update_registry(
            challenger_variant, challenger_weights,
            challenger_conf, challenger_iou, challenger_card_path
        )
    else:
        decision = "REJECTED"
        failed = []
        if not gate1_map_improvement:
            failed.append(
                f"mAP50-95 improvement {challenger_map50_95 - champion_map50_95:+.4f} "
                f"< {MIN_MAP_IMPROVEMENT_PP}"
            )
        if not gate2_precision_floor:
            failed.append(f"precision {challenger_precision:.4f} < {MIN_PRECISION_FLOOR}")
        if not gate3_latency_ceiling:
            failed.append(f"latency {challenger_latency_ms:.1f}ms > {EFFECTIVE_LATENCY_CEILING_MS:.0f}ms (effective ceiling)")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("CHALLENGER REJECTED — champion '%s' retained", champion_variant)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":               decision,
        "reason":                 reason,
        "evaluated_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_variant":     challenger_variant,
        "challenger_map50_95":    round(challenger_map50_95, 4),
        "challenger_precision":   round(challenger_precision, 4),
        "challenger_latency_ms":  round(challenger_latency_ms, 2),
        "champion_variant":       champion_variant,
        "champion_map50_95":      round(champion_map50_95, 4),
        "gates": {
            "map_improvement_passed":  gate1_map_improvement,
            "precision_floor_passed":  gate2_precision_floor,
            "latency_ceiling_passed":  gate3_latency_ceiling,
        }
    }

    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(variant, weights, conf, iou, model_card_path=None):
    """
    The ONLY function that should ever write latest_model.json, and
    it must only be called after the challenger has genuinely passed
    the promotion gates (or is the first model).
    """
    registry = {
        "variant":         variant,
        "weights":         weights,
        "conf_threshold":  round(float(conf), 4),
        "iou_threshold":   round(float(iou), 4),
        "model_card_path": model_card_path or "",
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated -> %s", registry["variant"])


def _save_challenger_log(result: dict):
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(result)

    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Challenger log saved -> %s", CHALLENGER_LOG)