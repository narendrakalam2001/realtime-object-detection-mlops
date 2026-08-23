# ============================================================
# TRAINING PIPELINE — Real-Time Object Detection ML System
# ============================================================
# All model variants are pretrained on COCO by Ultralytics — this
# pipeline does not run backprop. The dataset (coco128 — a real
# 128-image COCO subset with real YOLO-format ground truth) is
# AUTO-DOWNLOADED on first run, nothing to place manually.
#
#   1. Auto-download / resolve coco128
#   2. Leakage / near-duplicate check on the val split
#   3. Grid-search conf/iou thresholds per model variant using
#      ultralytics' own validator (real mAP/precision/recall)
#   4. Full evaluation of the best config: cost-sensitive eval,
#      isotonic calibration (real IoU-matched detections), alert
#      decisions
#   5. Select overall best variant
#   6. Build + save model card
#   7. Champion-Challenger 3-gate comparison + registry update
#   8. Persist monitoring reference stats (for PSI drift baseline)
#   9. Refresh demo images (for API / dashboard / simulator)
# ============================================================

import os
import logging

import pandas as pd
from ultralytics import YOLO

from src.config import MODEL_DIR
from src.data_loader import ensure_dataset, load_dataset_split, ensure_demo_images
from src.leakage_check import detect_frame_leakage
from src.model_tuning import compare_all_variants, MODEL_VARIANTS
from src.evaluation import evaluate_variant
from src.model_card import build_model_card, save_model_card
from src.model_loader import run_challenger_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_training(domain: str = "security_intrusion", version: str = "v1"):
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("REAL-TIME OBJECT DETECTION — BENCHMARK & REGISTRATION PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: auto-download / resolve dataset ─────────────────
    ensure_dataset()
    val_records = load_dataset_split("train")  # coco128 uses one split for both
    logger.info("coco128 val split: %d images, %d ground-truth instances",
                len(val_records), sum(len(r["annotations"]) for r in val_records))

    # ── Step 2: leakage / near-duplicate check ─────────────────
    image_paths = [r["image_path"] for r in val_records]
    leakage_warnings = detect_frame_leakage(image_paths)
    if leakage_warnings:
        logger.warning("Leakage check flagged %d issue(s) — see logs above", len(leakage_warnings))
    else:
        logger.info("Leakage check passed — no near-duplicate frames")

    # ── Step 3: grid-search thresholds across all variants ─────
    # NOTE: this runs a real ultralytics val() pass per (variant, conf, iou)
    # combo — can take 15-25+ minutes on CPU. See src/config.py CONF_GRID/IOU_GRID.
    summary_df = compare_all_variants()
    summary_path = os.path.join(MODEL_DIR, "model_experiment_results.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("Variant comparison saved -> %s", summary_path)
    logger.info("\n%s", summary_df.to_string(index=False))

    # ── Step 4: select best variant — REAL-TIME SLA IS A HARD GATE ──
    # A "real-time" detection system has a non-negotiable latency
    # ceiling: a model that scores higher mAP but blows the latency
    # budget is not a valid candidate, no matter how accurate. This
    # filter is not softened for the first model either — precision
    # floor and latency ceiling are absolute product requirements,
    # not comparisons against a prior champion.
    #
    # SELECTION uses EFFECTIVE_LATENCY_CEILING_MS (MAX_LATENCY_MS with
    # a safety margin applied), not the raw ceiling. Real hardware
    # testing showed the *same* model + config can vary 40%+ in
    # measured latency between benchmark sessions (thermal throttling,
    # laptop power plan, driver state) — a candidate that only barely
    # clears the true ceiling in one benchmark run is not reliably
    # real-time in production. The true MAX_LATENCY_MS remains the
    # hard gate enforced separately at promotion time in
    # model_loader.py; this margin only affects which candidate gets
    # proposed as challenger in the first place.
    from src.config import (
        MIN_PRECISION_FLOOR, MAX_LATENCY_MS, EFFECTIVE_LATENCY_CEILING_MS,
    )

    sla_compliant = summary_df[
        (summary_df["coco128_precision"] >= MIN_PRECISION_FLOOR)
        & (summary_df["mean_latency_ms"] <= EFFECTIVE_LATENCY_CEILING_MS)
    ]

    if len(sla_compliant):
        best_row = sla_compliant.sort_values("coco128_map50_95", ascending=False).iloc[0]
    else:
        # No variant/threshold combo meets the real-time SLA (with
        # safety margin) at all — this must be loud, not silently
        # swallowed. Fall back to the best latency (most deployable)
        # rather than best accuracy, and flag it clearly so a human
        # has to consciously accept a non-compliant model rather than
        # one slipping through.
        logger.warning(
            "!! NO variant/threshold combo meets the real-time SLA with "
            "safety margin (precision >= %.2f AND latency <= %.0fms, "
            "%.0f%% of the true %.0fms ceiling). Falling back to the "
            "lowest-latency candidate — review before trusting this as "
            "a production champion.",
            MIN_PRECISION_FLOOR, EFFECTIVE_LATENCY_CEILING_MS,
            EFFECTIVE_LATENCY_CEILING_MS / MAX_LATENCY_MS * 100, MAX_LATENCY_MS,
        )
        best_row = summary_df.sort_values("mean_latency_ms", ascending=True).iloc[0]

    logger.info(
        "SLA-compliant candidates (with safety margin): %d / %d  "
        "(precision>=%.2f, latency<=%.0fms effective / %.0fms true ceiling)",
        len(sla_compliant), len(summary_df),
        MIN_PRECISION_FLOOR, EFFECTIVE_LATENCY_CEILING_MS, MAX_LATENCY_MS,
    )

    variant_name = best_row["variant"]
    weights_path = best_row["weights"]
    best_conf    = float(best_row["best_conf"])
    best_iou     = float(best_row["best_iou"])

    logger.info(
        "Selected best variant: %s  |  conf=%.2f  iou=%.2f  mAP50-95=%.4f  latency=%.1fms",
        variant_name, best_conf, best_iou, best_row["coco128_map50_95"], best_row["mean_latency_ms"]
    )

    # ── Step 5: full evaluation of the selected variant ─────────
    model = YOLO(weights_path)
    ultralytics_metrics = {
        "map50_95":        float(best_row["coco128_map50_95"]),
        "map50":           float(best_row["coco128_map50"]),
        "precision":       float(best_row["coco128_precision"]),
        "recall":          float(best_row["coco128_recall"]),
        "mean_latency_ms": float(best_row["mean_latency_ms"]),
        "fps":             float(best_row["fps"]),
    }
    eval_result = evaluate_variant(
        variant_name, model, val_records, best_conf, best_iou,
        ultralytics_metrics, domain=domain,
    )

    # ── Step 6: build + save model card ──────────────────────────
    from src.config import COCO_BENCHMARK_METRICS
    card = build_model_card(
        variant=variant_name,
        weights=weights_path,
        conf_threshold=best_conf,
        iou_threshold=best_iou,
        metrics=eval_result["metrics"],
        coco_benchmark=COCO_BENCHMARK_METRICS.get(variant_name, {}),
        cost_result=eval_result["cost_result"],
        decision_counts=eval_result["decision_counts"],
        val_set_size=len(val_records),
        calibration_status=eval_result["calibration_status"],
        n_calibration_points=eval_result["n_calibration_points"],
        version=version,
    )
    card_path = save_model_card(card, MODEL_DIR, variant_name, version)

    # ── Step 7: champion-challenger comparison + registry ───────
    challenger_result = run_challenger_comparison(
        challenger_variant=variant_name,
        challenger_weights=weights_path,
        challenger_map50_95=eval_result["metrics"]["map50_95"],
        challenger_precision=eval_result["metrics"]["precision"],
        challenger_latency_ms=eval_result["metrics"]["mean_latency_ms"],
        challenger_conf=best_conf,
        challenger_iou=best_iou,
        challenger_card_path=card_path,
    )
    logger.info("Champion-Challenger decision: %s", challenger_result["decision"])

    # ── Step 8: persist monitoring reference stats (PSI baseline) ─
    monitor_df = pd.DataFrame({"confidence": eval_result["confidences"]})
    monitor_path = os.path.join(MODEL_DIR, "reference_confidence_scores.csv")
    monitor_df.to_csv(monitor_path, index=False)

    counts_df = pd.DataFrame({"objects_per_frame": eval_result["counts_per_frame"]})
    counts_path = os.path.join(MODEL_DIR, "reference_object_counts.csv")
    counts_df.to_csv(counts_path, index=False)

    logger.info("Reference monitoring stats saved -> %s , %s", monitor_path, counts_path)

    # ── Step 9: refresh demo images for API/dashboard/simulator ───
    demo_paths = ensure_demo_images()
    logger.info("Demo images ready -> %d files", len(demo_paths))

    # ── Report the ACTUAL registered champion, not just this run's ──
    # candidate. If the challenger was REJECTED, the true production
    # champion is whichever model was already registered (visible in
    # challenger_result["champion_variant"]) — reporting variant_name
    # here regardless of the decision previously made this log claim
    # a promotion had happened when it hadn't.
    if challenger_result["decision"] == "PROMOTED":
        actual_champion = challenger_result["challenger_variant"]
    else:
        actual_champion = challenger_result.get("champion_variant") or "NONE (no valid champion registered)"

    logger.info("=" * 60)
    logger.info(
        "PIPELINE COMPLETE — this run's candidate: %s (%s)  |  ACTUAL registered champion: %s",
        variant_name, challenger_result["decision"], actual_champion,
    )
    logger.info("=" * 60)

    return {
        "variant":            variant_name,       # this run's evaluated/selected candidate
        "actual_champion":    actual_champion,     # what's really registered in latest_model.json
        "card_path":          card_path,
        "challenger_result":  challenger_result,
        "summary_df":         summary_df,
    }


if __name__ == "__main__":
    run_training()