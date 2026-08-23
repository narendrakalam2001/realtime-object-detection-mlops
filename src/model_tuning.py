# ============================================================
# MODEL TUNING — Real-Time Object Detection ML System
# ============================================================
# Grid-searches confidence + IoU thresholds per pretrained variant
# using ultralytics' own validator (`model.val()`) against the
# auto-downloaded coco128 dataset — every number here is a genuine
# forward pass over real COCO images matched against real
# ground-truth boxes, not a proxy or an estimate.
#
# GPU MEMORY RESILIENCE:
# Sequential model.val() calls across 3 variants x a threshold grid
# (18+ real validation passes in one process) can fragment/exhaust
# VRAM on small GPUs (2-4GB laptop cards), surfacing as a CUDA
# "illegal memory access" crash partway through a run. A single
# hardware/driver hiccup on one config should not kill an otherwise
# successful benchmark run, so:
#   - GPU memory is explicitly released (empty_cache + gc.collect)
#     after every val() call and after every variant
#   - A CUDA failure on one (variant, conf, iou) combo triggers one
#     automatic CPU retry for that combo, so the grid search can
#     still complete instead of the whole pipeline crashing out
#   - If a variant fails entirely, it's excluded (with a loud
#     warning) rather than crashing the rest of the comparison
# ============================================================

import gc
import time
import logging

import pandas as pd
import torch
from ultralytics import YOLO

from src.config import (
    MODEL_VARIANTS, CONF_GRID, IOU_GRID, DATASET_YAML, COCO_BENCHMARK_METRICS,
)

logger = logging.getLogger(__name__)


def _release_gpu_memory():
    """Best-effort GPU memory cleanup between validation runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_val(model: YOLO, conf: float, iou_thr: float, device: str = None) -> dict:
    """
    Runs one real ultralytics val() pass and extracts the metrics
    we care about. box.map / box.map50 / box.mp / box.mr are
    computed by ultralytics itself against coco128's real labels.
    """
    t0 = time.perf_counter()
    metrics = model.val(
        data=DATASET_YAML, conf=conf, iou=iou_thr, imgsz=640,
        verbose=False, plots=False, save_json=False,
        device=device,  # None = ultralytics default (GPU if available)
    )
    elapsed_s = time.perf_counter() - t0

    speed = metrics.speed  # ms per image: preprocess/inference/loss/postprocess
    per_image_ms = sum(speed.get(k, 0.0) for k in ("preprocess", "inference", "postprocess"))

    return {
        "map50_95":       round(float(metrics.box.map), 4),
        "map50":          round(float(metrics.box.map50), 4),
        "precision":      round(float(metrics.box.mp), 4),
        "recall":         round(float(metrics.box.mr), 4),
        "mean_latency_ms": round(per_image_ms, 2),
        "fps":            round(1000.0 / per_image_ms, 2) if per_image_ms > 0 else 0.0,
        "val_wall_time_s": round(elapsed_s, 1),
    }


def _run_val_with_recovery(model: YOLO, conf: float, iou_thr: float) -> dict:
    """
    Runs one config with GPU-crash recovery: on CUDA RuntimeError,
    frees GPU memory and retries once on CPU rather than aborting
    the whole grid search. Returns None if both attempts fail.
    """
    try:
        result = _run_val(model, conf, iou_thr)
        _release_gpu_memory()
        return result

    except RuntimeError as e:
        if "CUDA" not in str(e) and "cuda" not in str(e):
            raise  # not a GPU issue — don't silently swallow real bugs

        logger.warning(
            "GPU error on conf=%.2f iou=%.2f (%s) — freeing memory and "
            "retrying this config on CPU...",
            conf, iou_thr, str(e).splitlines()[0]
        )
        _release_gpu_memory()

        try:
            result = _run_val(model, conf, iou_thr, device="cpu")
            logger.info("CPU retry succeeded for conf=%.2f iou=%.2f", conf, iou_thr)
            return result
        except Exception as retry_err:
            logger.error(
                "CPU retry also failed for conf=%.2f iou=%.2f: %s — skipping this config",
                conf, iou_thr, retry_err
            )
            return None


# ============================================================
# GRID SEARCH — one variant
# ============================================================

def tune_variant(variant_name: str, weights_path: str) -> pd.DataFrame:
    """
    Grid-searches CONF_GRID x IOU_GRID for one model variant against
    the real coco128 val split. Returns one row per successfully
    evaluated (conf, iou) combo — a config that fails on both GPU
    and CPU is logged and skipped rather than crashing the run.
    """
    model = YOLO(weights_path)
    rows = []

    for conf in CONF_GRID:
        for iou_thr in IOU_GRID:
            result = _run_val_with_recovery(model, conf, iou_thr)

            if result is None:
                continue  # both GPU and CPU attempts failed — skip, don't crash

            rows.append({"variant": variant_name, "conf": conf, "iou": iou_thr, **result})
            logger.info(
                "Tuned %-8s conf=%.2f iou=%.2f -> mAP50-95=%.4f  P=%.3f  R=%.3f  lat=%.1fms  (%.1fs)",
                variant_name, conf, iou_thr,
                result["map50_95"], result["precision"], result["recall"],
                result["mean_latency_ms"], result["val_wall_time_s"],
            )

    # Release this variant's weights from GPU memory before the next
    # variant loads — the single biggest lever against VRAM
    # fragmentation across a long multi-variant benchmark run.
    del model
    _release_gpu_memory()

    return pd.DataFrame(rows)


def select_best_config(results_df: pd.DataFrame) -> pd.Series:
    """
    Picks the (conf, iou) row maximizing mAP50-95, breaking ties by
    lower latency (favors real-time deployability).
    """
    return results_df.sort_values(
        ["map50_95", "mean_latency_ms"], ascending=[False, True]
    ).iloc[0]


# ============================================================
# FULL VARIANT COMPARISON — across model sizes
# ============================================================

def compare_all_variants() -> pd.DataFrame:
    """
    Tunes every variant in MODEL_VARIANTS on the real coco128 split
    and returns one summary row per variant with its best threshold
    config and real measured metrics, alongside the officially
    published full-COCO-val2017 benchmark for context (coco128 is a
    128-image subset, so its own mAP will differ from the published
    5,000-image val2017 number — both are reported so the gap is
    visible rather than hidden).

    A variant that fails on every config (e.g. persistent GPU/driver
    issue) is excluded from the summary with a loud warning rather
    than crashing the whole comparison — champion selection then
    proceeds with whichever variants did evaluate successfully.
    """
    summary_rows = []
    failed_variants = []

    for variant_name, weights_path in MODEL_VARIANTS.items():
        try:
            results_df = tune_variant(variant_name, weights_path)
        except Exception as e:
            logger.error(
                "Variant %s failed to load/evaluate entirely (%s) — "
                "excluding it from this comparison run.",
                variant_name, e
            )
            failed_variants.append(variant_name)
            _release_gpu_memory()
            continue

        if results_df.empty:
            logger.warning(
                "Variant %s produced zero successful configs — excluding it "
                "from this comparison run.",
                variant_name
            )
            failed_variants.append(variant_name)
            continue

        best = select_best_config(results_df)
        bench = COCO_BENCHMARK_METRICS.get(variant_name, {})

        summary_rows.append({
            "variant":              variant_name,
            "weights":              weights_path,
            "best_conf":            best["conf"],
            "best_iou":             best["iou"],
            "coco128_map50_95":     best["map50_95"],
            "coco128_map50":        best["map50"],
            "coco128_precision":    best["precision"],
            "coco128_recall":       best["recall"],
            "mean_latency_ms":      best["mean_latency_ms"],
            "fps":                  best["fps"],
            "published_map50_95":  bench.get("map50_95"),
            "published_map50":     bench.get("map50"),
            "params_m":            bench.get("params_m"),
        })

    if failed_variants:
        logger.warning(
            "!! %d variant(s) could not be evaluated and were excluded: %s. "
            "Champion selection below proceeds using only the variants that "
            "did evaluate successfully — re-run later to include the rest.",
            len(failed_variants), failed_variants
        )

    if not summary_rows:
        raise RuntimeError(
            "Every model variant failed to evaluate — nothing to compare. "
            "Check GPU/driver health or set CUDA_VISIBLE_DEVICES= to force CPU."
        )

    return pd.DataFrame(summary_rows)