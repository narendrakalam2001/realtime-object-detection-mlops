# ============================================================
# CONFIGURATION — Real-Time Object Detection ML System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE = 42
N_JOBS       = -1

# ── Model variants (pretrained on COCO, no training required) ─
# All weights are pulled automatically from the official
# Ultralytics release assets on first use — no manual COCO
# dataset download needed for inference.
MODEL_VARIANTS = {
    "YOLOv8n": "yolov8n.pt",   # nano   — fastest, edge / real-time streams
    "YOLOv8s": "yolov8s.pt",   # small  — balanced accuracy/speed
    "YOLOv8m": "yolov8m.pt",   # medium — highest accuracy, batch/offline
}

DEFAULT_VARIANT = "YOLOv8n"

# ── Benchmark dataset — AUTO-DOWNLOADED, no manual download needed ─
# coco128 is the official Ultralytics sanity-check subset of COCO
# train2017 — 128 real COCO images with real YOLO-format ground-truth
# labels across the full 80-class taxonomy. `ultralytics` downloads
# and unzips it automatically (~7MB) the first time it's referenced
# by name — nothing to download or place manually.
DATASET_YAML = "coco128.yaml"
DATASET_DIR  = "datasets"

# ── Published Ultralytics COCO val2017 benchmark metrics ──────
# Source: https://docs.ultralytics.com/models/yolov8/#performance-metrics
# (official reported mAP50-95, mAP50, params, and single-image CPU latency)
# Used for offline model-selection scoring since full COCO val2017
# annotations (~1GB) are not downloaded in this environment.
COCO_BENCHMARK_METRICS = {
    "YOLOv8n": {"map50_95": 37.3, "map50": 52.6, "params_m": 3.2,  "flops_g": 8.7},
    "YOLOv8s": {"map50_95": 44.9, "map50": 61.8, "params_m": 11.2, "flops_g": 28.6},
    "YOLOv8m": {"map50_95": 50.2, "map50": 67.2, "params_m": 25.9, "flops_g": 78.9},
}

# ── Inference thresholds — tuning search space ─────────────────
# Kept intentionally small — each (conf, iou) combo runs a full
# ultralytics val() pass over the dataset (real forward passes on
# real images), so grid size directly drives wall-clock runtime.
# On CPU, one val() pass over coco128 takes roughly 20-40s for
# YOLOv8n and longer for YOLOv8s/m — a 3x2 grid x 3 variants is
# already ~15-25 minutes end to end. Widen only if you have a GPU.
CONF_GRID = [0.25, 0.35, 0.45]
IOU_GRID  = [0.45, 0.60]

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD  = 0.45

# ── Dual-resolution inference profiles ─────────────────────────
# Mirrors the "scaled vs unscaled preprocessing" pattern from the
# tabular projects: one profile optimized for real-time throughput,
# one optimized for offline/batch accuracy.
STREAM_IMG_SIZE = 320   # low-latency real-time camera feed
BATCH_IMG_SIZE  = 640   # high-accuracy offline / audit batch

# ── COCO class ids grouped by business domain ──────────────────
# Reliance Retail-style shelf monitoring, Amazon-style warehouse
# safety, and surveillance / intrusion detection each only care
# about a subset of the 80 COCO classes.
DOMAIN_CLASS_SUBSETS = {
    "retail_shelf": {
        "bottle": 39, "wine glass": 40, "cup": 41, "bowl": 45,
        "banana": 46, "apple": 47, "sandwich": 48, "orange": 49,
        "book": 73, "handbag": 26, "backpack": 24, "person": 0,
    },
    "warehouse_safety": {
        "person": 0, "truck": 7, "forklift_proxy_truck": 7,
        "backpack": 24, "suitcase": 28, "bench": 13,
    },
    "security_intrusion": {
        "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
        "backpack": 24, "handbag": 26, "suitcase": 28,
    },
}

# ── Alert / decision engine rules ───────────────────────────────
# Three-tier decision system, same pattern as the credit-risk
# APPROVE / REVIEW / DECLINE engine.
INTRUSION_PERSON_THRESHOLD   = 1      # >=1 person in a restricted zone frame
INTRUSION_CRITICAL_COUNT     = 3      # >=3 persons -> CRITICAL
SHELF_EMPTY_MAX_ITEMS        = 1      # <=1 retail item detected -> restock alert
WAREHOUSE_CROWD_THRESHOLD    = 4      # >=4 persons near equipment -> safety alert
LOW_CONFIDENCE_ALERT_RATIO   = 0.35   # share of detections below this conf -> flag

# ── PSI drift thresholds (detection-count / confidence drift) ──
PSI_MODERATE = 0.10
PSI_HIGH     = 0.20

# ── Score / confidence monitoring alert ─────────────────────────
CONF_MEAN_ALERT = 0.30

# ── Champion-Challenger promotion gates (3-gate system) ─────────
MIN_MAP_IMPROVEMENT   = 0.5     # challenger must beat champion mAP50-95 by >= 0.5 pt
MIN_PRECISION_FLOOR   = 0.55    # challenger precision proxy must clear this floor
MAX_LATENCY_MS        = 120.0   # true real-time SLA — hard ceiling, checked at promotion time

# ── Latency measurement safety margin ────────────────────────────
# Observed on real hardware: the *same* model + config benchmarked
# in two separate sessions on the same laptop GPU produced latency
# readings 44% apart (99.5ms vs 178.9ms for YOLOv8m @ conf=0.25
# iou=0.60) — driven by thermal throttling / power-plan / driver
# state, not by the model itself. A candidate that only clears
# MAX_LATENCY_MS by a small margin in one benchmark run is not
# safely real-time; it can cross the SLA under worse thermal/power
# conditions in production. Candidate *selection* (training_pipeline.py)
# therefore filters against a stricter effective ceiling built from
# this margin, while MAX_LATENCY_MS above remains the true SLA used
# for the final champion-challenger gate itself (model_loader.py).
LATENCY_SAFETY_MARGIN         = 0.70
EFFECTIVE_LATENCY_CEILING_MS  = MAX_LATENCY_MS * LATENCY_SAFETY_MARGIN  # ~84ms

# ── Leakage / data-quality checks ────────────────────────────────
DUPLICATE_HASH_THRESHOLD = 5    # hamming distance for perceptual-hash near-duplicate flag
MIN_IMAGE_RESOLUTION     = 64   # px, minimum width/height accepted

# ── Demo / API sanity-check images ───────────────────────────────
# A handful of images from the auto-downloaded coco128 val split are
# copied here on first run purely so the API/dashboard have something
# to demo without requiring a user upload — NOT a separate manually
# curated dataset. Source of truth for benchmarking is always coco128.
DEMO_IMAGE_COUNT = 6

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR    = "detection_models"
METRICS_LOG  = "detection_models/metrics_log.csv"
TESTS_DIR    = "tests"
SERVING_DIR  = "serving"
DATA_DIR     = "data/demo_images"   # auto-populated from coco128, see data_loader.py

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)