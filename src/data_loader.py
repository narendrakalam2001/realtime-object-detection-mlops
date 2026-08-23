# ============================================================
# DATA LOADER — Real-Time Object Detection ML System
# ============================================================
# Dataset is AUTO-DOWNLOADED — nothing to download or place
# manually. `ensure_dataset()` triggers ultralytics' built-in
# downloader for coco128 (official Ultralytics sanity-check subset
# of real COCO train2017 images + real YOLO-format ground-truth
# labels, ~7MB) on first use.
# ============================================================

import os
import glob
import shutil
import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import DATASET_YAML, DATASET_DIR, DATA_DIR, DEMO_IMAGE_COUNT, MIN_IMAGE_RESOLUTION

logger = logging.getLogger(__name__)

# ── Full COCO 80-class label map (index -> name) ──────────────────
# Matches coco128.yaml / every pretrained YOLOv8 checkpoint 1:1.
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


# ============================================================
# DATASET AUTO-DOWNLOAD
# ============================================================

def ensure_dataset(dataset_yaml: str = DATASET_YAML) -> dict:
    """
    Triggers ultralytics' built-in dataset downloader. If
    `datasets/coco128/` isn't present locally it is fetched and
    unzipped automatically (~7MB); if it's already present this is
    a no-op path check. Returns the resolved dataset dict:
        {'path', 'train', 'val', 'names', 'nc', ...}
    """
    from ultralytics import settings as ul_settings
    from ultralytics.data.utils import check_det_dataset

    # Pin ultralytics' dataset location to a project-relative folder
    # instead of relying on its global (machine-wide) settings cache
    # — keeps the auto-download fully self-contained per project.
    project_datasets_dir = os.path.abspath(DATASET_DIR)
    if ul_settings.get("datasets_dir") != project_datasets_dir:
        ul_settings.update({"datasets_dir": project_datasets_dir})

    logger.info("Resolving dataset '%s' (auto-downloads if missing)...", dataset_yaml)
    data = check_det_dataset(dataset_yaml)
    logger.info(
        "Dataset ready  |  path=%s  classes=%d  train=%s",
        data["path"], data["nc"], data.get("train")
    )
    return data


def _split_image_dir(data: dict, split: str = "train") -> str:
    """
    coco128 only ships a train2017 split (used for both train and
    val in coco128.yaml — it's a sanity/demo subset, not a proper
    held-out benchmark). Resolves whichever key is available.
    """
    key = split if data.get(split) else "train"
    img_dir = str(data[key])
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Resolved image dir does not exist: {img_dir}")
    return img_dir


def _label_path_for_image(img_path: str) -> str:
    """
    coco128 mirrors images/<split>/x.jpg -> labels/<split>/x.txt

    Uses pathlib so this works correctly on Windows (backslash paths)
    as well as Linux/Mac (forward-slash paths) — a plain string
    .replace("/images/", "/labels/") silently fails on Windows and
    returns zero ground-truth annotations for every image.
    """
    p = Path(img_path)
    parts = list(p.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index("images")  # last occurrence of 'images'
        parts[idx] = "labels"
    except ValueError:
        pass  # no 'images' segment found — return best-effort path below
    return str(Path(*parts).with_suffix(".txt"))


# ============================================================
# ANNOTATION PARSING (real YOLO-format ground truth)
# ============================================================

def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> list:
    """
    Parses a YOLO-format label file (class_id x_center y_center w h,
    all normalized 0-1) into a list of dicts with both normalized
    and pixel-space (xyxy) boxes. Returns [] if the file is missing
    (a legitimate "background" image with no objects in coco128).
    """
    if not os.path.exists(label_path):
        return []

    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, w, h = parts
            cls_id = int(cls_id)
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)

            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h

            annotations.append({
                "class_id":   cls_id,
                "class_name": COCO_CLASSES.get(cls_id, f"class_{cls_id}"),
                "bbox_norm":  [xc, yc, w, h],
                "bbox_xyxy":  [x1, y1, x2, y2],
                "area_norm":  round(w * h, 6),
            })

    return annotations


def load_dataset_split(split: str = "train", limit: int = None) -> list:
    """
    Loads the auto-downloaded coco128 split as a list of dicts:
        {"image_path", "width", "height", "annotations": [...]}
    with real parsed ground-truth boxes for every image.
    """
    data = ensure_dataset()
    img_dir = _split_image_dir(data, split)

    image_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg"))
        + glob.glob(os.path.join(img_dir, "*.jpeg"))
        + glob.glob(os.path.join(img_dir, "*.png"))
    )
    if limit:
        image_paths = image_paths[:limit]

    records = []
    for img_path in image_paths:
        with Image.open(img_path) as im:
            w, h = im.size

        label_path = _label_path_for_image(img_path)
        annotations = parse_yolo_label(label_path, w, h)

        records.append({
            "image_path":  img_path,
            "width":       w,
            "height":      h,
            "annotations": annotations,
        })

    logger.info("Loaded %d images from coco128 split '%s'", len(records), split)
    return records


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(path: str) -> Image.Image:
    """Opens and validates an image file before it reaches the model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path)
        img.load()
    except UnidentifiedImageError:
        raise ValueError(f"File is not a valid image: {path}")
    except Exception as e:
        raise ValueError(f"Could not read image '{path}': {e}")

    img = img.convert("RGB")

    w, h = img.size
    if w < MIN_IMAGE_RESOLUTION or h < MIN_IMAGE_RESOLUTION:
        raise ValueError(
            f"Image resolution {w}x{h} below minimum "
            f"{MIN_IMAGE_RESOLUTION}x{MIN_IMAGE_RESOLUTION}"
        )

    logger.info("Image validated  |  path=%s  size=%dx%d", path, w, h)
    return img


def image_to_array(img: Image.Image) -> np.ndarray:
    """Converts a validated PIL image to an RGB numpy array."""
    return np.array(img.convert("RGB"))


# ============================================================
# DEMO IMAGES — auto-populated from coco128 (not a separate dataset)
# ============================================================

def ensure_demo_images(n: int = DEMO_IMAGE_COUNT, demo_dir: str = DATA_DIR) -> list:
    """
    Copies a handful of real coco128 images into data/demo_images/
    so the API / dashboard / simulator have something to demo
    against without a manual upload. These are drawn straight from
    the auto-downloaded dataset — not a separately curated set.
    """
    os.makedirs(demo_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(demo_dir, "*.jpg")))
    if len(existing) >= n:
        return existing

    split = load_dataset_split("train", limit=None)
    # Prefer images that actually contain objects for a meaningful demo
    with_objects = [r for r in split if r["annotations"]]
    chosen = with_objects[:n] if len(with_objects) >= n else split[:n]

    copied = []
    for i, record in enumerate(chosen):
        dest = os.path.join(demo_dir, f"demo_{i:02d}.jpg")
        shutil.copy(record["image_path"], dest)
        copied.append(dest)

    logger.info("Demo images ready -> %s (%d files)", demo_dir, len(copied))
    return copied


def load_sample_dataset(data_dir: str = DATA_DIR) -> list:
    """Back-compat helper: returns demo image paths, auto-populating if needed."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    if not paths:
        paths = ensure_demo_images()
    return paths


# ============================================================
# DOMAIN CLASS FILTERING
# ============================================================

def get_domain_class_ids(domain: str) -> set:
    from src.config import DOMAIN_CLASS_SUBSETS
    if domain not in DOMAIN_CLASS_SUBSETS:
        raise ValueError(
            f"Unknown domain '{domain}'. Valid: {list(DOMAIN_CLASS_SUBSETS)}"
        )
    return set(DOMAIN_CLASS_SUBSETS[domain].values())