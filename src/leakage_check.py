# ============================================================
# LEAKAGE CHECK — Real-Time Object Detection ML System
# ============================================================
# In tabular projects, leakage means a feature encodes the target.
# In a computer-vision pipeline sourced from continuous video, the
# equivalent risk is near-duplicate frames leaking between the
# benchmarking / calibration set and the "held-out" evaluation
# set (e.g. two frames one second apart from the same clip), which
# would silently inflate reported metrics.
#
# Two checks, run before any benchmarking or champion-challenger
# evaluation:
#   1. Exact duplicate  — identical average-hash
#   2. Near duplicate    — average-hash Hamming distance below
#                           DUPLICATE_HASH_THRESHOLD
# ============================================================

import logging
from itertools import combinations

import numpy as np
from PIL import Image

from src.config import DUPLICATE_HASH_THRESHOLD

logger = logging.getLogger(__name__)


def _average_hash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    """Simple perceptual average-hash — no external hashing dependency."""
    small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr   = np.asarray(small, dtype=np.float32)
    return (arr > arr.mean()).flatten()


def _hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))


def detect_frame_leakage(
    image_paths: list,
    threshold_hamming: int = DUPLICATE_HASH_THRESHOLD,
) -> list:
    """
    Runs pairwise near-duplicate detection across a list of image
    paths destined for the benchmarking / evaluation set.

    Returns a list of warning strings. Empty list = no leakage detected.
    """
    warnings = []
    hashes = {}

    for path in image_paths:
        try:
            hashes[path] = _average_hash(Image.open(path))
        except Exception as e:
            logger.warning("Could not hash '%s' for leakage check: %s", path, e)

    for path_a, path_b in combinations(hashes.keys(), 2):
        dist = _hamming_distance(hashes[path_a], hashes[path_b])

        if dist == 0:
            warnings.append(
                f"[LEAKAGE] '{path_a}' and '{path_b}' are exact duplicates "
                "→ remove one before evaluation"
            )
        elif dist <= threshold_hamming:
            warnings.append(
                f"[LEAKAGE] '{path_a}' and '{path_b}' are near-duplicates "
                f"(hamming={dist} <= {threshold_hamming}) → possible frame leakage"
            )

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Leakage check passed — no near-duplicate frames detected")

    return warnings


def check_class_distribution_sanity(class_counts: dict, min_classes: int = 1) -> list:
    """
    Secondary sanity check: flags a benchmark set with too little
    class diversity to produce a meaningful evaluation (e.g. every
    frame containing only 'person' would silently make the system
    look better/worse than it is on other domains).
    """
    warnings = []
    n_classes = len([c for c, n in class_counts.items() if n > 0])

    if n_classes < min_classes:
        warnings.append(
            f"[DATA QUALITY] Only {n_classes} distinct class(es) present in "
            f"benchmark set — evaluation may not generalize."
        )
        logger.warning(warnings[-1])

    return warnings
