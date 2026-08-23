# ============================================================
# FRAME SIMULATOR — Real-Time Object Detection ML System
# ============================================================
# Simulates a live camera feed hitting the /predict endpoint,
# cycling through the sample image set under three business
# scenarios:
#   retail    — shelf-monitoring camera (Reliance Retail-style)
#   warehouse — dock / floor safety camera (Amazon-warehouse-style)
#   security  — perimeter / entrance camera (Bosch / CCTV-style)
# ============================================================

import os
import sys
import random
import time
import logging

import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import ensure_demo_images  # noqa: E402

logger = logging.getLogger(__name__)

API_URL = os.getenv(
    "OBJECT_DETECTION_API_URL",
    "http://127.0.0.1:8000"
) + "/predict"

SCENARIO_DOMAIN_MAP = {
    "retail":    "retail_shelf",
    "warehouse": "warehouse_safety",
    "security":  "security_intrusion",
}


def _load_frame_pool() -> list:
    """
    Frames are drawn from the auto-downloaded coco128 dataset
    (copied into data/demo_images/ on first use) — nothing to
    place manually before running the simulator.
    """
    paths = ensure_demo_images()
    if not paths:
        raise FileNotFoundError("No demo frames available — run scripts/train_model.py first")
    return paths


def send_frame(path: str, domain: str, idx: int):
    try:
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "image/jpeg")}
            response = requests.post(API_URL, files=files, params={"domain": domain}, timeout=15)

        if response.status_code == 200:
            result = response.json()
            print(
                f"[{idx+1}] {os.path.basename(path):<15} domain={domain:<18} "
                f"objects={result['detection_count']:<3} "
                f"decision={result['decision']:<10} "
                f"latency={result['latency_ms']:.1f}ms"
            )
        else:
            print(f"[{idx+1}] API error: {response.status_code} — {response.text[:200]}")

    except Exception as e:
        print(f"[{idx+1}] Connection error: {e}")


def simulate_stream(n: int = 20, scenario: str = "security"):
    if scenario not in SCENARIO_DOMAIN_MAP:
        raise ValueError(f"Unknown scenario '{scenario}'. Valid: {list(SCENARIO_DOMAIN_MAP)}")

    domain = SCENARIO_DOMAIN_MAP[scenario]
    frames = _load_frame_pool()

    print(f"\nSimulating {n} frames  |  scenario={scenario}  domain={domain}\n" + "-" * 70)

    for i in range(n):
        frame_path = random.choice(frames)
        send_frame(frame_path, domain, i)
        time.sleep(0.5)

    print("-" * 70 + "\nSimulation complete")


if __name__ == "__main__":
    simulate_stream(n=20, scenario="security")
