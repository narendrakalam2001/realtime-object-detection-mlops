# ============================================================
# PREPROCESSING — Real-Time Object Detection ML System
# ============================================================

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.config import STREAM_IMG_SIZE, BATCH_IMG_SIZE

logger = logging.getLogger(__name__)


# ============================================================
# FRAME PREPROCESSOR — letterbox resize transformer
# ============================================================

class FramePreprocessor:
    """
    Letterbox-resizes a frame to a fixed square input size while
    preserving aspect ratio (pads with neutral grey), the same
    resize strategy YOLO models are trained with.

    Plays the same structural role as the `Clipper` transformer in
    the tabular projects: a small, stateless, testable unit that
    sits in front of the model and guarantees consistent inputs
    between training-time benchmarking and live serving.
    """

    def __init__(self, img_size: int = 640, pad_color: tuple = (114, 114, 114)):
        self.img_size  = img_size
        self.pad_color = pad_color

    def transform(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        scale = self.img_size / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.img_size, self.img_size), self.pad_color)
        offset = ((self.img_size - new_w) // 2, (self.img_size - new_h) // 2)
        canvas.paste(resized, offset)

        return canvas

    def get_params(self) -> dict:
        return {"img_size": self.img_size, "pad_color": self.pad_color}


# ============================================================
# DUAL PREPROCESSING PROFILES
# ============================================================
# Mirrors the "scaled ColumnTransformer for linear models / unscaled
# ColumnTransformer for tree models" pattern: one preprocessing
# profile tuned for real-time throughput, one for offline accuracy.

@dataclass
class InferenceProfile:
    name:     str
    img_size: int
    use_case: str


def build_preprocessing_profiles() -> tuple:
    """
    Returns (stream_profile, batch_profile) — independent
    FramePreprocessor + config pairs, never sharing fitted state
    (there is nothing to "fit" here, but the independence mirrors
    the clone(ord_pipeline) guarantee used in the tabular projects).
    """
    stream_profile = InferenceProfile(
        name="stream", img_size=STREAM_IMG_SIZE,
        use_case="real-time camera feed / low-latency alerting"
    )
    batch_profile = InferenceProfile(
        name="batch", img_size=BATCH_IMG_SIZE,
        use_case="offline audit / high-accuracy batch scoring"
    )

    logger.info(
        "Preprocessing profiles built  |  stream=%dpx  batch=%dpx",
        stream_profile.img_size, batch_profile.img_size
    )
    return stream_profile, batch_profile


# ============================================================
# IMAGE QUALITY METRICS — gate before inference
# ============================================================

def compute_image_quality(img: Image.Image) -> dict:
    """
    Cheap, dependency-light quality signals used to flag frames that
    are likely to produce unreliable detections (very dark, blown
    out, or blurry frames from a degraded camera feed).
    """
    arr  = np.asarray(img.convert("L"), dtype=np.float32)
    mean_brightness = float(arr.mean())
    std_contrast    = float(arr.std())

    # Laplacian-variance blur proxy (no scipy/cv2 dependency required)
    gy, gx = np.gradient(arr)
    sharpness = float((gx ** 2 + gy ** 2).mean())

    quality = {
        "mean_brightness": round(mean_brightness, 2),
        "std_contrast":    round(std_contrast, 2),
        "sharpness_proxy": round(sharpness, 2),
        "is_too_dark":     mean_brightness < 25,
        "is_too_bright":   mean_brightness > 230,
        "is_low_contrast": std_contrast < 10,
    }
    return quality


def resize_for_inference(img: Image.Image, profile: InferenceProfile) -> Image.Image:
    """Applies a FramePreprocessor sized to the given inference profile."""
    return FramePreprocessor(img_size=profile.img_size).transform(img)
