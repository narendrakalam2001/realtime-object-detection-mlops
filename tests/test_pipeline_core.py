# ============================================================
# TEST SUITE — Real-Time Object Detection ML System
# ============================================================

import os
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import config
from src.data_loader import (
    validate_image, load_sample_dataset, ensure_demo_images,
    load_dataset_split, parse_yolo_label, get_domain_class_ids,
    image_to_array, COCO_CLASSES,
)
from src.preprocessing import (
    FramePreprocessor, build_preprocessing_profiles,
    compute_image_quality, resize_for_inference,
)
from src.metrics import (
    psi, iou, duplicate_box_rate, latency_stats,
    detection_rate, class_coverage, match_predictions_to_ground_truth,
)
from src.leakage_check import detect_frame_leakage, check_class_distribution_sanity
from src.detection_engine import score_frame, summarize_decisions
from src.model_card import build_model_card, save_model_card, load_model_card
from src.model_loader import run_challenger_comparison, load_latest_model
from src.evaluation import cost_sensitive_evaluation, calibrate_confidence_scores


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def sample_image_path():
    paths = load_sample_dataset()
    return paths[0]


@pytest.fixture(scope="module")
def sample_image(sample_image_path):
    return validate_image(sample_image_path)


@pytest.fixture(scope="module")
def val_records():
    # Auto-downloads coco128 on first call if not already present.
    return load_dataset_split("train", limit=10)


@pytest.fixture()
def tmp_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr("src.model_loader.MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(
        "src.model_loader.CHALLENGER_LOG", os.path.join(str(tmp_path), "challenger_log.json")
    )
    return str(tmp_path)


# ============================================================
# CONFIG
# ============================================================

class TestConfig:
    def test_model_variants_defined(self):
        assert "YOLOv8n" in config.MODEL_VARIANTS
        assert config.DEFAULT_VARIANT in config.MODEL_VARIANTS

    def test_domain_subsets_have_person_where_relevant(self):
        assert "person" in config.DOMAIN_CLASS_SUBSETS["security_intrusion"]

    def test_psi_thresholds_ordered(self):
        assert config.PSI_MODERATE < config.PSI_HIGH

    def test_conf_grid_within_valid_range(self):
        assert all(0.0 < c < 1.0 for c in config.CONF_GRID)

    def test_dataset_yaml_configured(self):
        assert config.DATASET_YAML == "coco128.yaml"


# ============================================================
# DATA LOADER — auto-download + real annotation parsing
# ============================================================

class TestDataLoader:
    def test_dataset_auto_downloads_and_loads(self, val_records):
        assert len(val_records) == 10
        for record in val_records:
            assert os.path.exists(record["image_path"])
            assert record["width"] > 0 and record["height"] > 0

    def test_parse_yolo_label_returns_valid_boxes(self, val_records):
        with_objects = [r for r in val_records if r["annotations"]]
        assert len(with_objects) > 0, "Expected at least one coco128 image with objects"

        ann = with_objects[0]["annotations"][0]
        x1, y1, x2, y2 = ann["bbox_xyxy"]
        assert x2 > x1 and y2 > y1
        assert ann["class_name"] in COCO_CLASSES.values()

    def test_parse_yolo_label_missing_file_returns_empty(self):
        assert parse_yolo_label("does_not_exist.txt", 640, 480) == []

    def test_ensure_demo_images_populates_from_dataset(self):
        paths = ensure_demo_images(n=4)
        assert len(paths) >= 4
        assert all(os.path.exists(p) for p in paths)

    def test_load_sample_dataset_returns_paths(self):
        paths = load_sample_dataset()
        assert len(paths) >= 1
        assert all(os.path.exists(p) for p in paths)

    def test_validate_image_valid_file(self, sample_image_path):
        img = validate_image(sample_image_path)
        assert isinstance(img, Image.Image)

    def test_validate_image_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            validate_image("does_not_exist.jpg")

    def test_validate_image_rejects_tiny_image(self, tmp_path):
        tiny_path = tmp_path / "tiny.jpg"
        Image.new("RGB", (10, 10)).save(tiny_path)
        with pytest.raises(ValueError):
            validate_image(str(tiny_path))

    def test_validate_image_rejects_corrupt_file(self, tmp_path):
        bad_path = tmp_path / "bad.jpg"
        bad_path.write_bytes(b"not an image")
        with pytest.raises(ValueError):
            validate_image(str(bad_path))

    def test_get_domain_class_ids_known_domain(self):
        ids = get_domain_class_ids("security_intrusion")
        assert 0 in ids  # person

    def test_get_domain_class_ids_unknown_domain_raises(self):
        with pytest.raises(ValueError):
            get_domain_class_ids("not_a_real_domain")

    def test_image_to_array_shape(self, sample_image):
        arr = image_to_array(sample_image)
        assert arr.ndim == 3
        assert arr.shape[2] == 3

    def test_coco_classes_has_80_entries(self):
        assert len(COCO_CLASSES) == 80
        assert COCO_CLASSES[0] == "person"


# ============================================================
# PREPROCESSING
# ============================================================

class TestPreprocessing:
    def test_frame_preprocessor_output_size(self, sample_image):
        pre = FramePreprocessor(img_size=320)
        out = pre.transform(sample_image)
        assert out.size == (320, 320)

    def test_frame_preprocessor_preserves_aspect_via_padding(self):
        img = Image.new("RGB", (100, 50), (255, 0, 0))
        pre = FramePreprocessor(img_size=64)
        out = pre.transform(img)
        assert out.size == (64, 64)

    def test_build_preprocessing_profiles_independent(self):
        stream, batch = build_preprocessing_profiles()
        assert stream.img_size != batch.img_size

    def test_compute_image_quality_flags_dark_image(self):
        dark_img = Image.new("RGB", (100, 100), (5, 5, 5))
        quality = compute_image_quality(dark_img)
        assert quality["is_too_dark"] is True

    def test_compute_image_quality_flags_bright_image(self):
        bright_img = Image.new("RGB", (100, 100), (250, 250, 250))
        quality = compute_image_quality(bright_img)
        assert quality["is_too_bright"] is True

    def test_resize_for_inference_matches_profile(self, sample_image):
        stream, _ = build_preprocessing_profiles()
        out = resize_for_inference(sample_image, stream)
        assert out.size == (stream.img_size, stream.img_size)


# ============================================================
# METRICS
# ============================================================

class TestMetrics:
    def test_psi_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 1000)
        assert psi(data, data) < 1e-6

    def test_psi_shifted_distribution_is_positive(self):
        rng = np.random.default_rng(0)
        expected = rng.normal(0, 1, 1000)
        actual   = rng.normal(3, 1, 1000)
        assert psi(expected, actual) > config.PSI_HIGH

    def test_psi_empty_input_returns_nan(self):
        assert np.isnan(psi([], [1, 2, 3]))

    def test_iou_identical_boxes_is_one(self):
        box = (0, 0, 10, 10)
        assert iou(box, box) == pytest.approx(1.0)

    def test_iou_disjoint_boxes_is_zero(self):
        assert iou((0, 0, 5, 5), (10, 10, 15, 15)) == 0.0

    def test_iou_partial_overlap_between_zero_and_one(self):
        val = iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.0 < val < 1.0

    def test_duplicate_box_rate_no_boxes(self):
        assert duplicate_box_rate([]) == 0.0

    def test_duplicate_box_rate_detects_overlap(self):
        boxes = [(0, 0, 10, 10), (0.5, 0.5, 10.5, 10.5)]
        assert duplicate_box_rate(boxes, iou_threshold=0.5) == 1.0

    def test_latency_stats_empty_list(self):
        assert latency_stats([])["mean_ms"] == 0.0

    def test_latency_stats_computes_fps(self):
        stats = latency_stats([100, 100, 100])
        assert stats["mean_ms"] == 100.0
        assert stats["fps"] == pytest.approx(10.0)

    def test_detection_rate(self):
        assert detection_rate([1, 0, 2, 0]) == 0.5

    def test_class_coverage(self):
        coverage = class_coverage({"person", "bus"}, {"person", "bus", "truck"})
        assert coverage == pytest.approx(2 / 3)

    def test_match_predictions_correct_match(self):
        preds = [{"class_id": 0, "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}]
        gt    = [{"class_id": 0, "bbox_xyxy": [0, 0, 10, 10]}]
        result = match_predictions_to_ground_truth(preds, gt, iou_threshold=0.5)
        assert result[0]["is_correct"] is True

    def test_match_predictions_wrong_class_not_matched(self):
        preds = [{"class_id": 1, "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}]
        gt    = [{"class_id": 0, "bbox_xyxy": [0, 0, 10, 10]}]
        result = match_predictions_to_ground_truth(preds, gt, iou_threshold=0.5)
        assert result[0]["is_correct"] is False

    def test_match_predictions_low_overlap_not_matched(self):
        preds = [{"class_id": 0, "confidence": 0.9, "bbox_xyxy": [0, 0, 5, 5]}]
        gt    = [{"class_id": 0, "bbox_xyxy": [20, 20, 30, 30]}]
        result = match_predictions_to_ground_truth(preds, gt, iou_threshold=0.5)
        assert result[0]["is_correct"] is False

    def test_match_predictions_greedy_prefers_higher_confidence(self):
        preds = [
            {"class_id": 0, "confidence": 0.6, "bbox_xyxy": [0, 0, 10, 10]},
            {"class_id": 0, "confidence": 0.9, "bbox_xyxy": [1, 1, 11, 11]},
        ]
        gt = [{"class_id": 0, "bbox_xyxy": [0, 0, 10, 10]}]
        result = match_predictions_to_ground_truth(preds, gt, iou_threshold=0.3)
        # Higher-confidence prediction (processed first) should claim the match
        high_conf_result = [r for r in result if r["confidence"] == 0.9][0]
        low_conf_result  = [r for r in result if r["confidence"] == 0.6][0]
        assert high_conf_result["is_correct"] is True
        assert low_conf_result["is_correct"] is False


# ============================================================
# LEAKAGE CHECK
# ============================================================

class TestLeakageCheck:
    def test_detect_frame_leakage_no_duplicates(self):
        paths = load_sample_dataset()
        warnings = detect_frame_leakage(paths)
        assert warnings == []

    def test_detect_frame_leakage_exact_duplicate_flagged(self, tmp_path):
        src = load_sample_dataset()[0]
        dup_path = tmp_path / "duplicate.jpg"
        Image.open(src).save(dup_path)
        warnings = detect_frame_leakage([src, str(dup_path)])
        assert any("LEAKAGE" in w for w in warnings)

    def test_check_class_distribution_sanity_flags_low_diversity(self):
        warnings = check_class_distribution_sanity({"person": 5, "bus": 0}, min_classes=2)
        assert len(warnings) == 1

    def test_check_class_distribution_sanity_passes_diverse_set(self):
        warnings = check_class_distribution_sanity({"person": 5, "bus": 3}, min_classes=2)
        assert warnings == []


# ============================================================
# DETECTION ENGINE
# ============================================================

class TestDetectionEngine:
    def test_score_frame_normal_when_empty(self):
        assert score_frame([], domain="security_intrusion")["decision"] == "NORMAL"

    def test_score_frame_alert_on_single_person_intrusion(self):
        detections = [{"class_name": "person", "confidence": 0.9}]
        result = score_frame(detections, domain="security_intrusion")
        assert result["decision"] == "ALERT"
        assert result["rule_triggered"] == "INTRUSION_PERSON_DETECTED"

    def test_score_frame_critical_on_multiple_persons(self):
        detections = [{"class_name": "person", "confidence": 0.9}] * 3
        assert score_frame(detections, domain="security_intrusion")["decision"] == "CRITICAL"

    def test_score_frame_shelf_low_stock_alert(self):
        result = score_frame([], domain="retail_shelf")
        assert result["decision"] == "ALERT"
        assert result["rule_triggered"] == "SHELF_LOW_STOCK"

    def test_score_frame_low_confidence_flagged(self):
        detections = [{"class_name": "cat", "confidence": 0.1}] * 5
        assert score_frame(detections, domain="warehouse_safety")["decision"] == "ALERT"

    def test_summarize_decisions_counts_correctly(self):
        decisions = [{"decision": "NORMAL"}, {"decision": "ALERT"}, {"decision": "ALERT"}]
        counts = summarize_decisions(decisions)
        assert counts["ALERT"] == 2
        assert counts["NORMAL"] == 1
        assert counts["CRITICAL"] == 0


# ============================================================
# COST-SENSITIVE EVALUATION & CALIBRATION
# ============================================================

class TestEvaluation:
    def test_cost_sensitive_evaluation_perfect_model_zero_cost(self):
        result = cost_sensitive_evaluation(precision=1.0, recall=1.0, total_ground_truth_instances=100)
        assert result["estimated_false_negatives"] == 0
        assert result["estimated_false_positives"] == 0
        assert result["total_estimated_cost"] == 0.0

    def test_cost_sensitive_evaluation_low_recall_increases_missed_loss(self):
        result = cost_sensitive_evaluation(precision=1.0, recall=0.5, total_ground_truth_instances=100)
        assert result["estimated_false_negatives"] == pytest.approx(50.0)
        assert result["estimated_missed_loss"] > 0

    def test_calibrate_confidence_scores_skips_when_insufficient_data(self):
        calibrator = calibrate_confidence_scores([0.5, 0.6], [1, 0])
        assert calibrator is None

    def test_calibrate_confidence_scores_fits_with_sufficient_data(self):
        rng = np.random.default_rng(0)
        confidences = rng.uniform(0, 1, 40).tolist()
        correctness = [1 if c > 0.5 else 0 for c in confidences]
        calibrator = calibrate_confidence_scores(confidences, correctness)
        assert calibrator is not None


# ============================================================
# MODEL CARD
# ============================================================

class TestModelCard:
    def test_build_and_save_and_load_roundtrip(self, tmp_path):
        card = build_model_card(
            variant="YOLOv8n", weights="yolov8n.pt",
            conf_threshold=0.25, iou_threshold=0.45,
            metrics={"map50_95": 0.36, "map50": 0.47, "precision": 0.71,
                     "recall": 0.49, "mean_latency_ms": 150, "fps": 6.7},
            coco_benchmark={"map50_95": 37.3},
            cost_result={"total_estimated_cost": 0.0},
            decision_counts={"NORMAL": 1},
            val_set_size=128,
            calibration_status="fitted",
            n_calibration_points=250,
        )
        path = save_model_card(card, str(tmp_path), "YOLOv8n")
        assert os.path.exists(path)

        loaded = load_model_card(str(tmp_path), "YOLOv8n")
        assert loaded["model_name"] == "YOLOv8n"
        assert loaded["metrics"]["map50_95"] == 0.36

    def test_load_model_card_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_card(str(tmp_path), "NoSuchVariant")


# ============================================================
# CHAMPION-CHALLENGER SYSTEM
# ============================================================

class TestChallengerSystem:
    def test_first_model_auto_promoted_when_sla_compliant(self, tmp_model_dir):
        result = run_challenger_comparison(
            challenger_variant="YOLOv8s", challenger_weights="yolov8s.pt",
            challenger_map50_95=0.53, challenger_precision=0.78,
            challenger_latency_ms=32.0, challenger_conf=0.25, challenger_iou=0.60,
            challenger_card_path="",
        )
        assert result["decision"] == "PROMOTED"
        assert os.path.exists(os.path.join(tmp_model_dir, "latest_model.json"))

    def test_first_model_rejected_when_latency_violates_sla(self, tmp_model_dir):
        # Even with no prior champion, a model that blows the real-time
        # latency ceiling must NOT be auto-promoted — precision/latency
        # are hard SLA gates, not relative comparisons.
        result = run_challenger_comparison(
            challenger_variant="YOLOv8m", challenger_weights="yolov8m.pt",
            challenger_map50_95=0.54, challenger_precision=0.75,
            challenger_latency_ms=178.0, challenger_conf=0.25, challenger_iou=0.60,
            challenger_card_path="",
        )
        assert result["decision"] == "REJECTED"
        assert result["gates"]["latency_ceiling_passed"] is False
        assert not os.path.exists(os.path.join(tmp_model_dir, "latest_model.json"))

    def test_first_model_rejected_when_precision_below_floor(self, tmp_model_dir):
        result = run_challenger_comparison(
            challenger_variant="YOLOv8n", challenger_weights="yolov8n.pt",
            challenger_map50_95=0.30, challenger_precision=0.40,
            challenger_latency_ms=17.0, challenger_conf=0.45, challenger_iou=0.45,
            challenger_card_path="",
        )
        assert result["decision"] == "REJECTED"
        assert result["gates"]["precision_floor_passed"] is False

    def test_worse_challenger_rejected(self, tmp_model_dir):
        card_path = os.path.join(tmp_model_dir, "model_card_YOLOv8n_v1.json")
        with open(card_path, "w") as f:
            json.dump({"model_name": "YOLOv8n",
                       "metrics": {"map50_95": 0.40, "precision": 0.80,
                                   "mean_latency_ms": 40}}, f)  # well under the effective ceiling

        first = run_challenger_comparison(
            challenger_variant="YOLOv8n", challenger_weights="yolov8n.pt",
            challenger_map50_95=0.40, challenger_precision=0.80,
            challenger_latency_ms=40.0, challenger_conf=0.25, challenger_iou=0.45,
            challenger_card_path=card_path,
        )
        assert first["decision"] == "PROMOTED"  # sanity: champion actually got registered

        # A genuinely worse challenger should be rejected on all 3 gates
        result = run_challenger_comparison(
            challenger_variant="YOLOv8m", challenger_weights="yolov8m.pt",
            challenger_map50_95=0.30, challenger_precision=0.40,
            challenger_latency_ms=999.0, challenger_conf=0.25, challenger_iou=0.45,
            challenger_card_path="",
        )
        assert result["decision"] == "REJECTED"
        assert all(v is False for v in result["gates"].values())

    def test_stale_champion_replaced_regardless_of_map(self, tmp_model_dir):
        """
        A champion that no longer meets current gates (e.g. latency
        ceiling tightened after it was promoted) must be treated as
        invalid — a compliant challenger should replace it even with
        a LOWER mAP, since the stale champion doesn't get to keep its
        spot just by having a better mAP than a non-compliant model.
        """
        stale_card_path = os.path.join(tmp_model_dir, "model_card_YOLOv8m_v1.json")
        with open(stale_card_path, "w") as f:
            json.dump({"model_name": "YOLOv8m",
                       "metrics": {"map50_95": 0.60, "precision": 0.80,
                                   "mean_latency_ms": 100.0}}, f)  # exceeds effective ceiling (84ms)

        registry_path = os.path.join(tmp_model_dir, "latest_model.json")
        with open(registry_path, "w") as f:
            json.dump({"variant": "YOLOv8m", "weights": "yolov8m.pt",
                       "conf_threshold": 0.25, "iou_threshold": 0.45,
                       "model_card_path": stale_card_path}, f)

        # Lower mAP than the stale champion, but genuinely SLA-compliant
        result = run_challenger_comparison(
            challenger_variant="YOLOv8s", challenger_weights="yolov8s.pt",
            challenger_map50_95=0.50, challenger_precision=0.75,
            challenger_latency_ms=35.0, challenger_conf=0.25, challenger_iou=0.45,
            challenger_card_path="",
        )
        assert result["decision"] == "PROMOTED"

    def test_load_latest_model_missing_registry_raises(self, tmp_model_dir):
        with pytest.raises(FileNotFoundError):
            load_latest_model()


# ============================================================
# END-TO-END SANITY (uses trained artifacts if present)
# ============================================================

class TestEndToEndArtifacts:
    def test_registry_file_shape_if_present(self):
        registry_path = os.path.join(config.MODEL_DIR, "latest_model.json")
        if not os.path.exists(registry_path):
            pytest.skip("Run scripts/train_model.py first to generate the registry")
        with open(registry_path) as f:
            registry = json.load(f)
        assert "variant" in registry
        assert "weights" in registry

    def test_experiment_results_csv_if_present(self):
        exp_path = os.path.join(config.MODEL_DIR, "model_experiment_results.csv")
        if not os.path.exists(exp_path):
            pytest.skip("Run scripts/train_model.py first to generate experiment results")
        df = pd.read_csv(exp_path)
        assert "coco128_map50_95" in df.columns
        assert len(df) > 0