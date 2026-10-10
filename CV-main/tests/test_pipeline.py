"""파이프라인 전체 — 출력 규약 · 수동 수정 흐름 · 노이즈 재측정 · metadata 일관성 · 재현성 · 검출기 연동."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import RDD_IMG, config, synthetic_scene
from preprocessing.detector_adapter import to_detector_input, to_original_bbox
from preprocessing.pipeline import ERROR, MANUAL_REQUIRED, SUCCESS, process_image, save_result


def strip_timing(meta):
    meta = json.loads(json.dumps(meta))
    meta.pop("timing_ms", None)
    meta.get("roi", {}).pop("stage_timing_ms", None)
    return meta


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # 어두운 노면(평균 45 · σ 6) → gray_mean < 80 이고 dark_ratio(< 40) ≈ 0.2 > 0.07 → Gamma AND 조건 충족
        self.img, self.regions = synthetic_scene(road_gray=45)

    def test_success_outputs_and_outside_pixels(self):
        cfg = config(**{"gaussian.mode": "fixed"})
        result = process_image(self.img, cfg, image_id="synthetic")
        self.assertEqual(result.status, SUCCESS, result.metadata["errors"])
        processed, road, analysis = result.processed_image, result.road_mask, result.analysis_mask
        self.assertEqual(processed.shape, self.img.shape)
        self.assertEqual(road.shape, self.img.shape[:2])
        self.assertTrue(np.array_equal(processed[road == 0], self.img[road == 0]))
        self.assertFalse(np.any((analysis > 0) & (road == 0)))
        self.assertFalse(np.array_equal(processed[road > 0], self.img[road > 0]))
        meta = result.metadata
        self.assertTrue(meta["gamma"]["applied"])
        self.assertTrue(meta["gaussian"]["applied"])
        self.assertEqual(meta["gaussian"]["sigma_used"], cfg["gaussian"]["sigma"])

    def test_noise_is_remeasured_after_gamma(self):
        result = process_image(self.img, config(**{"gamma.mode": "fixed", "gamma.value": 0.7,
                                                   "gamma.guard.max_saturation_ratio_after": None}))
        quality = result.metadata["quality"]
        self.assertEqual(result.metadata["noise_remeasurement"]["noise_sigma"], quality["after_gamma"]["noise_sigma"])
        self.assertNotEqual(quality["after_gamma"]["noise_sigma"], quality["initial"]["noise_sigma"])
        self.assertGreater(quality["after_gamma"]["gray_mean"], quality["initial"]["gray_mean"])

    def test_gamma_and_gaussian_off_returns_input(self):
        result = process_image(self.img, config(**{"gamma.mode": "off", "gaussian.mode": "off"}))
        self.assertTrue(np.array_equal(result.processed_image, self.img))
        self.assertEqual(result.metadata["quality"]["initial"], result.metadata["quality"]["final"])

    def test_fail_requires_manual_mask_and_accepts_one(self):
        strict = config(**{"validation.metrics.mask_area_ratio.min": 0.95})       # 일부러 FAIL
        result = process_image(self.img, strict, image_id="fail_case")
        self.assertEqual(result.status, MANUAL_REQUIRED)
        self.assertIsNone(result.processed_image)
        self.assertEqual(result.metadata["validation"]["status"], "FAIL")
        self.assertTrue(result.metadata["manual_correction"]["required"])
        manual = np.zeros(self.img.shape[:2], np.uint8)
        manual[100:, 60:] = 255
        fixed = process_image(self.img, strict, manual_mask=manual, manual_source="manual.png")
        self.assertEqual(fixed.status, SUCCESS)
        self.assertEqual(fixed.metadata["final_mask"]["source"], "manual")
        self.assertTrue(np.array_equal(fixed.road_mask, manual))
        edited = process_image(self.img, strict, editor=lambda image, mask, title: manual)
        self.assertEqual(edited.metadata["manual_correction"]["source"], "interactive_editor")
        cancelled = process_image(self.img, strict, editor=lambda image, mask, title: None)
        self.assertEqual(cancelled.status, MANUAL_REQUIRED)

    def test_review_pass_result_calls_editor(self):
        calls = []
        process_image(self.img, config(), editor=lambda i, m, t: calls.append(t), review=False)
        self.assertEqual(calls, [])
        process_image(self.img, config(), editor=lambda i, m, t: calls.append(t), review=True)
        self.assertEqual(len(calls), 1)

    def test_invalid_input_is_error(self):
        result = process_image(np.zeros((10, 10), np.uint8), config(), image_id="bad")
        self.assertEqual(result.status, ERROR)
        self.assertEqual(result.metadata["errors"][0]["stage"], "validate_input")

    def test_metadata_consistency_and_saved_files(self):
        cfg = config()
        result = process_image(self.img, cfg, image_id="meta")
        meta = result.metadata
        self.assertEqual(meta["final_mask"]["road_pixels"], int(np.count_nonzero(result.road_mask)))
        self.assertEqual(meta["analysis_mask"]["pixels"], int(np.count_nonzero(result.analysis_mask)))
        if meta["gamma"]["applied"]:
            self.assertIn(meta["gamma"]["value_used"], [c for c in cfg["gamma"]["candidates"] if c < 1.0])   # adaptive
        else:
            self.assertEqual(meta["gamma"]["value_used"], 1.0)
        self.assertEqual(meta["config_version"], cfg["config_version"])
        for name in ("gray_mean", "dark_ratio", "noise_sigma", "laplacian_variance"):
            change = meta["quality"]["change_final_minus_initial"][name]
            self.assertAlmostEqual(change, meta["quality"]["final"][name] - meta["quality"]["initial"][name], places=5)
        with tempfile.TemporaryDirectory() as tmp:
            outputs = save_result(result, Path(tmp) / "meta", cfg["output"], self.img)
            for name in outputs.values():
                self.assertTrue((Path(tmp) / "meta" / name).is_file(), name)
            saved = json.loads((Path(tmp) / "meta" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], SUCCESS)
            self.assertEqual(saved["outputs"]["road_mask"], "road_mask.png")

    def test_reproducible(self):
        a = process_image(self.img, config(), image_id="r")
        b = process_image(self.img.copy(), config(), image_id="r")
        self.assertTrue(np.array_equal(a.processed_image, b.processed_image))
        self.assertTrue(np.array_equal(a.road_mask, b.road_mask))
        self.assertEqual(strip_timing(a.metadata), strip_timing(b.metadata))


class DetectorCompatibilityTests(unittest.TestCase):
    def test_adapter_feeds_existing_detector(self):
        from detect import detect
        img, _ = synthetic_scene(height=300, width=400)
        result = process_image(img, config())
        det_img, det_mask, geometry = to_detector_input(result.processed_image, result.road_mask)
        self.assertEqual(max(det_img.shape[:2]), 1024)
        self.assertEqual(det_mask.shape, det_img.shape[:2])
        self.assertTrue(set(np.unique(det_mask).tolist()) <= {0, 255})
        detections = detect(det_img)                          # 기본 D1 · 알고리즘 변경 없음
        self.assertIsInstance(detections, list)
        x, y, w, h = to_original_bbox((256, 384, 128, 64), geometry)
        self.assertAlmostEqual(x, 100.0)
        self.assertAlmostEqual(y, 150.0)

    def test_crop_to_road_geometry_roundtrip(self):
        img, _ = synthetic_scene(height=300, width=400)
        road = np.zeros((300, 400), np.uint8)
        road[120:300, 60:400] = 255
        det_img, det_mask, g = to_detector_input(img, road, 1024, crop_to_road=True, crop_margin=10)
        self.assertEqual((g["crop_x"], g["crop_y"], g["crop_width"], g["crop_height"]), (50, 110, 350, 190))
        self.assertEqual(max(det_img.shape[:2]), 1024)
        x, y, w, h = to_original_bbox((0, 0, g["output_width"], g["output_height"]), g)
        self.assertAlmostEqual(x, 50.0)
        self.assertAlmostEqual(y, 110.0)
        self.assertAlmostEqual(w, 350.0, places=6)
        self.assertAlmostEqual(h, 190.0, places=6)
        with self.assertRaises(ValueError):
            to_detector_input(img, np.zeros((300, 400), np.uint8), crop_to_road=True)


class TestSplitGuardTests(unittest.TestCase):
    @unittest.skipUnless(RDD_IMG.is_dir(), "RDD 데이터 없음")
    def test_test_split_images_are_skipped(self):
        from preprocessing import run_preprocess
        test_names = sorted(run_preprocess.test_split_names())
        self.assertEqual(len(test_names), 241)
        args = argparse.Namespace(dataset=None, input=str(RDD_IMG / test_names[0]), images=None, limit=None,
                                  allow_rdd_test=False)
        self.assertEqual(run_preprocess.collect_images(args), [])
        args.dataset, args.input = "rdd_test", None
        with self.assertRaises(SystemExit):
            run_preprocess.collect_images(args)


if __name__ == "__main__":
    unittest.main()
