"""GT 없는 적응 규칙·경계·동결 검출기·공개 실행 인터페이스를 검증한다."""
import copy
import hashlib
import json
from unittest.mock import patch
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import adaptive_preprocess as adaptive
import run_final
from metrics import measure_quality
from preprocess import validate_config, validate_msr_config


def config():
    return {"version": 1, "geometry": validate_config({"resize": {"long_side": 160}}),
            "msr": {"enabled": False, "alpha": .25, "parameters": validate_msr_config()},
            "gamma": {"enabled": True, "low": 80., "high": 160., "dark_value": .7, "bright_value": 1.2},
            "gaussian": {"enabled": True, "low": .8, "high": 1.5, "moderate_sigma": .5, "high_sigma": .8, "kernel": 3}}


class AdaptiveTests(unittest.TestCase):
    def test_gamma_boundaries_and_skip(self):
        rule = config()["gamma"]
        for value, expected in ((50, .7), (80, 1), (120, 1), (160, 1), (200, 1.2)):
            self.assertEqual(adaptive.gamma_choice({"gray_mean": value}, rule)[0], expected)
        self.assertEqual(adaptive.gamma_choice({"gray_mean": 50}, {**rule, "enabled": False})[0], 1)

    def test_noise_boundaries_and_unavailable(self):
        rule = config()["gaussian"]
        for value, expected in ((0, 0), (.8, 0), (1, .5), (1.5, .5), (2, .8), (None, 0), (float("nan"), 0)):
            self.assertEqual(adaptive.gaussian_choice({"noise_sigma": value}, rule)[0], expected)

    def test_gt_free_runtime_skip_and_metadata(self):
        image = np.full((120, 190, 3), 120, np.uint8)
        before = image.copy()
        with patch("data.load_gt", side_effect=AssertionError("GT access forbidden")), patch("data.list_images", side_effect=AssertionError("dataset access forbidden")), patch("detect.detect", side_effect=AssertionError("detector access forbidden")):
            fixed, meta = adaptive.adaptive_preprocess(image, config())
        self.assertFalse(meta["msr_applied"])
        self.assertFalse(meta["gamma_applied"])
        self.assertFalse(meta["gaussian_applied"])
        self.assertEqual(fixed.dtype, np.uint8)
        self.assertTrue(np.all(fixed[meta["roi_mask"] == 0] == 0))
        self.assertAlmostEqual(meta["quality_after"]["gray_mean"], 120)
        np.testing.assert_array_equal(image, before)

    def test_gamma_changes_dark_and_bright_in_correct_direction(self):
        for level, direction in ((50, 1), (200, -1)):
            _, meta = adaptive.adaptive_preprocess(np.full((90, 170, 3), level, np.uint8), config())
            self.assertTrue(meta["gamma_applied"])
            self.assertGreater(direction * (meta["quality_after"]["gray_mean"] - level), 0)

    def test_all_disabled_stages_equal_roi(self):
        cfg = config()
        cfg["gamma"]["enabled"] = cfg["gaussian"]["enabled"] = False
        image = np.random.default_rng(5).integers(0, 256, (91, 131, 3), dtype=np.uint8)
        outputs = [adaptive.adaptive_preprocess(image, cfg, stage)[0] for stage in ("B0", "B1", "B2", "B3", "FINAL")]
        for output in outputs[1:]:
            np.testing.assert_array_equal(output, outputs[0])

    def test_external_pixels_do_not_affect_quality(self):
        fixed, meta = adaptive.adaptive_preprocess(np.full((90, 170, 3), 120, np.uint8), config())
        modified = fixed.copy()
        modified[meta["roi_mask"] == 0] = 255
        self.assertEqual(measure_quality(fixed, meta["roi_mask"]), measure_quality(modified, meta["roi_mask"]))

    def test_validation_rejects_bad_rules_and_keeps_input(self):
        cfg = config()
        before = copy.deepcopy(cfg)
        adaptive.validate_adaptive_config(cfg)
        self.assertEqual(cfg, before)
        for stage, key, value in (("gamma", "low", float("inf")), ("gamma", "dark_value", 0),
                                  ("gaussian", "high_sigma", -1), ("msr", "alpha", 2)):
            changed = copy.deepcopy(cfg)
            changed[stage][key] = value
            with self.assertRaises(ValueError):
                adaptive.validate_adaptive_config(changed)

    def test_frozen_detector_and_dataset_files(self):
        metadata = ROOT / "backups/before_adaptive_preprocessing/frozen_hashes.json"
        for item in json.loads(metadata.read_text(encoding="utf-8-sig")):
            actual = hashlib.sha256((ROOT / item["file"]).read_bytes()).hexdigest()
            self.assertEqual(actual, item["sha256"], item["file"])

    def test_final_runner_rejects_test_before_loading_config_or_images(self):
        with patch.object(run_final, "load_adaptive_config") as loader:
            for dataset in ("rdd", "rdd_test"):
                with self.assertRaises(ValueError):
                    run_final.run(dataset)
            loader.assert_not_called()

    def test_public_preprocess_adaptive_interface_matches_runtime(self):
        from preprocess import adaptive_preprocess as public_api
        image = np.full((91, 131, 3), 50, np.uint8)
        actual, meta = public_api(image, config())
        expected, expected_meta = adaptive.adaptive_preprocess(image, config())
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(meta["quality_after"], expected_meta["quality_after"])


if __name__ == "__main__":
    unittest.main()
