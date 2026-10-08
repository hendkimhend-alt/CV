"""모든 실행 경로의 FINAL 설정 적용과 GT 평가·파일 저장을 검증한다."""
import copy
import csv
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import adaptive_preprocess as adaptive
import preprocess
import run_detect
import run_final
import run_pipeline
from paths import imread, imwrite


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AssertionError(f"CSV 열 중복: {path}")
        return list(reader)


class FinalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = adaptive.load_adaptive_config()
        self.cfg["geometry"]["resize"]["long_side"] = 160
        self.cfg["gamma"].update(enabled=True, low=80, high=160, dark_value=.7, bright_value=1.2)
        self.image = np.full((120, 190, 3), 40, np.uint8)

    def test_public_and_detect_interfaces_use_same_final_config(self):
        original = copy.deepcopy(self.cfg)
        expected, meta = adaptive.adaptive_preprocess(self.image, self.cfg)
        actual, mask = preprocess.preprocess(self.image, self.cfg, return_mask=True)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(mask, meta["roi_mask"])
        self.assertEqual(max(actual.shape[:2]), 160)
        self.assertGreater(int(actual[mask != 0].mean()), 40)
        gray, detector_mask = run_detect.prepare(self.image, True, self.cfg)
        np.testing.assert_array_equal(gray, cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY))
        np.testing.assert_array_equal(mask, detector_mask)
        self.assertEqual(self.cfg, original)

    def test_legacy_condition_rejected_before_loading_dataset(self):
        with patch.object(run_pipeline, "list_images", side_effect=AssertionError("dataset accessed")):
            for condition in ("P0_reference", "P1", "P1+"):
                with self.assertRaises(ValueError):
                    run_pipeline.run(conditions=(condition,))

    def test_roi_defaults_come_from_final_json(self):
        raw = json.loads(preprocess.FINAL_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(preprocess.DEFAULT_CONFIG, raw["geometry"])
        self.assertEqual(preprocess.FINAL_CONFIG_PATH, adaptive.CONFIG_PATH)

    def test_pipeline_and_final_share_pixels_settings_and_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image_path = folder / "road.png"
            imwrite(image_path, self.image)
            original_bytes = image_path.read_bytes()
            config_path = folder / "settings.json"
            config_path.write_text(json.dumps(self.cfg), encoding="utf-8")
            listed = ("captured", folder, [image_path])
            gt = [("crack", 80, 85, 110, 110)]
            with patch.object(run_pipeline, "OUTPUT_DIR", folder), patch.object(run_pipeline, "list_images", return_value=listed), patch.object(run_pipeline, "load_gt", return_value=lambda path: gt):
                evaluated = run_pipeline.run("captured", keypoints=False, save_images=-1, config_path=config_path)
            with patch.object(run_final, "list_images", return_value=listed):
                final = run_final.run("captured", output=folder / "final", config_path=config_path, save_images=-1)
            eval_rows, final_rows = read_rows(evaluated / "results.csv"), read_rows(final / "results.csv")
            self.assertEqual(len(eval_rows), 1)
            self.assertEqual(eval_rows[0]["condition"], "FINAL")
            self.assertEqual(eval_rows[0]["has_gt"], "True")
            self.assertEqual(eval_rows[0]["crack_n_gt"], "1")
            for key in ("gamma", "gamma_applied", "msr_applied", "gaussian_applied", *run_pipeline.QUALITY_KEYS):
                self.assertEqual(eval_rows[0][key], final_rows[0][key], key)
            count = int(eval_rows[0]["n_crack"]) + int(eval_rows[0]["n_pothole"])
            self.assertEqual(count, int(final_rows[0]["total_candidates"]))
            predictions = read_rows(final / "detections.csv")
            self.assertEqual(len(predictions), count)
            self.assertTrue((evaluated / "summary.csv").is_file())
            self.assertEqual(imread(next(evaluated.glob("images/**/*.png"))).shape, imread(next(final.glob("images/*.png"))).shape)
            self.assertEqual(image_path.read_bytes(), original_bytes)

    def test_quality_runner_uses_final_config_and_saves_corrected_image(self):
        quality_runner = importlib.import_module("A")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            input_dir = folder / "input"
            input_dir.mkdir()
            image_path = input_dir / "road.png"
            imwrite(image_path, self.image)
            with patch.object(quality_runner, "prepare_plots", return_value=None), patch.object(quality_runner, "create_plots"):
                out = quality_runner.run_experiments(input_dir, folder / "quality", cfg=self.cfg)
            rows = read_rows(out / "results.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["condition"], "FINAL")
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[0]["gamma_applied"], "True")
            expected, _ = adaptive.adaptive_preprocess(self.image, self.cfg)
            np.testing.assert_array_equal(imread(rows[0]["output_path"]), expected)
            self.assertEqual(json.loads((out / "run_config.json").read_text(encoding="utf-8"))["config"], self.cfg)


if __name__ == "__main__":
    unittest.main()
