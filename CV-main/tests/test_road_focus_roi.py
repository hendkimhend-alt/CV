"""ROI 적용·원본 보존·좌표 변환·분석 이전 파라미터 유지 검증."""
import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import detect
import evaluate
import metrics
import preprocess
import roi
import run_detect
import run_pipeline


def baseline_module(name):
    spec = importlib.util.spec_from_file_location("baseline_" + name, ROOT / "backups" / "before_road_focus_roi" / "src" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RoadFocusRoiTests(unittest.TestCase):
    def setUp(self):
        self.cfg = preprocess.validate_config({})
        self.image = np.full((512, 512, 3), 160, np.uint8)

    def test_selected_coordinates_and_polygon_region(self):
        expected = dict(top_y_ratio=0.60, top_left_x_ratio=0.075, top_right_x_ratio=0.725,
                        bottom_left_x_ratio=0.05, bottom_right_x_ratio=0.95, bottom_y_ratio=1.00)
        self.assertEqual(self.cfg["roi"], {"type": "trapezoid", **expected})
        cases = (
            (512, 512, [[38,307],[370,307],[485,511],[26,511]]),
            (721, 1281, [[96,432],[928,432],[1216,720],[64,720]]),
            (1080, 1920, [[144,647],[1391,647],[1823,1079],[96,1079]]),
        )
        for height, width, expected_points in cases:
            mask, points = roi.build_trapezoid_mask(height, width, self.cfg["roi"])
            self.assertEqual(points.tolist(), expected_points)
            self.assertFalse(mask[:expected_points[0][1]].any())
            self.assertEqual(mask[-1, width // 2], 255)
            self.assertEqual(mask[-1, 0], 0)
            self.assertEqual(mask[-1, -1], 0)

    def test_crop_resize_mask_and_source_preservation(self):
        before = self.image.copy()
        ref, geom = preprocess.geometry_preprocess(self.image, self.cfg)
        self.assertEqual(json.loads(geom["roi_polygon_json"]), [[38,307],[370,307],[485,511],[26,511]])
        self.assertEqual(tuple(geom[k] for k in ("crop_x","crop_y","crop_width","crop_height")), (26,307,460,205))
        self.assertEqual(ref.shape, (456,1024,3))
        mask = roi.geometry_mask(geom)
        self.assertTrue(np.all(ref[mask == 0] == 0))
        self.assertTrue(np.all(ref[mask != 0] == 160))
        np.testing.assert_array_equal(self.image, before)
        json.dumps(geom, allow_nan=False)

    def test_final_preserves_mask_and_disables_legacy_conditions(self):
        ref, geom = preprocess.geometry_preprocess(self.image, self.cfg)
        mask = roi.geometry_mask(geom)
        self.assertEqual(preprocess.CONDITIONS, ("FINAL",))
        fixed = preprocess.preprocess(self.image)
        np.testing.assert_array_equal(fixed, ref)
        self.assertTrue(np.all(fixed[mask == 0] == 0))
        self.assertFalse(hasattr(preprocess, "preprocess_condition"))
        for condition in ("P0_reference", "P1", "P1+"):
            with self.assertRaises(ValueError):
                preprocess.preprocess(self.image, {"condition": condition})

    def test_masked_quality_excludes_padding_and_boundary(self):
        ref, geom = preprocess.geometry_preprocess(self.image, self.cfg)
        values = metrics.measure_quality(ref, roi.geometry_mask(geom))
        for name in ("gray_mean", "hsv_v_mean"):
            self.assertEqual(values[name], 160)
        for name in ("gray_std", "saturation_ratio", "dark_ratio", "bright_ratio",
                     "block_mean_std_4x4", "laplacian_variance", "noise_sigma"):
            self.assertEqual(values[name], 0)
        self.assertEqual(preprocess.classify_quality(values), ("blur",))

    def test_uniform_road_does_not_detect_polygon_boundary(self):
        ref, geom = preprocess.geometry_preprocess(self.image, self.cfg)
        mask = roi.geometry_mask(geom)
        fixed = preprocess.preprocess(self.image)
        for detector in ("D0", "D1"):
            self.assertEqual(detect.detect(fixed, {"detector": detector}, roi_mask=mask), [])
        self.assertEqual(run_pipeline.count_edges(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), mask), 0)

    def test_detector_excludes_outside_components(self):
        ref, geom = preprocess.geometry_preprocess(self.image, self.cfg)
        mask = roi.geometry_mask(geom)
        fake = np.zeros(mask.shape, np.uint8)
        fake[20:30, 2:8] = 255
        self.assertTrue(np.all(mask[20:30, 2:8] == 0))
        with patch.object(detect, "_d0_mask", return_value=fake):
            self.assertEqual(detect.detect(ref, {"detector":"D0", "keep_noise":True}, roi_mask=mask), [])

    def test_gt_crop_scale_and_polygon_exclusion(self):
        _, geom = preprocess.geometry_preprocess(self.image, self.cfg)
        boxes = [("crack", 40,310,45,320),       # 외접 bbox와 사다리꼴 안
                 ("pothole", 470,310,480,320), # bbox 안, 사다리꼴 밖
                 ("crack", 200,400,220,430),   # 완전히 안
                 ("pothole", 10,490,40,511),   # 일부만 보임
                 ("crack", 100,0,200,100)]     # 위쪽 밖
        transformed = evaluate.transform_gt(boxes, geom)
        self.assertEqual(len(transformed), 3)
        middle = transformed[1]
        self.assertEqual(middle[0], "crack")
        self.assertAlmostEqual(middle[1], (200-26) * geom["scale_x"])
        self.assertAlmostEqual(middle[2], (400-307) * geom["scale_y"])
        self.assertEqual(transformed[2][1], 0)

    def test_fractional_bbox_intersection_matches_pixel_area(self):
        mask = np.array([[1,0,1],[0,1,1],[1,1,0]], np.uint8)
        integral = cv2.integral(mask, sdepth=cv2.CV_64F)
        for bbox in ((0.2,0.3,2.8,2.7), (-2,-1,0.5,0.8), (2.5,2.5,4,5)):
            x1,y1,x2,y2 = bbox
            expected = sum(float(mask[y,x]) * max(0,min(x+1,x2)-max(x,x1))
                           * max(0,min(y+1,y2)-max(y,y1)) for y in range(3) for x in range(3))
            self.assertAlmostEqual(roi.bbox_intersection_area(integral, bbox), expected, places=12)

    def test_low_level_rectangular_geometry_equal_baseline(self):
        previous = baseline_module("preprocess")
        rng = np.random.default_rng(42)
        image = rng.integers(0,256,(129,211,3),dtype=np.uint8)
        for setting in ("bottom_half", "full", [3,7,120,80]):
            old_cfg = previous.validate_config({"roi": setting})
            cfg = preprocess.validate_config({"roi": setting})
            before, old_geom = previous.geometry_preprocess(image, old_cfg)
            after, geom = preprocess.geometry_preprocess(image, cfg)
            np.testing.assert_array_equal(before, after)
            self.assertIsNone(roi.geometry_mask(geom))
            for key in old_geom:
                self.assertEqual(geom[key], old_geom[key])

    def test_no_mask_metrics_and_detectors_equal_baseline(self):
        old_metrics, old_detect = baseline_module("metrics"), baseline_module("detect")
        rng = np.random.default_rng(43)
        image = rng.integers(0,256,(73,121,3),dtype=np.uint8)
        self.assertEqual(metrics.measure_quality(image), old_metrics.measure_quality(image))
        full = np.full(image.shape[:2],255,np.uint8)
        self.assertEqual(metrics.measure_quality(image, full), old_metrics.measure_quality(image))
        for name in ("D0","D1"):
            cfg = {"detector":name,"keep_noise":True}
            self.assertEqual(detect.detect(image,cfg), old_detect.detect(image,cfg))
            self.assertEqual(detect.detect(image,cfg,full), old_detect.detect(image,cfg))

    def test_parameters_and_correction_functions_unchanged(self):
        previous = baseline_module("preprocess")
        for key in previous.DEFAULT_CONFIG:
            if key != "roi":
                self.assertEqual(preprocess.DEFAULT_CONFIG[key], previous.DEFAULT_CONFIG[key])
        self.assertEqual(detect.DEFAULT_CFG, baseline_module("detect").DEFAULT_CFG)
        old_ast = ast.parse((ROOT / "backups/before_road_focus_roi/src/preprocess.py").read_text(encoding="utf-8"))
        new_ast = ast.parse((ROOT / "src/preprocess.py").read_text(encoding="utf-8"))
        for name in ("apply_gamma","gamma_lut","apply_clahe","apply_unsharp","apply_gaussian",
                     "classify_quality","validate_thresholds","parse_tags"):
            old = next(node for node in old_ast.body if isinstance(node,ast.FunctionDef) and node.name == name)
            new = next(node for node in new_ast.body if isinstance(node,ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(old), ast.dump(new))

    def test_invalid_coordinates_and_overrides(self):
        for key, value in (("top_y_ratio",float("nan")), ("bottom_y_ratio",0.50),
                           ("top_left_x_ratio",True), ("top_right_x_ratio",1.01)):
            with self.assertRaises(ValueError):
                preprocess.validate_config({"roi":{**self.cfg["roi"],key:value}})
        cfg = preprocess.validate_config({"roi_overrides":{"sample":"full"}})
        ref, geom = preprocess.geometry_preprocess(self.image,cfg,"sample")
        self.assertIsNone(roi.geometry_mask(geom))
        self.assertTrue(ref.any())

    def test_public_preprocess_and_detector_prepare_use_same_mask(self):
        fixed, mask = preprocess.preprocess(self.image,return_mask=True)
        gray, detector_mask = run_detect.prepare(self.image,return_mask=True)
        np.testing.assert_array_equal(gray,cv2.cvtColor(fixed,cv2.COLOR_BGR2GRAY))
        np.testing.assert_array_equal(mask,detector_mask)
        np.testing.assert_array_equal(preprocess.preprocess(self.image,{"condition":"FINAL"}),fixed)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
