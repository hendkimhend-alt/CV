"""PASS/FAIL 검증 · 수동 마스크(편집기 · 외부 파일) · analysis_mask 침식."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import config, synthetic_scene
from preprocessing.analysis_mask import make_analysis_mask
from preprocessing.image_io import write_image
from preprocessing.mask_editor import (MaskEditor, combine_masks, find_external_mask, load_external_mask,
                                       mask_to_polygon, polygon_to_mask)
from preprocessing.mask_validation import FAIL, PASS, compute_mask_metrics, validate_road_mask


def rect_mask(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()["validation"]

    def test_metric_definitions(self):
        mask = rect_mask((100, 100), 50, 100, 0, 100)    # 면적 0.5
        mask[60:70, 40:50] = 0                           # 구멍 100px
        mask[0:10, 0:10] = 255                           # 작은 요소 100px
        m = compute_mask_metrics(mask, [(50, 80), (5, 5), (45, 65), (50, 20)])
        road = 50 * 100 - 100 + 100
        self.assertAlmostEqual(m["mask_area_ratio"], road / 10000)
        self.assertAlmostEqual(m["largest_component_ratio"], (5000 - 100) / road)
        self.assertAlmostEqual(m["hole_ratio"], 100 / (road + 100))
        self.assertAlmostEqual(m["seed_consistency"], 2 / 4)   # (45,65)는 구멍, (50,20)은 비도로
        self.assertEqual(m["n_components"], 2)

    def test_pass_and_fail_reasons(self):
        good = rect_mask((100, 100), 40, 100, 0, 100)
        result = validate_road_mask(compute_mask_metrics(good, [(50, 70), (20, 90), (80, 90)]), self.cfg)
        self.assertEqual(result["status"], PASS)
        small = rect_mask((100, 100), 95, 100, 0, 100)    # 면적 0.05 < 0.10
        result = validate_road_mask(compute_mask_metrics(small, [(50, 97)]), self.cfg)
        self.assertEqual(result["status"], FAIL)
        self.assertTrue(any(r.startswith("mask_area_ratio<") for r in result["fail_reasons"]))

    def test_empty_mask_and_no_seed_fail_and_disabled_metric(self):
        empty = np.zeros((50, 50), np.uint8)
        result = validate_road_mask(compute_mask_metrics(empty, []), self.cfg)
        self.assertEqual(result["status"], FAIL)
        self.assertIn("no_seed", result["fail_reasons"])
        self.assertIn("empty_mask", result["fail_reasons"])
        self.assertIn("largest_component_ratio:undefined", result["fail_reasons"])
        split = rect_mask((100, 100), 40, 100, 0, 30)
        split[40:100, 60:100] = 255                        # 떨어진 두 노면 → 가장 큰 요소 비율 0.57
        metrics = compute_mask_metrics(split, [(10, 70), (80, 70)])
        self.assertEqual(validate_road_mask(metrics, self.cfg)["status"], FAIL)
        cfg = config(**{"validation.metrics.largest_component_ratio.enabled": False})["validation"]
        self.assertEqual(validate_road_mask(metrics, cfg)["status"], PASS)


class ManualMaskTests(unittest.TestCase):
    def test_polygon_modes(self):
        base = rect_mask((60, 80), 30, 60, 0, 80)
        poly = polygon_to_mask([(10, 10), (40, 10), (40, 40), (10, 40)], base.shape)
        self.assertEqual(np.count_nonzero(poly), 31 * 31)
        self.assertTrue(np.array_equal(combine_masks(base, poly, "replace"), poly))
        self.assertEqual(np.count_nonzero(combine_masks(base, poly, "add")),
                         np.count_nonzero((base > 0) | (poly > 0)))
        self.assertFalse(combine_masks(base, poly, "subtract")[35, 20])

    def test_editor_state_without_gui(self):
        img, _ = synthetic_scene()
        auto = rect_mask(img.shape[:2], 100, 240, 50, 320)
        auto[150:160, 300:320] = 0                                       # 영상 가장자리로 열린 오목부
        second = auto.copy()
        second[0:40, 0:40] = 255                                         # 떨어진 두 번째 영역
        editor = MaskEditor(img, second, "test")
        self.assertEqual(editor.polygon, [])                             # 시작 = 기존 마스크 그대로
        self.assertTrue(np.array_equal(editor.confirm(), second))        # 아무것도 안 그리고 확정 → 변화 없음
        editor.load_outline()                                            # e: 가장 큰 외곽선 불러오기
        self.assertEqual(editor.mode, "replace")
        self.assertGreaterEqual(len(editor.polygon), 4)
        self.assertFalse(editor.confirm()[10, 10])                       # 바꾸기 → 다른 영역은 사라짐
        editor = MaskEditor(img, auto, "test")
        editor.set_mode("subtract")
        editor.set_polygon([(50, 100), (120, 100), (120, 240), (50, 240)])
        edited = editor.confirm()
        self.assertFalse(edited[150, 80])
        self.assertTrue(edited[150, 200])
        self.assertTrue(set(np.unique(edited).tolist()) <= {0, 255})

    def test_external_mask_lookup_and_load(self):
        img, _ = synthetic_scene()
        cfg = config()["manual_correction"]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_external_mask(tmp, "Japan_000001.jpg"))
            mask = rect_mask(img.shape[:2], 100, 240, 0, 320)
            mask[mask == 255] = 200                                    # 임계값 128 이상 = 도로
            path = write_image(Path(tmp) / "Japan_000001_road_mask.png", mask)
            self.assertEqual(find_external_mask(tmp, "Japan_000001.jpg"), path)
            loaded = load_external_mask(path, img.shape, cfg)
            self.assertTrue(set(np.unique(loaded).tolist()) == {0, 255})
            write_image(Path(tmp) / "empty.png", np.zeros(img.shape[:2], np.uint8))
            with self.assertRaises(ValueError):
                load_external_mask(Path(tmp) / "empty.png", img.shape, cfg)

    def test_mask_to_polygon_roundtrip(self):
        mask = rect_mask((80, 80), 20, 60, 10, 70)
        poly = mask_to_polygon(mask)
        rebuilt = polygon_to_mask(poly, mask.shape)
        self.assertGreater(((rebuilt > 0) & (mask > 0)).sum() / (mask > 0).sum(), 0.95)


class AnalysisMaskTests(unittest.TestCase):
    def test_erosion_subset_and_original_unchanged(self):
        road = rect_mask((60, 80), 20, 60, 10, 70)
        original = road.copy()
        for size in (3, 5):
            cfg = config(**{"analysis_mask.kernel_size": size, "analysis_mask.min_pixels": 10})["analysis_mask"]
            analysis, info = make_analysis_mask(road, cfg)
            self.assertTrue(np.array_equal(road, original))                    # road_mask 불변
            self.assertFalse(np.any((analysis > 0) & (road == 0)))             # ⊆ road_mask
            r = size // 2
            # 위 · 왼 · 오른 경계에서 r px 침식, 영상 아래 테두리는 침식하지 않음
            expected = rect_mask((60, 80), 20 + r, 60, 10 + r, 70 - r)
            self.assertTrue(np.array_equal(analysis, expected))
            self.assertTrue(info["sufficient"])

    def test_too_small_is_flagged(self):
        road = rect_mask((60, 80), 20, 24, 10, 70)
        cfg = config()["analysis_mask"]
        analysis, info = make_analysis_mask(road, cfg)
        self.assertFalse(info["sufficient"])
        self.assertEqual(np.count_nonzero(analysis), 0)


if __name__ == "__main__":
    unittest.main()
