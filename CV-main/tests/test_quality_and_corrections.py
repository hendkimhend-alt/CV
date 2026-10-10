"""품질 지표 공식 · 마스크 밖 화소 배제 · Gamma 공식/판단 · Gaussian ON/OFF · 마스크 밖 화소 보존."""
import math
import unittest

import cv2
import numpy as np

from helpers import config
from preprocessing.gamma import apply_gamma, choose_gamma, decide_gamma, gamma_lut, guard_triggered
from preprocessing.gaussian import decide_gaussian, masked_gaussian
from preprocessing.quality import measure_in_masks, measure_quality


def gray_bgr(gray):
    return np.repeat(np.asarray(gray, np.uint8)[..., None], 3, axis=2)


class QualityTests(unittest.TestCase):
    def test_brightness_formulas_inside_mask_only(self):
        gray = np.full((40, 40), 200, np.uint8)
        gray[20:, :] = np.tile(np.array([0, 30, 100, 255], np.uint8), (20, 10))   # 측정 영역 값 4종
        mask = np.zeros((40, 40), np.uint8)
        mask[20:, :] = 255
        values = measure_quality(gray_bgr(gray), mask, dark_threshold=40, bright_threshold=215)
        pixels = np.array([0, 30, 100, 255], np.float64)
        self.assertAlmostEqual(values["gray_mean"], pixels.mean())
        self.assertAlmostEqual(values["gray_std"], pixels.std())
        self.assertAlmostEqual(values["dark_ratio"], 0.5)          # 0, 30 < 40
        self.assertAlmostEqual(values["bright_ratio"], 0.25)       # 255 > 215
        self.assertAlmostEqual(values["saturation_ratio"], 0.5)    # 0, 255

    def test_outside_pixels_do_not_change_any_metric(self):
        rng = np.random.default_rng(3)
        img = rng.integers(60, 140, (64, 64, 3), dtype=np.uint8)
        road = np.zeros((64, 64), np.uint8)
        road[24:, 8:56] = 255
        analysis = cv2.erode(road, np.ones((5, 5), np.uint8))
        support = cv2.erode(road, np.ones((3, 3), np.uint8))
        a = measure_quality(img, analysis, support_mask=support)
        changed = img.copy()
        changed[road == 0] = rng.integers(0, 256, (int((road == 0).sum()), 3), dtype=np.uint8)
        b = measure_quality(changed, analysis, support_mask=support)
        self.assertEqual(a, b)

    def test_block_std_and_noise_definitions(self):
        gray = np.zeros((40, 40), np.uint8)
        gray[:, 20:] = 100                          # 왼쪽 0 · 오른쪽 100
        mask = np.full((40, 40), 255, np.uint8)
        values = measure_quality(gray_bgr(gray), mask, block_grid=4)
        self.assertAlmostEqual(values["block_mean_std_4x4"], 50.0)   # 블록 평균 0 · 100 각 8개
        flat = gray_bgr(np.full((30, 30), 120, np.uint8))
        self.assertEqual(measure_quality(flat, np.full((30, 30), 255, np.uint8))["noise_sigma"], 0.0)
        noisy = np.clip(120 + np.random.default_rng(0).normal(0, 5, (200, 200)), 0, 255).astype(np.uint8)
        sigma = measure_quality(gray_bgr(noisy), np.full((200, 200), 255, np.uint8))["noise_sigma"]
        self.assertTrue(3.5 < sigma < 6.0, sigma)   # Immerkær 추정치가 실제 σ=5 근처

    def test_full_image_matches_legacy_formula(self):
        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
        values = measure_quality(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.assertEqual(values["gray_mean"], float(gray.mean()))
        lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=1, borderType=cv2.BORDER_REFLECT_101)
        self.assertEqual(values["laplacian_variance"], float(lap.var()))
        response = cv2.filter2D(gray, cv2.CV_64F, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64))
        expected = math.sqrt(math.pi / 2) * np.abs(response[1:-1, 1:-1]).sum() / (6 * 51 * 35)
        self.assertAlmostEqual(values["noise_sigma"], expected, places=9)

    def test_empty_mask_gives_none(self):
        values = measure_quality(gray_bgr(np.full((10, 10), 50, np.uint8)), np.zeros((10, 10), np.uint8))
        self.assertTrue(all(v is None for v in values.values()))

    def test_measurement_scale_is_recorded(self):
        img = gray_bgr(np.full((300, 600), 90, np.uint8))
        mask = np.full((300, 600), 255, np.uint8)
        values, info = measure_in_masks(img, mask, mask, config()["quality"])
        self.assertEqual(info["measured_size"], [1024, 512])
        self.assertAlmostEqual(values["gray_mean"], 90.0)


class GammaTests(unittest.TestCase):
    def test_formula_and_identity(self):
        lut = gamma_lut(0.8)
        for value in (0, 1, 64, 128, 200, 255):
            self.assertEqual(lut[value], int(np.rint(255 * (value / 255) ** 0.8)))
        self.assertTrue(np.array_equal(gamma_lut(1.0), np.arange(256)))
        self.assertGreater(int(gamma_lut(0.7)[100]), 100)      # γ<1 → 밝아짐

    def test_only_road_pixels_change(self):
        img = gray_bgr(np.full((20, 20), 60, np.uint8))
        mask = np.zeros((20, 20), np.uint8)
        mask[10:, :] = 255
        out = apply_gamma(img, 0.8, mask)
        self.assertTrue(np.array_equal(out[:10], img[:10]))
        self.assertTrue(np.all(out[10:] == gamma_lut(0.8)[60]))

    def test_decision_logic(self):
        cfg = config()["gamma"]
        dark = {"gray_mean": 60.0, "dark_ratio": 0.2}
        only_mean = {"gray_mean": 60.0, "dark_ratio": 0.01}
        self.assertTrue(decide_gamma(dark, cfg, True)["apply"])
        self.assertFalse(decide_gamma(only_mean, cfg, True)["apply"])                 # AND
        cfg_or = config(**{"gamma.conditions.logic": "OR"})["gamma"]
        self.assertTrue(decide_gamma(only_mean, cfg_or, True)["apply"])               # OR
        self.assertEqual(decide_gamma({"gray_mean": None, "dark_ratio": 0.2}, cfg, True)["reason"], "metric_unavailable")
        self.assertEqual(decide_gamma(dark, cfg, False)["reason"], "insufficient_analysis_pixels")
        self.assertFalse(decide_gamma(dark, config(**{"gamma.mode": "off"})["gamma"], True)["apply"])
        self.assertTrue(decide_gamma({"gray_mean": 200.0, "dark_ratio": 0.0},
                                     config(**{"gamma.mode": "fixed"})["gamma"], False)["apply"])
        self.assertFalse(decide_gamma(dark, config(**{"gamma.selection": "fixed", "gamma.value": 1.0})["gamma"], True)["apply"])
        self.assertFalse(decide_gamma(dark, config(**{"gamma.candidates": [1.0]})["gamma"], True)["apply"])   # adaptive, 1 미만 후보 없음

    def test_adaptive_picks_weakest_gamma_that_resolves(self):
        cfg = config()["gamma"]                                       # adaptive · candidates 1.0/0.9/0.8/0.7
        self.assertEqual(cfg["selection"], "adaptive")

        def measure(img):
            g = img[..., 0].astype(float)
            return {"gray_mean": g.mean(), "dark_ratio": (g < 40).mean(), "saturation_ratio": (g >= 250).mean()}

        gray = np.full((20, 20), 110, np.uint8)
        gray[:10] = 30                                                # 평균 70 · 어두운 화소 50%
        img = gray_bgr(gray)
        g, out, after, tried = choose_gamma(img, None, measure(img), cfg, measure)
        self.assertEqual(g, 0.8)                                      # 0.9: 30→37 아직 어두움, 0.8: 30→46
        self.assertEqual([t["value"] for t in tried], [0.9, 0.8])
        self.assertTrue(tried[-1]["resolved"] and not tried[0]["resolved"])
        self.assertTrue(np.array_equal(out, apply_gamma(img, 0.8)))
        very_dark = gray_bgr(np.full((20, 20), 10, np.uint8))
        g, _, _, tried = choose_gamma(very_dark, None, measure(very_dark), cfg, measure)
        self.assertEqual(g, 0.7)                                      # 끝까지 안 풀림 → 가장 강한 후보
        self.assertFalse(any(t["resolved"] for t in tried))
        saturating = lambda im: {"gray_mean": 50.0, "dark_ratio": 0.2, "saturation_ratio": 0.5}
        g, out, _, tried = choose_gamma(img, None, {"saturation_ratio": 0.0}, cfg, saturating)
        self.assertIsNone(g)                                          # 첫 후보부터 포화 → 되돌림, 더 강한 후보는 안 봄
        self.assertIsNone(out)
        self.assertEqual(len(tried), 1)

    def test_saturation_guard(self):
        cfg = config()["gamma"]
        self.assertTrue(guard_triggered({"saturation_ratio": 0.0}, {"saturation_ratio": 0.05}, cfg))
        self.assertFalse(guard_triggered({"saturation_ratio": 0.0}, {"saturation_ratio": 0.01}, cfg))
        off = config(**{"gamma.guard.max_saturation_ratio_after": None})["gamma"]
        self.assertFalse(guard_triggered({"saturation_ratio": 0.0}, {"saturation_ratio": 0.5}, off))


class GaussianTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.img = rng.integers(80, 120, (40, 40, 3), dtype=np.uint8)
        self.img[:, :15] = 255                         # 밝은 비도로
        self.mask = np.zeros((40, 40), np.uint8)
        self.mask[:, 15:] = 255

    def test_outside_unchanged_inside_smoothed(self):
        out = masked_gaussian(self.img, self.mask, 5, 1.0)
        self.assertTrue(np.array_equal(out[:, :15], self.img[:, :15]))
        self.assertLess(out[:, 20:].astype(float).std(), self.img[:, 20:].astype(float).std())

    def test_mask_aware_prevents_boundary_contamination(self):
        flat = self.img.copy()
        flat[:, 15:] = 100
        aware = masked_gaussian(flat, self.mask, 5, 1.0, mask_aware=True)
        plain = masked_gaussian(flat, self.mask, 5, 1.0, mask_aware=False)
        self.assertTrue(np.all(aware[:, 15:] == 100))
        self.assertGreater(int(plain[:, 15].min()), 100)   # 비도로 255가 섞임

    def test_default_off(self):
        self.assertEqual(config()["gaussian"]["mode"], "off")                           # v1.1: dev 진단으로 꺼 둠
        self.assertFalse(decide_gaussian({"noise_sigma": 3.0}, config()["gaussian"], True)["apply"])

    def test_decision(self):
        cfg = config(**{"gaussian.mode": "conditional"})["gaussian"]
        self.assertTrue(decide_gaussian({"noise_sigma": 0.6}, cfg, True)["apply"])     # ≥ 경계 포함
        self.assertFalse(decide_gaussian({"noise_sigma": 0.59}, cfg, True)["apply"])
        self.assertFalse(decide_gaussian({"noise_sigma": None}, cfg, True)["apply"])
        self.assertFalse(decide_gaussian({"noise_sigma": 3.0}, config(**{"gaussian.mode": "off"})["gaussian"], True)["apply"])
        self.assertTrue(decide_gaussian({"noise_sigma": 0.0}, config(**{"gaussian.mode": "fixed"})["gaussian"], True)["apply"])
        # Laplacian 분산이 커도 noise_sigma가 낮으면 적용 안 함
        self.assertFalse(decide_gaussian({"noise_sigma": 0.1, "laplacian_variance": 1e6}, cfg, True)["apply"])


if __name__ == "__main__":
    unittest.main()
