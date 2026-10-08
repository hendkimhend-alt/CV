"""src의 MSR 혼합·마스크 Gaussian 수치와 가는 선 손실을 검증한다."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import adaptive_preprocess as adaptive
from preprocess import apply_msr, geometry_preprocess, validate_config
from roi import geometry_mask


class StrengthTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(107)
        image = rng.integers(0, 256, (151, 211, 3), dtype=np.uint8)
        self.cfg = validate_config({"resize": {"long_side": 200}})
        self.ref, geom = geometry_preprocess(image, self.cfg)
        self.mask = geometry_mask(geom)
        self.msr = apply_msr(self.ref, roi_mask=self.mask)

    def test_alpha_endpoints_exact_and_no_mutation(self):
        before = self.ref.copy(), self.msr.copy()
        for alpha, expected in ((0, self.ref), (1, self.msr)):
            actual = adaptive.blend_msr_lightness(self.ref, self.msr, alpha, self.mask)
            np.testing.assert_array_equal(actual, expected)
            self.assertIsNot(actual, expected)
        np.testing.assert_array_equal(before[0], self.ref)
        np.testing.assert_array_equal(before[1], self.msr)

    def test_intermediate_alpha_independent_lab_formula_and_mask(self):
        for alpha in (.25, .5):
            lab = cv2.cvtColor(self.ref, cv2.COLOR_BGR2LAB)
            l = cv2.cvtColor(self.msr, cv2.COLOR_BGR2LAB)[:, :, 0]
            lab[:, :, 0] = np.rint(lab[:, :, 0].astype(np.float64) * (1 - alpha) + l.astype(np.float64) * alpha).astype(np.uint8)
            expected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            expected[self.mask == 0] = 0
            actual = adaptive.blend_msr_lightness(self.ref, self.msr, alpha, self.mask)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.uint8)
            self.assertFalse(actual[self.mask == 0].any())

    def test_masked_gaussian_preserves_constant_at_polygon_edges(self):
        image = np.full_like(self.ref, 100)
        for sigma in (.5, .8):
            actual = adaptive.masked_gaussian(image, self.mask, sigma)
            np.testing.assert_array_equal(actual[self.mask != 0], image[self.mask != 0])
            self.assertFalse(actual[self.mask == 0].any())

    def test_gaussian_outside_independence_and_zero_bypass(self):
        changed = self.ref.copy()
        changed[self.mask == 0] = 255
        for sigma in (.5, .8):
            np.testing.assert_array_equal(adaptive.masked_gaussian(changed, self.mask, sigma), adaptive.masked_gaussian(self.ref, self.mask, sigma))
        np.testing.assert_array_equal(adaptive.masked_gaussian(self.ref, self.mask, 0), self.ref)

    def test_gaussian_reduces_high_frequency_and_exposes_thin_line_loss(self):
        image = np.full((31, 31, 3), 120, np.uint8)
        image[:, 15] = 0
        mask = np.full(image.shape[:2], 255, np.uint8)
        smoothed = adaptive.masked_gaussian(image, mask, .8)
        self.assertGreater(int(smoothed[15, 15, 0]), 0)
        self.assertLess(float(smoothed.std()), float(image.std()))


if __name__ == "__main__":
    unittest.main()
