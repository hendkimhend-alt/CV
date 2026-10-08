"""src MSR의 수치·마스크 경계·기존 전처리 보존을 검증한다."""
import ast
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import preprocess
from metrics import measure_quality
from roi import geometry_mask


class MSRTests(unittest.TestCase):
    def setUp(self):
        self.cfg = preprocess.validate_config({"resize": {"long_side": 160}})
        rng = np.random.default_rng(8)
        self.image = rng.integers(0, 256, (101, 151, 3), dtype=np.uint8)
        self.ref, self.geom = preprocess.geometry_preprocess(self.image, self.cfg)
        self.mask = geometry_mask(self.geom)

    def test_black_uniform_and_disabled_no_nan_or_mutation(self):
        for value in (0, 127, 255):
            image = np.full_like(self.ref, value)
            original = image.copy()
            output = preprocess.apply_msr(image, roi_mask=self.mask)
            self.assertEqual(output.dtype, np.uint8)
            self.assertTrue(np.isfinite(output).all())
            np.testing.assert_array_equal(image, original)
            np.testing.assert_array_equal(output[self.mask != 0], image[self.mask != 0])
            self.assertFalse(output[self.mask == 0].any())
        output = preprocess.apply_msr(self.ref, {"enabled": False}, self.mask)
        np.testing.assert_array_equal(output, self.ref)
        self.assertIsNot(output, self.ref)

    def test_outside_pixels_do_not_affect_msr_or_quality(self):
        altered = self.ref.copy()
        altered[self.mask == 0] = (255, 12, 223)
        first = preprocess.apply_msr(self.ref, roi_mask=self.mask)
        second = preprocess.apply_msr(altered, roi_mask=self.mask)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(measure_quality(self.ref, self.mask), measure_quality(altered, self.mask))

    def test_direct_msr_matches_independent_single_scale_formula(self):
        cfg = preprocess.validate_msr_config({"scales": [3.0], "weights": [2.0], "blur_mode": "direct"})
        lab = cv2.cvtColor(self.ref, cv2.COLOR_BGR2LAB)
        light = lab[:, :, 0].astype(np.float32)
        valid = (self.mask != 0).astype(np.float32)
        numerator = cv2.GaussianBlur(light * valid, (0, 0), 3, borderType=cv2.BORDER_REFLECT_101)
        denominator = cv2.GaussianBlur(valid, (0, 0), 3, borderType=cv2.BORDER_REFLECT_101)
        result = np.log(light + 1) - np.log(numerator / np.maximum(denominator, 1e-8) + 1)
        low, high = np.percentile(result[self.mask != 0], [1, 99])
        lab[:, :, 0] = np.rint(np.clip((result - low) / (high - low) * 255, 0, 255)).astype(np.uint8)
        expected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        expected[self.mask == 0] = 0
        np.testing.assert_array_equal(preprocess.apply_msr(self.ref, cfg, self.mask), expected)

    def test_pyramid_surround_approximates_direct_and_keeps_constant(self):
        height, width = 160, 320
        y, x = np.mgrid[:height, :width]
        light = (30 + 150 * x / width + 20 * np.sin(y / 25)).astype(np.float32)
        valid = ((x > y * .5) & (x < width - y * .3)).astype(np.float32)
        cfg = preprocess.validate_msr_config({})
        for sigma in (15, 80, 250):
            pyramid = preprocess._msr_surround(light, valid, sigma, cfg)
            direct = preprocess._msr_surround(light, valid, sigma, {**cfg, "blur_mode": "direct"})
            self.assertLess(float(np.mean(np.abs(pyramid[valid != 0] - direct[valid != 0]))), 1.5)
            constant = preprocess._msr_surround(np.full_like(light, 80), valid, sigma, cfg)
            np.testing.assert_allclose(constant[valid != 0], 80, atol=1e-3)

    def test_settings_validation_and_normalization(self):
        self.assertEqual(preprocess.validate_msr_config({"weights": [2, 1, 1]})["weights"], [.5, .25, .25])
        for bad in ({"scales": []}, {"weights": [1]}, {"weights": [0, 0, 0]}, {"scales": [0, 1, 2]},
                    {"epsilon": 0}, {"output_percentiles": [99, 1]}, {"scales": [float('nan')]}, {"enabled": 1}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                preprocess.validate_msr_config(bad)

    def test_operators_unchanged_and_legacy_execution_preserved_in_comments(self):
        before = ast.parse((ROOT / "backups/before_fixedroi_msr/preprocess.py").read_text(encoding="utf-8"))
        source = (ROOT / "src/preprocess.py").read_text(encoding="utf-8")
        after = ast.parse(source)
        names = {n.name: n for n in after.body if isinstance(n, ast.FunctionDef)}
        commented = source.split("# LEGACY_PREPROCESS_BEGIN\n", 1)[1].split("# LEGACY_PREPROCESS_END", 1)[0]
        legacy = ast.parse("\n".join(line[2:] if line.startswith("# ") else "" for line in commented.splitlines()))
        legacy_names = {n.name: n for n in legacy.body if isinstance(n, ast.FunctionDef)}
        for node in before.body:
            if isinstance(node, ast.FunctionDef):
                current = legacy_names[node.name] if node.name in ("preprocess_condition", "preprocess") else names[node.name]
                self.assertEqual(ast.dump(node), ast.dump(current), node.name)
        before_config = next(n for n in before.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEFAULT_CONFIG" for t in n.targets))
        self.assertEqual(ast.literal_eval(before_config.value), preprocess.DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
