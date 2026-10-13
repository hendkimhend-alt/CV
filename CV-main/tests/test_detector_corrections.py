"""검출기 입력 보정(preprocessing/detector_corrections.py) — 노면 마스크 안에만 · 흐림 판단 · 설정 확인."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.detector_corrections import apply_corrections, judge_tags, validate  # noqa: E402

FINAL = {"gaussian": {"ksize": 3, "sigma": 0.8, "apply_to": "not_blur_rt", "blur_thr": 2.76},
         "line_close": {"length": 7, "angles": 12, "apply_to": "not_blur_rt"},
         "unsharp": {"amount": 3.0, "sigma": 3.0, "apply_to": "blur_rt"},
         "clahe": {"clip": 1.0, "tile": 8}, "order": ["gaussian", "line_close", "unsharp", "clahe"]}
GEO = {"original_width": 256, "original_height": 256, "scale_x": 4.0}     # 1024 척도로 되돌릴 때 배율 1


class DetectorCorrectionsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.sharp = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)       # 거친 무늬 = 선명
        self.flat = np.full((256, 256, 3), 120, np.uint8)                      # 무늬 없음 = 아주 흐림
        self.mask = np.zeros((256, 256), np.uint8)
        self.mask[64:192, 64:192] = 255

    def test_only_inside_mask(self):
        out, applied = apply_corrections(self.sharp, self.mask, FINAL, {"not_blur_rt"})
        self.assertTrue(applied)
        np.testing.assert_array_equal(out[self.mask == 0], self.sharp[self.mask == 0])
        self.assertFalse(np.array_equal(out[self.mask > 0], self.sharp[self.mask > 0]))

    def test_blur_judgement(self):
        self.assertIn("not_blur_rt", judge_tags(self.sharp, self.mask, GEO, FINAL))
        self.assertIn("blur_rt", judge_tags(self.flat, self.mask, GEO, FINAL))
        self.assertEqual(judge_tags(self.sharp, self.mask, GEO, {"clahe": {"clip": 1.0}}), set())

    def test_validate(self):
        validate(FINAL)
        with self.assertRaises(ValueError):
            validate({"sharpen": {}})


if __name__ == "__main__":
    unittest.main()
