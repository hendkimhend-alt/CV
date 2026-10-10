"""설정 검증 · 이미지 입출력 · 잘못된 입력 처리."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import ROOT, config  # noqa: F401  (src 경로 등록)
from preprocessing.config import ConfigError, apply_override, check_config, config_hash, load_config, strip_comments
from preprocessing.image_io import ImageLoadError, read_image, read_mask, validate_image, validate_mask, write_image


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self):
        cfg = load_config()
        self.assertEqual(cfg["roi"]["grid"]["cell_size"], 32)
        self.assertNotIn("_comment", cfg)

    def test_missing_section_rejected(self):
        raw = strip_comments(json.loads((ROOT / "configs" / "preprocessing.json").read_text(encoding="utf-8")))
        del raw["gamma"]
        with self.assertRaises(ConfigError):
            check_config(raw)

    def test_bad_values_rejected(self):
        with self.assertRaises(ConfigError):
            load_config(overrides=["analysis_mask.kernel_size=4"])   # 짝수 커널
        with self.assertRaises(ConfigError):
            load_config(overrides=["gaussian.mode=conditional", "gaussian.sigma=0"])
        with self.assertRaises(ConfigError):
            load_config(overrides=["gamma.selection=best"])
        with self.assertRaises(ConfigError):
            load_config(overrides=["gamma.valeu=0.9"])                # 없는 키

    def test_override_and_hash(self):
        a, b = load_config(), load_config(overrides=["gamma.value=0.9"])
        self.assertEqual(b["gamma"]["value"], 0.9)
        self.assertNotEqual(config_hash(a), config_hash(b))
        with self.assertRaises(ConfigError):
            apply_override({"gamma": {}}, "gamma.nothing=1")


class ImageIoTests(unittest.TestCase):
    def test_missing_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ImageLoadError):
                read_image(Path(tmp) / "없음.jpg")
            broken = Path(tmp) / "broken.jpg"
            broken.write_bytes(b"not an image")
            with self.assertRaises(ImageLoadError):
                read_image(broken)

    def test_validate_image_rejects_wrong_format(self):
        for bad in (np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32), np.zeros((0, 4, 3), np.uint8), None):
            with self.assertRaises(ImageLoadError):
                validate_image(bad)

    def test_roundtrip_korean_path_and_mask(self):
        img = np.random.default_rng(1).integers(0, 256, (20, 30, 3), dtype=np.uint8)
        mask = np.zeros((20, 30), np.uint8)
        mask[5:15, 5:25] = 255
        with tempfile.TemporaryDirectory() as tmp:
            path = write_image(Path(tmp) / "한글 폴더" / "영상.png", img)
            self.assertTrue(np.array_equal(read_image(path), img))
            mpath = write_image(Path(tmp) / "mask.png", mask)
            self.assertTrue(np.array_equal(read_mask(mpath, img.shape), mask))
            with self.assertRaises(ValueError):
                read_mask(mpath, (10, 10, 3))
        validate_mask(mask, img.shape)
        with self.assertRaises(ValueError):
            validate_mask(mask // 2, img.shape)


if __name__ == "__main__":
    unittest.main()
