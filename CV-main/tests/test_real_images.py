"""실제 개발 이미지(rdd_dev) 몇 장으로 하는 작은 통합 확인 — 데이터가 없으면 건너뛴다.

통과해도 도로 분할이 의미상 맞다는 뜻은 아니다 (형식 · 규약 · 기존 측정값 호환만 확인). 육안 검토는 review_sheet.jpg.
"""
import unittest

import numpy as np

from helpers import RDD_IMG, config
from preprocessing.image_io import read_image
from preprocessing.pipeline import MANUAL_REQUIRED, SUCCESS, process_image

# 근접(드론 · 오토바이) · 전방(일본 · 미국) · 강한 그림자(인도) · 다른 해상도(노르웨이 3650px)
SAMPLES = ("China_Drone_000035.jpg", "China_MotorBike_000083.jpg", "Japan_000015.jpg", "United_States_000269.jpg",
           "India_002022.jpg", "Norway_000169.jpg")


@unittest.skipUnless(RDD_IMG.is_dir(), "RDD 데이터 없음")
class RealImageTests(unittest.TestCase):
    def test_pipeline_contract_on_dev_images(self):
        cfg = config()
        for name in SAMPLES:
            with self.subTest(image=name):
                img = read_image(RDD_IMG / name)
                result = process_image(img, cfg, image_id=name)
                self.assertIn(result.status, (SUCCESS, MANUAL_REQUIRED), result.metadata["errors"])
                self.assertEqual(result.auto_mask.shape, img.shape[:2])
                if result.status == SUCCESS:
                    road = result.road_mask
                    self.assertEqual(result.processed_image.shape, img.shape)
                    self.assertTrue(np.array_equal(result.processed_image[road == 0], img[road == 0]))
                    self.assertFalse(np.any((result.analysis_mask > 0) & (road == 0)))

    def test_full_image_quality_on_real_image(self):
        from preprocessing.quality import measure_quality
        img = read_image(RDD_IMG / "Japan_000015.jpg")
        values = measure_quality(img)
        self.assertEqual(len(values), 8)
        self.assertAlmostEqual(values["gray_mean"], float(img.dot([0.114, 0.587, 0.299]).mean()), delta=0.6)


if __name__ == "__main__":
    unittest.main()
