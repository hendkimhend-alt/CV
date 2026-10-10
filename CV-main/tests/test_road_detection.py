"""도로 손상 검출 시스템 — 검출 설정 · 후보의 도로 위 판정 · 전처리→검출 연결 · 실행기."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import RDD_IMG, ROOT, config, synthetic_scene
from road_detection import config as det_config
from road_detection.config import DetectionConfigError, detector_cfg, load_detection_config, needs_ximgproc
from road_detection.pipeline import road_overlap, run_image

DEFAULT = ROOT / "configs" / "detection_default.json"
P2BD = ROOT / "configs" / "detection_p2bd.json"


class DetectionConfigTests(unittest.TestCase):
    def test_detector_name_flags(self):
        self.assertEqual(detector_cfg("D1"), {"detector": "D1", "valley_check": False})
        c = detector_cfg("D1hv", valley_ratio=0.35, line_hi_abs=128.6)
        self.assertEqual((c["detector"], c["valley_check"], c["crack_find"], c["line_hi_abs"], c["valley_min_ratio"]),
                         ("D1", True, "line", 128.6, 0.35))
        self.assertEqual(detector_cfg("D1vl")["link_dist"], 15)
        for bad in ("D2", "D1x", "D1vv"):
            with self.assertRaises(DetectionConfigError):
                detector_cfg(bad)

    def test_config_files(self):
        d = load_detection_config(DEFAULT)
        self.assertFalse(d["needs_ximgproc"])
        self.assertEqual(d["integration"]["input_mode"], "crop_to_road")
        p = load_detection_config(P2BD)
        self.assertTrue(p["needs_ximgproc"])                                    # Hessian 선 찾기 = 세선화
        self.assertEqual(p["detect_cfg"]["line_hi_abs"], 128.609375)            # 이전 p2bd 값 그대로
        self.assertTrue(needs_ximgproc({"guided_filter": {"r": 4}}))
        self.assertFalse(needs_ximgproc({"detector": "D1", "valley_check": True}))

    def test_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for spec in ({"detector": "D1", "oops": 1}, {"detector": "D1", "set": {"not_a_key": 1}},
                         {"detector": "D1", "integration": {"input_mode": "weird"}}, {"valley_ratio": 0.3}):
                p = Path(tmp) / "c.json"
                p.write_text(json.dumps(spec), encoding="utf-8")
                with self.assertRaises(DetectionConfigError):
                    load_detection_config(p)

    def test_missing_contrib_is_reported_before_running(self):
        p = load_detection_config(P2BD)
        with mock.patch.object(det_config, "cv2", object()):                   # cv2.ximgproc가 없는 환경을 흉내
            with self.assertRaises(DetectionConfigError):
                det_config.check_runtime(p)
            det_config.check_runtime(load_detection_config(DEFAULT))           # 기본 설정은 contrib 없이 실행 가능


class IntegrationTests(unittest.TestCase):
    def test_road_overlap(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[50:, :] = 255
        self.assertAlmostEqual(road_overlap(mask, (0, 25, 10, 50)), 0.5)
        self.assertEqual(road_overlap(mask, (0, 0, 10, 10)), 0.0)
        self.assertEqual(road_overlap(mask, (0, 80, 10, 10)), 1.0)
        self.assertEqual(road_overlap(mask, (200, 200, 5, 5)), 0.0)            # 영상 밖

    def test_end_to_end_on_synthetic_scene(self):
        img, regions = synthetic_scene(height=300, width=400)
        det = load_detection_config(DEFAULT)
        out = run_image(img, config(), det, image_id="syn")
        self.assertEqual(out["status"], "detected")
        meta = out["preprocess"].metadata["detection"]
        self.assertEqual(meta["detector"], "D1v")
        h, w = img.shape[:2]
        for c in out["candidates"]:
            x, y, bw, bh = c["bbox_original"]
            self.assertTrue(-1 <= x <= w and -1 <= y <= h and x + bw <= w + 1 and y + bh <= h + 1)
            self.assertEqual(c["on_road"], c["road_overlap"] >= det["integration"]["min_road_overlap"])
        self.assertEqual(meta["counts_on_road"]["crack"], sum(c["on_road"] and c["type"] == "crack" for c in out["candidates"]))

    def test_fail_mask_skips_or_uses_candidate(self):
        img, _ = synthetic_scene(height=300, width=400)
        strict = config(**{"validation.metrics.mask_area_ratio.min": 0.95})     # 자동 마스크를 일부러 FAIL
        det = load_detection_config(DEFAULT)
        self.assertEqual(det["integration"]["on_fail"], "use_candidate")          # 기본값 (2026-10-10 dev 진단)
        det["integration"]["on_fail"] = "skip"
        out = run_image(img, strict, det)
        self.assertEqual(out["status"], "skipped_manual_required")
        self.assertEqual(out["candidates"], [])
        det["integration"]["on_fail"] = "use_candidate"
        out = run_image(img, strict, det)
        self.assertEqual(out["status"], "detected")
        self.assertTrue(out["unverified_mask"])
        self.assertTrue(out["preprocess"].metadata["detection"]["unverified_mask"])
        manual = np.zeros(img.shape[:2], np.uint8)
        manual[120:, 60:] = 255
        det["integration"]["on_fail"] = "skip"
        out = run_image(img, strict, det, manual_mask=manual, manual_source="m.png")     # 수동 마스크가 있으면 진행
        self.assertEqual(out["status"], "detected")
        self.assertFalse(out["unverified_mask"])


@unittest.skipUnless(RDD_IMG.is_dir(), "RDD 데이터 없음")
class RunnerTests(unittest.TestCase):
    def test_cli_writes_outputs(self):
        import run_road_detection
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            code = run_road_detection.main(["--dataset", "rdd_dev", "--images", "Japan_000015.jpg", "China_Drone_000701.jpg", "--output", str(out)])
            self.assertEqual(code, 0)
            for name in ("summary.csv", "detections.csv", "run_config.json", "run_summary.json", "review_sheet.jpg"):
                self.assertTrue((out / name).is_file(), name)
            self.assertTrue((out / "images" / "Japan_000015" / "result.jpg").is_file())
            self.assertTrue((out / "images" / "Japan_000015" / "detections.json").is_file())
            meta = json.loads((out / "images" / "Japan_000015" / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("detection", meta)
            summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_images"], 2)

    def test_cli_blocks_rdd_test(self):
        import run_road_detection
        with self.assertRaises(SystemExit):
            run_road_detection.main(["--dataset", "rdd_test", "--limit", "1"])


if __name__ == "__main__":
    unittest.main()
