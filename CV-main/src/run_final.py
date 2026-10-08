"""동결 설정으로 FINAL 전처리→고정 D1 검출. GT 없이도 실행할 수 있다."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from adaptive_preprocess import adaptive_preprocess, load_adaptive_config
from data import list_images
from detect import DEFAULT_CFG, detect
from paths import OUTPUT_DIR, imread, imwrite
from visualize import draw_detections


def run(dataset="provided", limit=None, output=None, config_path=None, save_images=5):
    # test는 사용자 최종 평가 요청 전까지 이 실행 경로에서도 차단한다.
    if dataset in ("rdd", "rdd_test"):
        raise ValueError("현재는 rdd_dev/provided/captured만 허용합니다. test는 별도 요청 후 평가합니다.")
    if limit is not None and limit < 1:
        raise ValueError("limit은 1 이상이어야 합니다")
    cfg = load_adaptive_config(config_path)
    name, _, paths = list_images(dataset)
    paths = paths[:limit] if limit is not None else paths
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    out = Path(output) if output else OUTPUT_DIR / "final_runtime" / stamp
    if out.exists() and any(out.iterdir()):
        raise ValueError("비어 있는 새 출력 폴더가 필요합니다")
    out.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    for number, path in enumerate(paths, 1):
        image = imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        fixed, meta = adaptive_preprocess(image, cfg)
        start = time.perf_counter()
        dets = detect(fixed, roi_mask=meta["roi_mask"])
        detect_ms = (time.perf_counter() - start) * 1000
        rows.append({"image": path.name, "dataset": name, "condition": "FINAL", "detector": "D1",
                     "msr_applied": meta["msr_applied"], "msr_alpha": meta["msr_alpha"],
                     "gamma_applied": meta["gamma_applied"], "gamma": meta["gamma"],
                     "gaussian_applied": meta["gaussian_applied"], "gaussian_sigma": meta["gaussian_sigma"],
                     "preprocess_ms": meta["preprocess_ms"], "detect_ms": detect_ms,
                     "total_ms": meta["preprocess_ms"] + detect_ms,
                     "total_candidates": len(dets), "quality_before_json": json.dumps(meta["quality_before"]),
                     **meta["quality_after"]})
        predictions.extend({"image": path.name, "type": d["type"], "bbox_xywh": json.dumps(d["bbox"])} for d in dets)
        if save_images < 0 or number <= save_images:
            imwrite(out / "images" / (path.stem + ".png"), draw_detections(fixed, dets, "FINAL D1"))
        if number == 1 or number % 25 == 0 or number == len(paths):
            print(f"[{number}/{len(paths)}] {path.name}", flush=True)
    for filename, values, fields in (("results.csv", rows, list(rows[0]) if rows else ["image"]),
                                      ("detections.csv", predictions, ["image", "type", "bbox_xywh"])):
        with (out / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    (out / "config.json").write_text(json.dumps({"runtime_config": cfg, "detector": DEFAULT_CFG,
          "detector_sha256": hashlib.sha256(Path(__file__).with_name("detect.py").read_bytes()).hexdigest(),
          "GT_loaded": False, "test_executed": False, "n_images": len(paths)}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("완료:", out)
    return out


def main():
    parser = argparse.ArgumentParser(description="GT 없는 FINAL 전처리 및 고정 D1 검출")
    parser.add_argument("--dataset", choices=("provided", "captured", "rdd_dev"), default="provided")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--save-images", type=int, default=5)
    args = parser.parse_args()
    run(args.dataset, args.limit, args.output, args.config, args.save_images)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
