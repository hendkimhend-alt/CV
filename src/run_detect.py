"""
개발 B - 13장 일괄 검출 (전처리 없음 = P0 조건).

A의 preprocess()가 붙기 전까지 혼자 돌리는 용도. prepare()는 노면 ROI(하단 절반) +
긴 변 1024 리사이즈만 한다 — 커널·블록 크기를 실제 파이프라인과 같은 스케일로 맞추기 위함.

출력: outputs/detect/<D0|D1>/<name>.png, outputs/detect/summary.csv
실행: python src/run_detect.py
"""
import csv, glob, os, time
from collections import Counter

import cv2

from paths import OUTPUT_DIR, PROVIDED_DIR, imread

from detect import detect
from visualize import draw_detections, save

DATA_DIR = str(PROVIDED_DIR)
OUT_DIR = str(OUTPUT_DIR / "detect")
DETECTORS = ["D0", "D1"]
LONG_SIDE = 1024


def prepare(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = LONG_SIDE / max(h, w)
    gray = cv2.resize(gray, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return gray[gray.shape[0] // 2:, :]


def main():
    for d in DETECTORS:
        os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        img = prepare(imread(path))
        for d in DETECTORS:
            t0 = time.perf_counter()
            dets = detect(img, {"detector": d})
            ms = (time.perf_counter() - t0) * 1000
            n = Counter(det["type"] for det in dets)
            save(os.path.join(OUT_DIR, d, f"{name}.png"),
                 draw_detections(img, dets, f"{d} crack={n['crack']} pothole={n['pothole']}"))
            rows.append(dict(name=name, detector=d, n_crack=n["crack"], n_pothole=n["pothole"],
                             total_area=round(sum(det["area"] for det in dets)), time_ms=round(ms, 1)))
            print(f"{name:28s} {d}  crack={n['crack']:3d}  pothole={n['pothole']:3d}  {ms:6.1f}ms")

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
