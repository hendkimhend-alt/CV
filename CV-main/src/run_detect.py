"""검출 실행 진입점. FINAL 전처리와 공통 평가·저장을 사용한다.

실행: python src/run_detect.py --dataset rdd_dev --limit 5
출력: outputs/pipeline/run_<시각>/ (run_pipeline.py와 동일)
"""
import cv2

from adaptive_preprocess import adaptive_preprocess
from run_pipeline import main


def prepare(bgr, return_mask=False, config=None):
    """FINAL JSON 설정으로 전처리한 grayscale과 선택적으로 ROI 마스크를 반환한다."""
    fixed, metadata = adaptive_preprocess(bgr, config)
    gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    return (gray, metadata["roi_mask"]) if return_mask else gray


if __name__ == "__main__":
    main()


# 이전 P0 단독 검출 실행 — 주석으로 보존하며 실행하지 않는다.
# """
# 개발 B - 13장 일괄 검출 (전처리 없음 = P0 조건).
#
# A의 preprocess()가 붙기 전까지 혼자 돌리는 용도. prepare()는 공통 road_focus ROI +
# 긴 변 1024 리사이즈만 한다. 보정과 검출기 파라미터는 변경하지 않는다.
#
# 출력: outputs/detect/<D0|D1>/<name>.png, outputs/detect/summary.csv
# 실행: python src/run_detect.py
# """
# import csv, glob, os, time
# from collections import Counter
#
# import cv2
#
# from paths import OUTPUT_DIR, PROVIDED_DIR, imread
#
# from detect import detect
# from visualize import draw_detections, save
# from preprocess import geometry_preprocess, validate_config
# from roi import geometry_mask
#
# DATA_DIR = str(PROVIDED_DIR)
# OUT_DIR = str(OUTPUT_DIR / "detect")
# DETECTORS = ["D0", "D1"]
# LONG_SIDE = 1024
#
#
# def prepare(bgr, return_mask=False):
#     reference, geometry = geometry_preprocess(bgr, validate_config({"resize": {"long_side": LONG_SIDE}}))
#     gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
#     return (gray, geometry_mask(geometry)) if return_mask else gray
#
#
# def main():
#     for d in DETECTORS:
#         os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)
#
#     rows = []
#     for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg"))):
#         name = os.path.splitext(os.path.basename(path))[0]
#         img, roi_mask = prepare(imread(path), return_mask=True)
#         for d in DETECTORS:
#             t0 = time.perf_counter()
#             dets = detect(img, {"detector": d}, roi_mask=roi_mask)
#             ms = (time.perf_counter() - t0) * 1000
#             n = Counter(det["type"] for det in dets)
#             save(os.path.join(OUT_DIR, d, f"{name}.png"),
#                  draw_detections(img, dets, f"{d} crack={n['crack']} pothole={n['pothole']}"))
#             rows.append(dict(name=name, detector=d, n_crack=n["crack"], n_pothole=n["pothole"],
#                              total_area=round(sum(det["area"] for det in dets)), time_ms=round(ms, 1)))
#             print(f"{name:28s} {d}  crack={n['crack']:3d}  pothole={n['pothole']:3d}  {ms:6.1f}ms")
#
#     with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=rows[0].keys())
#         writer.writeheader()
#         writer.writerows(rows)
#
#
# if __name__ == "__main__":
#     main()
