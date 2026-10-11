"""
Harris 코너 밀도로 "거친 노면"을 가르는 기준값 정하기 + 진단 (개발 세트, 10/12).

기준값: 정답을 보지 않고, 개발 563장의 노면 화소에서 코너 밀도의 분위수(50 · 75%)로 정한다 → run_road_pipeline --extra harris_adapt의 thr
진단(예상용, 기준값 선택에는 안 씀): 질감 가짜(b_fakes.csv의 E) · 균열 정답 · 거북등 정답 박스 안의 평균 밀도 —
  질감 가짜는 높고 균열 정답은 낮아야 이 방법이 통할 수 있다. 거북등이 높으면 지워질 위험.

실행: python analysis/harris_density.py --cache <run_road_pipeline 캐시>

결과 (10/12): 실험 전 진단에서 중단 — 코너 밀도는 질감 가짜보다 균열 · 거북등 정답 박스에서 같거나 더 높다
  (상대 0.01: 노면 대부분 0 · 상대 0.001: 질감 0.011 / 균열 0.021 / 거북등 0.041 · 세기 평균: 0.95 / 2.88 / 3.46)
  → 균열은 가지 · 꺾임 · 끝점 · 교차가 코너이고, 남은 질감 가짜는 짧은 줄(1차원 구조 = 에지)이라 Harris 코너로는 질감을 못 가린다
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import run_road_pipeline as R  # noqa: E402
from data import RDD_DIR, list_images  # noqa: E402

WIN = 31


def density(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resp = cv2.cornerHarris(g, 2, 3, 0.04)
    return cv2.boxFilter((resp > 0.01 * resp.max()).astype(np.float32), -1, (WIN, WIN))


def to_det(b, geo):
    return [int((b[0] - geo["crop_x"]) * geo["scale_x"]), int((b[1] - geo["crop_y"]) * geo["scale_y"]),
            int((b[2] - geo["crop_x"]) * geo["scale_x"]), int((b[3] - geo["crop_y"]) * geo["scale_y"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    a = ap.parse_args()
    pre_cfg, det = R.load_config(R.PRE_CFG), R.load_detection_config(R.DET_CFG)
    _, _, images = list_images("rdd_dev")
    fakes = {}
    for f in csv.DictReader(open(ROOT / "results" / "analysis" / "b_fakes.csv", encoding="utf-8-sig")):
        if f["type"] == "crack" and f["kind"] == "E":
            fakes.setdefault(f["image"], []).append([float(f[k]) for k in ("x1", "y1", "x2", "y2")])
    rng = np.random.default_rng(20261012)
    sample, esample, diag = [], [], {"질감 가짜": [], "균열 정답 (거북등 제외)": [], "거북등 정답": []}
    for path in images:
        meta, img, mask, geo = R.prepare(path, pre_cfg, det, a.cache)
        if img is None:
            continue
        D = density(img)
        v = D[mask > 0]
        sample.append(rng.choice(v, min(2000, v.size), replace=False))
        resp = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), 2, 3, 0.04)
        e = cv2.boxFilter(np.maximum(resp, 0), -1, (WIN, WIN))
        e = e[mask > 0] / max(float(np.median(e[mask > 0])), 1e-9)
        esample.append(rng.choice(e, min(2000, e.size), replace=False))
        ann = RDD_DIR / "ann" / f"{path.name}.json"
        objs = json.loads(ann.read_text(encoding="utf-8"))["objects"] if ann.exists() else []
        boxes = []
        for o in objs:
            if "crack" not in o["classTitle"]:
                continue
            (ax, ay), (bx, by) = o["points"]["exterior"]
            k = "거북등 정답" if o["classTitle"] == "alligator crack" else "균열 정답 (거북등 제외)"
            boxes.append((k, [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]))
        boxes += [("질감 가짜", b) for b in fakes.get(path.name, [])]
        for k, b in boxes:
            x1, y1, x2, y2 = to_det(b, geo)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, D.shape[1]), min(y2, D.shape[0])
            if x2 > x1 and y2 > y1:
                diag[k].append(float(D[y1:y2, x1:x2].mean()))
    allv = np.concatenate(sample)
    ev = np.concatenate(esample)
    print("노면 코너 세기 (사진 노면 중앙값 대비) 분위수 — harris_adapt thr 후보, 정답 안 봄:",
          {p: round(float(np.quantile(ev, p / 100)), 3) for p in (50, 75, 90)})
    q = {p: float(np.quantile(allv, p / 100)) for p in (25, 50, 75, 90)}
    print("노면 코너 밀도 분위수 (기준값 후보, 정답 안 봄):", {k: round(v, 4) for k, v in q.items()})
    for k, v in diag.items():
        v = np.array(v)
        print(f"  {k:16} 박스 {len(v):4} · 평균 밀도 중앙값 {np.median(v):.4f} · 50% 기준 넘는 비율 {np.mean(v >= q[50]):.0%} · 75% 기준 {np.mean(v >= q[75]):.0%}")


if __name__ == "__main__":
    main()
