"""
거르기 R1 — 노면 마스크 v2를 위한 단서별 재기 (개발 세트, 새 검출기 E9-b의 후보).
후보마다 "주변 고리"(상자를 2배로 넓힌 창 − 후보 상자)를 그 사진의 도로 표본(높이 60~80% · 너비 30~70%)과 비교.
값이 클수록 도로답지 않음.
  color      HSV 색상 · 채도 평면 좌표의 중앙값 거리 ÷ 표본 퍼짐                    (2-1 컬러 모델)
  bright     |명도 V 중앙값 차| ÷ 표본 사분위 범위                                  (2-1 히스토그램)
  energy     |log(구조 텐서 λ1+λ2 중앙값 비)| — 너무 매끈 · 너무 강함 둘 다           (3-1 · 3-2)
  coherence  정렬도 ((λ1−λ2)/(λ1+λ2))² 고리 평균 − 표본 평균                         (3-2)
  corner     log(λ2 중앙값 비) — 모서리                                              (3-2 Shi-Tomasi · 해리스)
  road_now   지금 도로다움 (filter_features.csv의 road, 비교용 — 작을수록 도로답지 않으므로 부호 반대)
입력: outputs/analysis/filter_features/features.csv (후보 상자 · 진짜 여부)
출력: outputs/analysis/road_cues/ (cues.csv, road_cues.md)
"""
import csv, os, sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_features import cut_table
from paths import OUTPUT_DIR, RDD_DIR, imread
from preprocess import geometry_preprocess, validate_config

SRC = OUTPUT_DIR / "analysis" / "filter_features" / "features.csv"
OUT = OUTPUT_DIR / "analysis" / "road_cues"
CUES = ["color", "bright", "energy", "coherence", "corner"]
from roadcue import cue_values, maps, ref_stats   # 검출기와 같은 코드 (src/roadcue.py)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image"]].append(r)
    pcfg = validate_config({"roi": "auto"})
    out = []
    for i, (name, rs) in enumerate(by_img.items()):
        ref, _ = geometry_preprocess(imread(RDD_DIR / "img" / name), pcfg, name)
        H, W = ref.shape[:2]
        m = maps(ref)
        st = ref_stats(m, H, W)
        for r in rs:
            b = tuple(int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
            cv = cue_values(m, st, b, W, H)
            if cv is None:
                continue
            out.append({"image": name, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "real": int(r["real"]),
                        "road_now": -float(r["road"]), **{k: round(v, 4) for k, v in cv.items()}})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(by_img)}", flush=True)
    with open(OUT / "cues.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    report(out)


def report(out, sample_labels=None):
    R = [o for o in out if o["real"]]; F = [o for o in out if not o["real"]]
    L = ["# 거르기 R1 — 노면 단서별 (개발 세트, E9-b 후보)", "", f"후보 {len(out)} = 진짜 쪽 {len(R)} · 가짜 {len(F)}", "",
         "값이 클수록 도로답지 않음 (road_now는 지금 도로다움에 − 를 붙인 것)", "",
         "| 단서 | 진짜 중앙값 | 가짜 중앙값 | AUC (가짜가 클수록 > 0.5) | 진짜 5% 잃을 때 **가짜 버림** | 진짜 10% → 가짜 버림 |",
         "|---|---|---|---|---|---|"]
    th = {}
    for k in ["road_now"] + CUES:
        r = np.array([o[k] for o in R]); f = np.array([o[k] for o in F])
        t = cut_table(r, f)
        th[k] = (t["dir"], t[0.05][0])
        L.append(f"| {k} | {np.median(r):.3f} | {np.median(f):.3f} | {1 - t['auc']:.3f} | {t[0.05][1] * 100:.0f}% ({t['dir']} {t[0.05][0]:.3f}) | {t[0.10][1] * 100:.0f}% |")
    md = "\n".join(L) + "\n"
    (OUT / "road_cues.md").write_text(md, encoding="utf-8")
    print(md)
    return th


if __name__ == "__main__":
    main()
