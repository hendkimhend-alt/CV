"""
논문 반영 1단계 (관문) — 방향 일치 재기. Hessian 방향 지도(0단계)로 최종 후보마다
  dir_coh   후보를 따라 방향이 고른가: 후보 픽셀에 (w·cos2θ, w·sin2θ, w)를 놓고 가우시안(σ 3)으로 모은 국소 일치도의 후보 위 평균
  bg_align  배경 결과 같은가: 주변 고리(상자 2배 − 상자)에서 약한 기준 이상인 점들의 방향 합 → 고른 정도 R_bg × cos²(후보 방향 − 배경 방향)
  dir_combo dir_coh × (1 − bg_align)
θ = 선이 뻗는 방향 (방향 지도 + 90°) · 180° 대칭이라 2θ로 더함 (구조 텐서와 같은 계산, 3-2) · w = Hessian 선 점수
바탕 = 약한 기준 ×0.2 최종 후보(노면 단서 통과). 개발 세트만.
출력: outputs/analysis/direction_support/ (features.csv, direction_support.md)
"""
import csv, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continuation import CFG, CRACKS, CUE
from data import list_images
from detect import _components, _crack_mask, _to_gray, _touches_border, classify, shape_features, valley_score
from evaluate import _area, _inter, transform_gt
from filter_features import cut_table
from onroad_features import greedy
from paths import OUTPUT_DIR, RDD_DIR, imread
from preprocess import geometry_preprocess, validate_config
from roadcue import cue_values, maps, ref_stats, ring

OUT = OUTPUT_DIR / "analysis" / "direction_support"
FEATURES = ["dir_coh", "bg_align", "dir_combo"]
SIG = 3.0


def measure(pts, score, theta, lo, b, W, H):
    x1, y1, x2, y2 = b
    pad = int(3 * SIG) + 1
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    w = score[pts[:, 1], pts[:, 0]].astype(np.float32)
    t2 = 2 * theta[pts[:, 1], pts[:, 0]]
    C = np.zeros((Y2 - Y1, X2 - X1), np.float32); S = C.copy(); Wm = C.copy()
    yy, xx = pts[:, 1] - Y1, pts[:, 0] - X1
    C[yy, xx], S[yy, xx], Wm[yy, xx] = w * np.cos(t2), w * np.sin(t2), w
    Cb, Sb, Wb = (cv2.GaussianBlur(a, (0, 0), SIG) for a in (C, S, Wm))
    local = np.sqrt(Cb[yy, xx] ** 2 + Sb[yy, xx] ** 2) / np.maximum(Wb[yy, xx], 1e-6)
    dir_coh = float(local.mean())
    cand = np.arctan2((w * np.sin(t2)).sum(), (w * np.cos(t2)).sum())          # 후보 주 방향 (2θ)
    sl, keep = ring(b, W, H)
    rs, rt = score[sl][keep], 2 * theta[sl][keep]
    m = rs >= lo
    if m.sum() < 10:
        bg = 0.0
    else:
        bc, bs = (rs[m] * np.cos(rt[m])).sum(), (rs[m] * np.sin(rt[m])).sum()
        r_bg = float(np.hypot(bc, bs) / rs[m].sum())
        delta = 0.5 * (cand - np.arctan2(bs, bc))                             # θ 차이
        bg = r_bg * float(np.cos(delta) ** 2)
    return {"dir_coh": dir_coh, "bg_align": bg, "dir_combo": dir_coh * (1 - bg)}


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, images = list_images("rdd_dev")
    images = images[:limit] if limit else images
    pcfg = validate_config({"roi": "auto"})
    rows = []
    for i, path in enumerate(images):
        ann = json.loads((RDD_DIR / "ann" / f"{path.name}.json").read_text(encoding="utf-8"))
        objs = [(o["classTitle"], *o["points"]["exterior"][0], *o["points"]["exterior"][1]) for o in ann["objects"]]
        ref, geo = geometry_preprocess(imread(path), pcfg, path.name)
        boxes = [(o[0], min(o[1], o[3]), min(o[2], o[4]), max(o[1], o[3]), max(o[2], o[4])) for o in objs]
        kept = transform_gt(boxes, geo)
        if len(kept) != len(boxes):
            continue
        gts = [tuple(k[1:]) for o, k in zip(objs, kept) if o[0] in CRACKS]
        gray = _to_gray(ref)
        H, W = gray.shape
        smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        m = maps(ref); st = ref_stats(m, H, W)
        mask, score, hi, angle = _crack_mask(gray, CFG, with_angle=True)
        theta = angle + np.pi / 2                                              # 선이 뻗는 방향
        for pts in _components(mask, 0):
            d = shape_features(pts, score, hi)
            if classify(d, H * W, "crack", CFG) != "crack" or _touches_border(d["bbox"], W, H, CFG):
                continue
            if valley_score(smooth, pts, d["width"], CFG) < CFG["valley_min_ratio"]:
                continue
            x, y, w, h = d["bbox"]
            b = (x, y, x + w, y + h)
            cv = cue_values(m, st, b, W, H) or {}
            if any(k in cv and cv[k] > t for k, t in CUE.items()):
                continue
            real = any(_inter(b, g) >= 0.5 * max(_area(b), 1.0) for g in gts)
            f = measure(pts, score, theta, CFG["line_lo_ratio"] * hi, b, W, H)
            rows.append({"image": path.name, "real": int(real), **{k: round(float(f[k]), 4) for k in FEATURES},
                         "x1": x, "y1": y, "x2": x + w, "y2": y + h})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(images)}", flush=True)
    with open(OUT / "features.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    report(rows)


def report(rows):
    R = [r for r in rows if int(r["real"])]; Fk = [r for r in rows if not int(r["real"])]
    L = ["# 논문 반영 1단계 — 방향 일치 (개발 세트, ×0.2 최종 후보)", "",
         f"후보 {len(rows)} = 진짜 쪽 {len(R)} · 가짜 {len(Fk)}", "",
         "| 특징 | 진짜 중앙값 | 가짜 중앙값 | 버리는 쪽 | 진짜 5% 손실 → **가짜 버림** | 진짜 10% → 가짜 버림 |",
         "|---|---|---|---|---|---|"]
    for k in FEATURES:
        r = np.array([float(x[k]) for x in R]); f = np.array([float(x[k]) for x in Fk])
        t = cut_table(r, f)
        L.append(f"| {k} | {np.median(r):.3f} | {np.median(f):.3f} | {t['dir']} | "
                 f"{t[0.05][0]:.4f} → **{t[0.05][1] * 100:.0f}%** | {t[0.10][1] * 100:.0f}% |")
    L += ["", "하나씩 더하기 (진짜 5%씩 · 가짜 15% 이상 · 누적 진짜 손실 10% 이하)", "",
          "| 순서 | 특징 | 기준 | 이번에 버린 가짜 | 남은 진짜 | 남은 가짜 |", "|---|---|---|---|---|---|"]
    for n, (k, dr, thr, gain, nr, nf) in enumerate(greedy(
            [{k: float(x[k]) for k in FEATURES} for x in R], [{k: float(x[k]) for k in FEATURES} for x in Fk],
            feats=FEATURES), 1):
        L.append(f"| {n} | {k} | {dr} {thr:.4f} | {gain * 100:.0f}% | {nr} ({nr / len(R) * 100:.0f}%) | {nf} ({nf / len(Fk) * 100:.0f}%) |")
    md = "\n".join(L) + "\n"
    (OUT / "direction_support.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(list(csv.DictReader(open(OUT / "features.csv", encoding="utf-8"))))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
