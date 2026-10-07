"""
노면 위 거르기 O1 — 노면 단서까지 통과한 최종 균열 후보마다 "노면 위에서 균열과 질감을 가르는" 특징을 재고,
진짜 쪽(정답 균열 박스 안 50% 이상) / 가짜로 나눠 비교한다. 바탕 = Hessian 약한 기준 ×0.3 · ×0.2. 개발 세트만.

특징 (filter_features와 겹치는 것은 그 함수를 그대로 씀)
  contrast  진하기: 후보 위 Hessian 점수 평균 ÷ 강한 기준                       (3-1 · 4-1 2차 미분)
  strong    강한 부분 비율: 후보 픽셀 중 점수 ≥ 강한 기준                         (3-1 Canny 이중 임계값)
  snr       두드러짐: 후보 위 점수 평균 ÷ 주변 고리 점수 상위 5%                  (2-1 · 3-1 국소 대비)
  depth     깊이: (고리 밝기 중앙값 − 후보 밝기 평균) ÷ 고리 사분위 범위          (2-1 히스토그램)
  streak    결 방향: 고리 구조 텐서 정렬도 × cos²(후보 방향 − 고리 결 방향)       (3-2 2차 모멘트 행렬)
  density   주변 흔적 밀도 · resid 직선 잔차 · paint 페인트 옆 · straight 곧음    (filter_features)
  valley    양쪽 확인 · length 길이
  ms3 · ms4 · ms6  여러 크기에서 유지: σ 3 · 4 · 6 단일 크기 선 점수(σ² 맞춤, 5×5 최댓값)의 흔적 평균
                   ÷ 찾기 점수(σ 1~2)의 흔적 평균 — 질감은 큰 σ에서 사라지고 균열은 남는다는 가설 (4-1 스케일 공간)
출력: outputs/analysis/onroad_features/ (features.csv, onroad_features.md)
"""
import csv, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_images
from detect import (DEFAULT_CFG, _components, _crack_mask, _to_gray, _touches_border, classify, line_score,
                    shape_features, valley_score)
from evaluate import _area, _inter, transform_gt
from filter_features import cut_table, features as base_features, paint_pixels, road_pixels
from paths import OUTPUT_DIR, RDD_DIR, imread
from preprocess import geometry_preprocess, validate_config
from roadcue import cue_values, maps, onroad_values, ref_stats, ring, tensor

OUT = OUTPUT_DIR / "analysis" / "onroad_features"
CUE = {"energy": 1.6082, "coherence": 0.2941, "color": 2.9964}
CFG = {**DEFAULT_CFG, "detector": "D1", "crack_find": "line", "line_hi_abs": 32.5, "line_thicken": 1,
       "crack_min_length": 45, "valley_check": True, "valley_min_ratio": 0.35}
BASES = (0.3, 0.2)
CRACKS = {"longitudinal crack", "transverse crack", "alligator crack"}
FEATURES = ["contrast", "strong", "snr", "depth", "streak", "density", "resid", "straight", "paint", "valley", "length",
            "ms3", "ms4", "ms6"]
BIG = (3.0, 4.0, 6.0)


def onroad(pts, d, score, hi, smooth, J, b, W, H):
    sl, keep = ring(b, W, H)
    on = score[pts[:, 1], pts[:, 0]]
    return {"contrast": d["contrast"], "strong": float((on >= hi).mean()),
            "snr": float(on.mean() / max(np.percentile(score[sl][keep], 95), 1e-3)),
            **onroad_values(pts, smooth, J, b, W, H)}                # depth · streak — 검출기와 같은 코드


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
        J = tensor(gray)
        road, paint = road_pixels(ref), paint_pixels(ref)
        big = {s: cv2.dilate(line_score(gray, (s,))[0], np.ones((5, 5), np.uint8)) for s in BIG}
        for lo in BASES:
            mask, score, hi = _crack_mask(gray, {**CFG, "line_lo_ratio": lo})
            trace = mask > 0
            for pts in _components(mask, 0):
                d = shape_features(pts, score, hi)
                if classify(d, H * W, "crack", CFG) != "crack" or _touches_border(d["bbox"], W, H, CFG):
                    continue
                v = valley_score(smooth, pts, d["width"], CFG)
                if v < CFG["valley_min_ratio"]:
                    continue
                x, y, w, h = d["bbox"]
                b = (x, y, x + w, y + h)
                cv = cue_values(m, st, b, W, H) or {}
                if any(k in cv and cv[k] > t for k, t in CUE.items()):
                    continue                                       # 노면 단서에서 버려짐 (검출기와 같은 순서)
                real = any(_inter(b, g) >= 0.5 * max(_area(b), 1.0) for g in gts)
                f = base_features(pts, d, gray, trace, road, paint, H, W)
                f.update(onroad(pts, d, score, hi, smooth, J, b, W, H))
                f["valley"] = v
                base = max(float(score[pts[:, 1], pts[:, 0]].mean()), 1e-3)
                for s_ in BIG:
                    f[f"ms{int(s_)}"] = float(big[s_][pts[:, 1], pts[:, 0]].mean()) / base
                rows.append({"base": lo, "image": path.name, "real": int(real),
                             **{k: round(float(f[k]), 4) for k in FEATURES}, "x1": x, "y1": y, "x2": x + w, "y2": y + h})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(images)}", flush=True)
    with open(OUT / "features.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    report(rows)


def greedy(R, Fk, cap=0.10):
    """규칙: 매번 남은 진짜 5% 손실 기준값 · 가짜 15% 이상 · 누적 진짜 손실 cap 이하."""
    keepR, keepF, used, steps = list(R), list(Fk), set(), []
    while True:
        best = None
        for k in FEATURES:
            if k in used:
                continue
            t = cut_table(np.array([x[k] for x in keepR]), np.array([x[k] for x in keepF]), (0.05,))
            if t[0.05][1] >= 0.15 and (best is None or t[0.05][1] > best[1][0.05][1]):
                best = (k, t)
        if best is None:
            break
        k, t = best
        thr = t[0.05][0]
        ok = (lambda v: v >= thr) if t["dir"] == "아래 버림" else (lambda v: v <= thr)
        nR = [x for x in keepR if ok(x[k])]; nF = [x for x in keepF if ok(x[k])]
        if 1 - len(nR) / len(R) > cap:
            break
        keepR, keepF = nR, nF; used.add(k)
        steps.append((k, t["dir"], thr, t[0.05][1], len(keepR), len(keepF)))
    return steps


def report(rows):
    L = ["# 노면 위 거르기 O1 — 최종 후보(노면 단서 통과)의 특징 (개발 세트)", ""]
    for lo in BASES:
        rs = [r for r in rows if float(r["base"]) == lo]
        R = [r for r in rs if int(r["real"])]; Fk = [r for r in rs if not int(r["real"])]
        L += [f"## 바탕 약한 기준 ×{lo} — 후보 {len(rs)} = 진짜 쪽 {len(R)} · 가짜 {len(Fk)} (진짜 비율 {len(R) / len(rs):.3f})", "",
              "| 특징 | 진짜 중앙값 | 가짜 중앙값 | AUC | 버리는 쪽 | 진짜 5% 손실 → **가짜 버림** | 진짜 10% → 가짜 버림 |",
              "|---|---|---|---|---|---|---|"]
        for k in FEATURES:
            r = np.array([float(x[k]) for x in R]); f = np.array([float(x[k]) for x in Fk])
            t = cut_table(r, f)
            L.append(f"| {k} | {np.median(r):.3f} | {np.median(f):.3f} | {t['auc']:.3f} | {t['dir']} | "
                     f"{t[0.05][0]:.3f} → **{t[0.05][1] * 100:.0f}%** | {t[0.10][1] * 100:.0f}% |")
        L += ["", "하나씩 더하기 (진짜 5%씩 · 가짜 15% 이상 · 누적 진짜 손실 10% 이하)", "",
              "| 순서 | 특징 | 기준 | 이번에 버린 가짜 | 남은 진짜 | 남은 가짜 | 진짜 비율 |", "|---|---|---|---|---|---|---|"]
        for n, (k, dr, thr, gain, nr, nf) in enumerate(greedy(
                [{k: float(x[k]) for k in FEATURES} for x in R], [{k: float(x[k]) for k in FEATURES} for x in Fk]), 1):
            L.append(f"| {n} | {k} | {dr} {thr:.4f} | {gain * 100:.0f}% | {nr} ({nr / len(R) * 100:.0f}%) | "
                     f"{nf} ({nf / len(Fk) * 100:.0f}%) | {nr / (nr + nf):.3f} |")
        L.append("")
    md = "\n".join(L) + "\n"
    (OUT / "onroad_features.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(list(csv.DictReader(open(OUT / "features.csv", encoding="utf-8"))))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
