"""
거르기 F1 — 새 검출기(E9-b: Hessian 32.5 · 1px · 길이 ≥ 45 · 양쪽 확인 0.35)의 최종 균열 후보마다 특징을 재고,
진짜 쪽(정답 균열 박스 안 50% 이상) / 가짜로 나눠 "어떤 특징이 둘을 가르나"를 본다. 개발 세트만.

특징 (찾기가 보지 않은 성질 위주)
  road      노면 안: 후보 상자 주변(8px 넓힘)에서 "도로다움"(ROI auto와 같은 색 · 질감 · 밝기 판정) 비율
  density   주변이 빽빽함: 상자를 2배로 넓힌 창에서 Hessian 흔적 밀도 (후보 자신 제외)
  straight  곧음: 두 끝점 사이 거리 ÷ 흔적 길이(픽셀 수) — 1 = 직선
  resid     직선에서 벗어난 정도: 직선 맞춤 잔차 RMS (px)
  paint     페인트 옆: 양옆(4px) 표본 중 흰색 · 노란색 비율
  length    길이 (회전 사각형 긴 변)
  valley    양쪽 확인 값
  contrast  진하기: 후보 위 Hessian 점수 평균 ÷ 강한 기준
  ypos      세로 위치 (0 위 ~ 1 아래) — 참고
출력: outputs/analysis/filter_features/ (features.csv, filter_features.md)
"""
import csv, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_images
from detect import (DEFAULT_CFG, _components, _crack_mask, _to_gray, _touches_border, classify, line_trace,
                    shape_features, valley_score)
from evaluate import _area, _inter, transform_gt
from paths import OUTPUT_DIR, RDD_DIR, imread
from preprocess import geometry_preprocess, validate_config

OUT = OUTPUT_DIR / "analysis" / "filter_features"
CFG = {**DEFAULT_CFG, "detector": "D1", "crack_find": "line", "line_hi_abs": 32.5, "line_thicken": 1,
       "crack_min_length": 45, "valley_check": True, "valley_min_ratio": 0.35}
CRACKS = {"longitudinal crack", "transverse crack", "alligator crack"}
NB = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int16)
FEATURES = ["road", "density", "straight", "resid", "paint", "length", "valley", "contrast", "ypos"]


def road_pixels(ref):
    """ROI auto(src/roi.py)와 같은 도로다움 판정을 픽셀마다 — 넓히지 않음."""
    lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    H, W = L.shape
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(L, cv2.CV_32F, ksize=3)), (0, 0), 3)
    r = (slice(int(.60 * H), int(.80 * H)), slice(int(.30 * W), int(.70 * W)))
    ma, mb = np.median(A[r]), np.median(B[r])
    sa, sb = max(float(A[r].std()), 3.0), max(float(B[r].std()), 3.0)
    mL, sL, mt = np.median(L[r]), float(L[r].std()), np.median(tex[r])
    return (np.sqrt(((A - ma) / sa) ** 2 + ((B - mb) / sb) ** 2) < 3) & (tex > 0.35 * mt) & (L < mL + max(3 * sL, 45))


def paint_pixels(ref):
    hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((v > 200) & (s < 40)) | ((h >= 15) & (h <= 35) & (s > 80) & (v > 120))


def window(b, W, H, grow):
    x1, y1, x2, y2 = b
    if grow >= 1:                                    # 배수로 넓힘 (가운데 기준)
        cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * grow / 2 + 4, (y2 - y1) * grow / 2 + 4
        x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
    else:
        g = int(-grow)
        x1, y1, x2, y2 = x1 - g, y1 - g, x2 + g, y2 + g
    return max(0, int(x1)), max(0, int(y1)), min(W, int(np.ceil(x2))), min(H, int(np.ceil(y2)))


def features(pts, d, gray, trace, road, paint, H, W):
    x, y, w, h = d["bbox"]
    b = (x, y, x + w, y + h)
    a1, b1, a2, b2 = window(b, W, H, -8)
    f = {"road": float(road[b1:b2, a1:a2].mean())}
    a1, b1, a2, b2 = window(b, W, H, 2.0)
    own = np.zeros((b2 - b1, a2 - a1), bool)
    inn = (pts[:, 0] >= a1) & (pts[:, 0] < a2) & (pts[:, 1] >= b1) & (pts[:, 1] < b2)
    own[pts[inn, 1] - b1, pts[inn, 0] - a1] = True
    f["density"] = float((trace[b1:b2, a1:a2] & ~own).mean())
    # 끝점 (이웃 1개) — 없으면(고리) 가장 먼 두 점
    m = np.zeros((h + 2, w + 2), np.uint8)
    m[pts[:, 1] - y + 1, pts[:, 0] - x + 1] = 1
    nb = cv2.filter2D(m, cv2.CV_16S, NB, borderType=cv2.BORDER_CONSTANT)
    ends = np.argwhere((m == 1) & (nb == 1))
    cand = ends if len(ends) >= 2 else np.argwhere(m == 1)
    if len(cand) > 60:
        cand = cand[np.linspace(0, len(cand) - 1, 60).astype(int)]
    dd = np.sqrt(((cand[:, None, :] - cand[None, :, :]) ** 2).sum(-1))
    f["straight"] = float(dd.max() / max(len(pts), 1))
    c = pts.astype(np.float64) - pts.mean(0)
    _, sv, _ = np.linalg.svd(c, full_matrices=False)
    f["resid"] = float(sv[-1] / np.sqrt(len(pts)))
    # 페인트 옆: 무작위 위치 20곳에서 선에 직각인 양옆 4px
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=min(20, len(pts)), replace=False)
    hit = []
    for px, py in pts[idx]:
        near = pts[((pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2) <= 49].astype(np.float64)
        if len(near) < 3:
            continue
        _, vecs = np.linalg.eigh(np.cov((near - near.mean(0)).T))
        nx, ny = -vecs[1, 1], vecs[0, 1]
        side = []
        for s in (1, -1):
            sx, sy = int(round(px + s * 4 * nx)), int(round(py + s * 4 * ny))
            side.append(0 <= sx < W and 0 <= sy < H and paint[sy, sx])
        hit.append(any(side))
    f["paint"] = float(np.mean(hit)) if hit else 0.0
    f["length"] = d["length"]
    f["contrast"] = d["contrast"]
    f["ypos"] = (y + h / 2) / H
    return f


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
    # 같은 값은 평균 순위
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cut_table(real, fake, losses=(0.05, 0.10)):
    """진짜 쪽이 큰 값 쪽(AUC ≥ 0.5)이면 '작은 값 버리기', 아니면 '큰 값 버리기'.
    진짜 손실이 loss가 되는 기준값에서 가짜를 몇 % 버리나."""
    a = auc(real, fake)
    out = {"auc": a, "dir": "아래 버림" if a >= 0.5 else "위 버림"}
    for loss in losses:
        if a >= 0.5:
            t = np.quantile(real, loss)
            out[loss] = (t, float((fake < t).mean()))
        else:
            t = np.quantile(real, 1 - loss)
            out[loss] = (t, float((fake > t).mean()))
    return out


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
        mask, score, hi = _crack_mask(gray, CFG)
        trace = mask > 0
        road, paint = road_pixels(ref), paint_pixels(ref)
        smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        for pts in _components(mask, 0):
            d = shape_features(pts, score, hi)
            if classify(d, H * W, "crack", CFG) != "crack" or _touches_border(d["bbox"], W, H, CFG):
                continue
            v = valley_score(smooth, pts, d["width"], CFG)
            if v < CFG["valley_min_ratio"]:
                continue
            x, y, w, h = d["bbox"]
            b = (x, y, x + w, y + h)
            real = any(_inter(b, g) >= 0.5 * max(_area(b), 1.0) for g in gts)
            f = features(pts, d, gray, trace, road, paint, H, W)
            f["valley"] = v
            rows.append({"image": path.name, "real": int(real), **{k: round(float(f[k]), 4) for k in FEATURES},
                         "x1": x, "y1": y, "x2": x + w, "y2": y + h})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(images)}", flush=True)
    with open(OUT / "features.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    report(rows)


def report(rows):
    R = [r for r in rows if r["real"]]; Fk = [r for r in rows if not r["real"]]
    L = ["# 거르기 F1 — 후보 특징으로 진짜 · 가짜 가르기 (개발 세트, E9-b 후보)", "",
         f"후보 {len(rows)}개 = 진짜 쪽 {len(R)} · 가짜 {len(Fk)}", "",
         "| 특징 | 진짜 중앙값 | 가짜 중앙값 | AUC (진짜가 큰 쪽이면 > 0.5) | 버리는 방향 | 진짜 5% 손실 기준값 → **가짜 버림** | 진짜 10% 손실 → 가짜 버림 |",
         "|---|---|---|---|---|---|---|"]
    for k in FEATURES:
        r = np.array([x[k] for x in R]); f = np.array([x[k] for x in Fk])
        t = cut_table(r, f)
        L.append(f"| {k} | {np.median(r):.3f} | {np.median(f):.3f} | {t['auc']:.3f} | {t['dir']} | "
                 f"{t[0.05][0]:.3f} → **{t[0.05][1] * 100:.0f}%** | {t[0.10][0]:.3f} → {t[0.10][1] * 100:.0f}% |")
    # 규칙대로 하나씩 더하기: 진짜 손실 5% 이하에서 가짜 15% 이상 버리는 특징, 많이 버리는 순
    L += ["", "## 하나씩 더하기 (규칙: 매번 남은 진짜 중 5% 손실 기준값 · 가짜 15% 이상 버릴 때만 · 누적 진짜 손실 15% 이하)", "",
          "| 순서 | 특징 | 기준 | 이번에 버린 가짜 (남은 것 중) | 누적: 남은 진짜 | 남은 가짜 | 진짜 비율 |", "|---|---|---|---|---|---|---|"]
    keepR, keepF = list(R), list(Fk)
    used = set(); step = 0
    while True:
        best = None
        for k in FEATURES:
            if k in used or k == "ypos":
                continue
            r = np.array([x[k] for x in keepR]); f = np.array([x[k] for x in keepF])
            t = cut_table(r, f, (0.05,))
            if t[0.05][1] >= 0.15 and (best is None or t[0.05][1] > best[2]):
                best = (k, t, t[0.05][1])
        if best is None:
            break
        k, t, gain = best
        thr = t[0.05][0]
        keep = (lambda v: v >= thr) if t["dir"] == "아래 버림" else (lambda v: v <= thr)
        nR = [x for x in keepR if keep(x[k])]; nF = [x for x in keepF if keep(x[k])]
        if 1 - len(nR) / len(R) > 0.15:
            break
        keepR, keepF = nR, nF; used.add(k); step += 1
        L.append(f"| {step} | {k} | {t['dir']} {thr:.3f} | {gain * 100:.0f}% | {len(keepR)} ({len(keepR) / len(R) * 100:.0f}%) | "
                 f"{len(keepF)} ({len(keepF) / len(Fk) * 100:.0f}%) | {len(keepR) / (len(keepR) + len(keepF)):.3f} |")
    L += ["", f"처음 진짜 비율 {len(R) / len(rows):.3f}"]
    md = "\n".join(L) + "\n"
    (OUT / "filter_features.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
