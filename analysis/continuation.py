"""
노면 위 거르기 O3 — 이어짐 재기. 후보를 붙이지 않고, 끝에서 더 따라가 보며 "후보 바깥으로 계속 이어지나"를 잰다.
바탕 = Hessian 약한 기준 ×0.2 최종 후보(노면 단서 통과). 개발 세트만.

① 방향 따라 이어 보기 (Hessian 선 점수 4-1 · 3-1, 직선 맞춤 5-1)
  양 끝(주축 방향 양 끝점)에서 끝 12px에 직선을 맞춰 바깥 방향을 정하고, 3px씩 전진하며 좌우 ±1px 중
  선 점수가 가장 큰 점으로 이동. 약한 기준 미만이 6px 넘게 계속되면 멈춤 (한쪽 최대 45px)
  ext       이어진 길이 (양 끝 합, px)
  ext_score 따라간 길의 점수 평균 ÷ 강한 기준
  ext_turn  걸음마다 방향이 꺾인 정도 평균 (라디안)
② 번져 나가기 (국소 이진화 · 연결 요소 · 세장비 2-1)
  상자 2배 창에서 "창 밝기 중앙값 − 0.5 × 사분위 범위"보다 어두운 픽셀 중 후보와 이어진 덩어리
  perc_spread 퍼진 넓이 ÷ 후보 픽셀 수 · perc_elong 퍼진 덩어리의 세장비
출력: outputs/analysis/continuation/ (features.csv, continuation.md, paths.json — 그림용 경로)
"""
import csv, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_images
from detect import (DEFAULT_CFG, _components, _crack_mask, _to_gray, _touches_border, classify, shape_features,
                    valley_score)
from evaluate import _area, _inter, transform_gt
from filter_features import cut_table, window
from onroad_features import greedy
from paths import OUTPUT_DIR, RDD_DIR, imread
from preprocess import geometry_preprocess, validate_config
from roadcue import cue_values, maps, ref_stats

OUT = OUTPUT_DIR / "analysis" / "continuation"
CUE = {"energy": 1.6082, "coherence": 0.2941, "color": 2.9964}
LO = 0.2
CFG = {**DEFAULT_CFG, "detector": "D1", "crack_find": "line", "line_hi_abs": 32.5, "line_thicken": 1,
       "crack_min_length": 45, "valley_check": True, "valley_min_ratio": 0.35, "line_lo_ratio": LO}
CRACKS = {"longitudinal crack", "transverse crack", "alligator crack"}
FEATURES = ["ext", "ext_score", "ext_turn", "perc_spread", "perc_elong"]
STEP, SIDE, GAP, MAXLEN, TIP = 3, 1, 6, 45, 12


def ends(pts):
    """주축 방향 양 끝점과 바깥 방향 (끝 TIP px에 맞춘 직선)."""
    p = pts.astype(np.float64)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    proj = (p - c) @ vt[0]
    out = []
    for e in (p[proj.argmin()], p[proj.argmax()]):
        near = p[((p - e) ** 2).sum(1) <= TIP * TIP]
        if len(near) >= 3:
            _, _, v = np.linalg.svd(near - near.mean(0), full_matrices=False)
            d = v[0]
        else:
            d = vt[0]
        if np.dot(d, e - c) < 0:                                    # 가운데에서 바깥쪽으로
            d = -d
        out.append((e, d / np.linalg.norm(d)))
    return out


def follow(e, d, score, own, lo, hi):
    """끝 e에서 방향 d로 따라가기 → (이어진 길이, 좋은 점 점수들, 꺾임들, 경로)."""
    H, W = score.shape
    pos, path, good, turns = e.copy(), [], [], []
    last_good, gap, walked = 0.0, 0, 0.0
    while walked < MAXLEN:
        n = np.array([-d[1], d[0]])
        best, bp = -1.0, None
        for k in range(-SIDE, SIDE + 1):
            q = pos + STEP * d + k * n
            x, y = int(round(q[0])), int(round(q[1]))
            if not (0 <= x < W and 0 <= y < H) or own[y, x]:
                continue
            if score[y, x] > best:
                best, bp = float(score[y, x]), q
        if bp is None:
            break
        nd = (bp - pos) / np.linalg.norm(bp - pos)
        turns.append(float(np.arccos(np.clip(np.dot(nd, d), -1, 1))))
        d = nd
        pos = bp
        walked += STEP
        path.append((float(pos[0]), float(pos[1])))
        if best >= lo:
            good.append(best / hi)
            last_good, gap = walked, 0
        else:
            gap += STEP
            if gap > GAP:
                break
    return last_good, good, turns, path[: int(last_good / STEP)]


def percolate(pts, gray, b, W, H):
    a1, b1, a2, b2 = window(b, W, H, 2.0)
    g = gray[b1:b2, a1:a2].astype(np.float32)
    q1, med, q3 = np.percentile(g, [25, 50, 75])
    dark = (g < med - 0.5 * (q3 - q1)).astype(np.uint8)
    inn = (pts[:, 0] >= a1) & (pts[:, 0] < a2) & (pts[:, 1] >= b1) & (pts[:, 1] < b2)
    dark[pts[inn, 1] - b1, pts[inn, 0] - a1] = 1                   # 후보 자신은 씨앗
    n, lab = cv2.connectedComponents(dark, connectivity=8)
    ids = np.unique(lab[pts[inn, 1] - b1, pts[inn, 0] - a1])
    reg = np.isin(lab, ids[ids > 0])
    ys, xs = np.nonzero(reg)
    (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.int32))
    rw, rh = rw + 1, rh + 1
    return {"perc_spread": float(reg.sum() / len(pts)), "perc_elong": float(max(rw, rh) / min(rw, rh))}


def measure(pts, score, hi, gray, W, H, b):
    own = np.zeros(score.shape, bool)
    own[pts[:, 1], pts[:, 0]] = True
    own = cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    total, good, turns, paths = 0.0, [], [], []
    for e, d in ends(pts):
        L, g, t, p = follow(e, d, score, own, LO * hi, hi)
        total += L; good += g; turns += t[: max(int(L / STEP), 1)]; paths.append(p)
    f = {"ext": total, "ext_score": float(np.mean(good)) if good else 0.0,
         "ext_turn": float(np.mean(turns)) if turns else 0.0}
    f.update(percolate(pts, gray, b, W, H))
    return f, paths


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, images = list_images("rdd_dev")
    images = images[:limit] if limit else images
    pcfg = validate_config({"roi": "auto"})
    rows, allpaths = [], {}
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
        mask, score, hi = _crack_mask(gray, CFG)
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
            f, paths = measure(pts, score, hi, smooth, W, H, b)
            rows.append({"image": path.name, "real": int(real), **{k: round(float(f[k]), 4) for k in FEATURES},
                         "x1": x, "y1": y, "x2": x + w, "y2": y + h})
            allpaths[f"{path.name}|{x}|{y}|{x + w}|{y + h}"] = paths
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(images)}", flush=True)
    with open(OUT / "features.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    (OUT / "paths.json").write_text(json.dumps(allpaths), encoding="utf-8")
    report(rows)


def report(rows):
    R = [r for r in rows if int(r["real"])]; Fk = [r for r in rows if not int(r["real"])]
    L = [f"# 노면 위 거르기 O3 — 이어짐 재기 (개발 세트, ×{LO} 최종 후보)", "",
         f"후보 {len(rows)} = 진짜 쪽 {len(R)} · 가짜 {len(Fk)}", "",
         "| 특징 | 진짜 중앙값 | 가짜 중앙값 | 버리는 쪽 | 진짜 5% 손실 → **가짜 버림** | 진짜 10% → 가짜 버림 |",
         "|---|---|---|---|---|---|"]
    for k in FEATURES:
        r = np.array([float(x[k]) for x in R]); f = np.array([float(x[k]) for x in Fk])
        t = cut_table(r, f)
        L.append(f"| {k} | {np.median(r):.3f} | {np.median(f):.3f} | {t['dir']} | "
                 f"{t[0.05][0]:.3f} → **{t[0.05][1] * 100:.0f}%** | {t[0.10][1] * 100:.0f}% |")
    L += ["", "하나씩 더하기 (진짜 5%씩 · 가짜 15% 이상 · 누적 진짜 손실 10% 이하)", "",
          "| 순서 | 특징 | 기준 | 이번에 버린 가짜 | 남은 진짜 | 남은 가짜 |", "|---|---|---|---|---|---|"]
    for n, (k, dr, thr, gain, nr, nf) in enumerate(greedy(
            [{k: float(x[k]) for k in FEATURES} for x in R], [{k: float(x[k]) for k in FEATURES} for x in Fk],
            feats=FEATURES), 1):
        L.append(f"| {n} | {k} | {dr} {thr:.4f} | {gain * 100:.0f}% | {nr} ({nr / len(R) * 100:.0f}%) | {nf} ({nf / len(Fk) * 100:.0f}%) |")
    md = "\n".join(L) + "\n"
    (OUT / "continuation.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(list(csv.DictReader(open(OUT / "features.csv", encoding="utf-8"))))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
