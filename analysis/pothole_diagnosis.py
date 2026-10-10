"""
포트홀 진단 · 설정 선택 (개발 세트만, rdd_test 안 씀) — src/pothole.py의 근거.
① 1차 방식이 왜 못 찾나: 정답 94개 중 후보가 걸친 정답 수
② 정답 포트홀 상자 vs 같은 높이 노면 상자 vs 1차 가짜 후보: 특징별 중앙값 · AUC
③ 2단계 방식 후보 · 검증 조합을 튜닝 394장에서 고르고 (규칙: 가짜/장 ≤ 1차(4.7) 안에서 in50 맞힘 최대) 검증 169장에서 확인
실행: python analysis/pothole_diagnosis.py        (약 10분)
출력: 화면 표 + outputs/analysis/pothole_diagnosis/grid.csv
"""
import csv, itertools, os, random, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data import list_images, load_gt
from detect import DEFAULT_CFG, _components, _pothole_mask, _touches_border, classify, shape_features
from evaluate import count_cut, evaluate, prf, transform_gt
from metrics import measure_quality
from paths import LABELS_DIR, OUTPUT_DIR, imread
from pothole import _box_cues, texture_z
from preprocess import classify_quality, geometry_preprocess, preprocess_condition, validate_config

OUT = OUTPUT_DIR / "analysis" / "pothole_diagnosis"


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                               # 동점 평균 순위
        m = allv == v; ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def xyxy(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)


def inside(d, g):
    iw = max(0, min(d[2], g[2]) - max(d[0], g[0])); ih = max(0, min(d[3], g[3]) - max(d[1], g[1]))
    return iw * ih / max(1, (d[2] - d[0]) * (d[3] - d[1]))


def baseline(gray):
    cfg = DEFAULT_CFG; H, W = gray.shape; out = []
    for pts in _components(_pothole_mask(gray, cfg), cfg["group_ksize"]):
        d = shape_features(pts); d["type"] = classify(d, H * W, "pothole", cfg)
        if d["type"] == "pothole" and not _touches_border(d["bbox"], W, H, cfg):
            out.append(d)
    return out


def features(gray, grad, chrom, box):
    c = _box_cues(gray, chrom, tuple(int(round(v)) for v in box))
    if c is None:
        return None
    H, W = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    bw, bh = x2 - x1, y2 - y1
    rx1, ry1, rx2, ry2 = max(0, x1 - bw // 2), max(0, y1 - bh // 2), min(W, x2 + bw // 2), min(H, y2 + bh // 2)
    ring = np.ones((ry2 - ry1, rx2 - rx1), bool); ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
    gi, gr = gray[y1:y2, x1:x2], gray[ry1:ry2, rx1:rx2][ring]
    q1, q3 = np.percentile(gr, [25, 75])
    c.update(dark=(np.median(gr) - np.median(gi)) / max(q3 - q1, 1.0), tex=gi.std() / max(gr.std(), 1e-3),
             grad=grad[y1:y2, x1:x2].mean() / max(grad[ry1:ry2, rx1:rx2][ring].mean(), 1e-3), rel_y=(y1 + y2) / 2 / H)
    return c


def score(rows):
    tp = fp = fn = 0
    for dets, gt, cut in rows:
        e = evaluate(dets, gt, cut)["pothole"]["in50"]; tp += e["tp"]; fp += e["fp"]; fn += e["fn"]
    p, r, _ = prf(tp, fp, fn)
    return tp, fp, fn, r or 0.0, p or 0.0, fp / max(1, len(rows))


def main():
    name, _, images = list_images("rdd_dev")
    gl = load_gt("rdd_dev", name)
    cfg_a = validate_config({"roi": "auto"})                 # p2bd와 같은 전처리 (ROI auto · 보정 없음)
    split = {r["image"]: r["split"] for r in csv.DictReader(open(LABELS_DIR / "split_rdd_dev.csv", encoding="utf-8"))}
    rng = random.Random(0)
    KEYS = [(2.5, 21), (3.0, 21), (3.0, 31), (3.5, 31)]
    data, P, N, F = [], [], [], []
    touched = 0
    for i, path in enumerate(images, 1):
        img = imread(path)
        ref, geo = geometry_preprocess(img, cfg_a, path.name)
        bgr, _ = preprocess_condition(ref, cfg_a, "none", classify_quality(measure_quality(ref)), {})
        gt = gl(path); gt_t = transform_gt(gt, geo); cut = count_cut(gt, gt_t)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = gray.shape
        s = bgr.astype(np.float32).sum(2) + 1
        chrom = np.stack([bgr[..., 2] / s, bgr[..., 1] / s], 2)
        g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        grad = cv2.magnitude(cv2.Sobel(g1, cv2.CV_32F, 1, 0), cv2.Sobel(g1, cv2.CV_32F, 0, 1))
        pot = [g[1:] for g in gt_t if g[0] == "pothole"]; allg = [g[1:] for g in gt_t]
        base = baseline(gray.astype(np.uint8))
        touched += sum(any(inside(xyxy(d["bbox"]), g) > 0 for d in base) for g in pot)
        for g in pot:                                        # ② 정답 · 같은 높이 노면
            f = features(gray, grad, chrom, g)
            if f: P.append(f)
            bw = g[2] - g[0]
            for _ in range(20):
                x = rng.uniform(0, W - bw); cb = (x, g[1], x + bw, g[3])
                if all(cb[2] <= o[0] or cb[0] >= o[2] or cb[3] <= o[1] or cb[1] >= o[3] for o in allg):
                    f2 = features(gray, grad, chrom, cb)
                    if f2: N.append(f2); break
        for d in base:
            bb = xyxy(d["bbox"])
            if not any(inside(bb, g) >= 0.5 for g in allg) and rng.random() < 0.15:
                f3 = features(gray, grad, chrom, bb)
                if f3: F.append(f3)
        cands = {}                                           # ③ 후보 (조합마다)
        for t, k in KEYS:
            Z = texture_z(gray, k)
            m = cv2.morphologyEx(((Z > t) * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            L = []
            for j in range(1, n):
                x, y, w, h, a = (int(v) for v in st[j])
                if 0.0006 * H * W <= a <= 0.08 * H * W:
                    c = _box_cues(gray, chrom, (x, y, x + w, y + h))
                    if c:
                        L.append({"bbox": (x, y, w, h), "type": "pothole", "aspect": max(w, h) / max(1, min(w, h)), **c,
                                  "rel_y": (y + h / 2) / H})
            cands[(t, k)] = L
        data.append({"split": split[path.name], "gt": gt_t, "cut": cut, "base": base, "cands": cands})
        if i % 100 == 0:
            print(f"  [{i}/{len(images)}]", flush=True)

    tp, fp, fn, r, p, fppi = score([(d["base"], d["gt"], d["cut"]) for d in data])
    print(f"\n① 1차 방식: in50 맞힘 {tp}/{tp + fn} · 가짜/장 {fppi:.2f} · 후보가 조금이라도 걸친 정답 {touched}/{tp + fn}")
    print(f"\n② 정답 {len(P)} · 같은 높이 노면 {len(N)} · 1차 가짜 표본 {len(F)}")
    print(f"{'특징':<10}{'정답':>8}{'노면':>8}{'1차가짜':>8}{'AUC 정답vs노면':>16}{'AUC 정답vs가짜':>16}")
    for k in ("dark", "dark_p10", "tex", "grad", "chroma", "rel_y"):
        a, b, c = [x[k] for x in P], [x[k] for x in N], [x[k] for x in F]
        print(f"{k:<10}{np.median(a):>8.3f}{np.median(b):>8.3f}{np.median(c):>8.3f}{auc(a, b):>16.3f}{auc(a, c):>16.3f}")

    grid = list(itertools.product(KEYS, [99, 1.0, 0.6], [-99, 0.0, 0.2], [99, 3.0, 2.0], [0.0, 0.3, 0.5]))
    rows = []
    for key, ch, dp, asp, ry in grid:
        res = {}
        for part in ("tune", "val"):
            sel = [([c for c in d["cands"][key] if c["chroma"] <= ch and c["dark_p10"] >= dp and c["aspect"] <= asp
                     and c["rel_y"] >= ry], d["gt"], d["cut"]) for d in data if d["split"] == part]
            res[part] = score(sel)
        rows.append({"z": key[0], "k": key[1], "chroma_max": ch, "darkp10_min": dp, "aspect_max": asp, "rel_y_min": ry,
                     **{f"{part}_{n}": v for part in res for n, v in zip(("tp", "fp", "fn", "R", "P", "fppi"), res[part])}})
    base_t = score([(d["base"], d["gt"], d["cut"]) for d in data if d["split"] == "tune"])
    ok = sorted([r for r in rows if r["tune_fppi"] <= base_t[5]], key=lambda r: (r["tune_tp"], -r["tune_fppi"]), reverse=True)
    print(f"\n③ 튜닝에서 고르기 (가짜/장 ≤ 1차 {base_t[5]:.2f} 안에서 맞힘 최대) → 검증 확인")
    for r in ok[:5]:
        print(f"  z>{r['z']} k{r['k']} 색도≤{r['chroma_max']} 어두운10%≥{r['darkp10_min']} 세장비≤{r['aspect_max']} 높이≥{r['rel_y_min']}"
              f" | 튜닝 {r['tune_tp']}/{r['tune_tp'] + r['tune_fn']} 가짜/장 {r['tune_fppi']:.2f}"
              f" | 검증 {r['val_tp']}/{r['val_tp'] + r['val_fn']} 가짜/장 {r['val_fppi']:.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "grid.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n저장 → {OUT / 'grid.csv'}")


if __name__ == "__main__":
    main()
