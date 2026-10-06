"""
진단 — 주 경로(ROI auto → D1 + 양쪽 확인)에서 ① 진짜 균열을 어느 판정 단계에서 놓치나 ② 남은 가짜는 무엇인가.
개발 세트만 쓴다 (테스트 세트는 최종 확인용).

① 놓친 균열 — 정답 균열 박스마다, 박스 안(후보 면적 50% 이상)에 들어간 후보가 판정을 어디까지 통과했나
   단계: 후보 없음 → 너무 작음(면적) → 덜 길쭉함(세장비 < 3) → 위쪽 가장자리 → 양쪽 확인 → 찾음
   (포트홀 갈래로만 잡힌 경우 = "포트홀로 분류") · 그 박스 안 후보 중 가장 멀리 간 것 기준
② 가짜 — 최종 균열 후보 중 어떤 정답 균열 박스에도 50% 이상 들어가지 않은 것, 위에서부터 먼저 걸리는 출처
   정답 걸침(정답 박스에 일부 겹침 — 라벨이 좁았을 가능성) → 노면 밖(도로다움 없음 · 초록) → 페인트 옆
   (흰색 · 노란색 ±6px) → 그림자 · 조명 경계(조명 배경 기울기 큼) → 노면 위 기타(짧은 조각 / 긴 것)
출력: outputs/analysis/diagnose_d1.md + 출처별 · 단계별 샘플 그림
실행: python analysis/diagnose_d1.py [--valley-ratio 0.35] [--crack-lo-ratio 0.5] [--limit N]
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data import list_images
from detect import (DEFAULT_CFG, _components, _crack_mask, _pothole_mask, _to_gray, _touches_border,
                    classify, shape_features, valley_score)
from evaluate import _area, _inter, _xyxy, transform_gt
from paths import OUTPUT_DIR, RDD_DIR, imread, imwrite
from preprocess import flatten_background, geometry_preprocess, validate_config

TYPES = {"longitudinal crack": "세로", "transverse crack": "가로", "alligator crack": "거북등"}
STAGES = ["후보 없음", "포트홀로 분류", "너무 작음", "덜 길쭉함", "위쪽 가장자리", "양쪽 확인", "찾음"]
FP_SOURCES = ["정답 걸침", "노면 밖", "페인트 옆", "그림자 · 조명 경계", "노면 위 · 짧은 조각", "노면 위 · 긴 것"]
PAD = 6


def inside(det_box, gt_box):
    return _inter(det_box, gt_box) >= 0.5 * max(_area(det_box), 1.0)


def road_mask(ref):
    """ROI auto와 같은 '도로다움' 판정을 사진 전체 픽셀에 (src/roi.py 1 · 2단계)."""
    lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    H, W = L.shape
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(L, cv2.CV_32F, ksize=3)), (0, 0), 3)
    r = (slice(int(.60 * H), int(.80 * H)), slice(int(.30 * W), int(.70 * W)))
    ma, mb = np.median(A[r]), np.median(B[r])
    sa, sb = max(float(A[r].std()), 3.0), max(float(B[r].std()), 3.0)
    mL, sL, mt = np.median(L[r]), float(L[r].std()), np.median(tex[r])
    road = ((np.sqrt(((A - ma) / sa) ** 2 + ((B - mb) / sb) ** 2) < 3) & (tex > 0.35 * mt) & (L < mL + max(3 * sL, 45)))
    return cv2.dilate(road.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0     # 경계 근처는 도로로 봐줌


def paint_mask(ref):
    hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (v > 200) & (s < 40)
    yellow = (h >= 15) & (h <= 35) & (s > 80) & (v > 120)
    return white | yellow


def candidates(ref, cfg):
    """detect()의 D1 판정을 단계별로 펼친다 → 후보마다 (상자, 갈래, 통과한 단계, 최종 균열 여부, 특징)."""
    gray = _to_gray(ref)
    H, W = gray.shape
    smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    mask, bh, hi = _crack_mask(gray, cfg)
    out = []
    for branch, m, b, h in (("crack", mask, bh, hi), ("pothole", _pothole_mask(gray, cfg), None, None)):
        for pts in _components(m, cfg["group_ksize"]):
            d = shape_features(pts, b, h)
            kind = classify(d, H * W, branch, cfg)
            if branch == "pothole":
                stage = "포트홀로 분류"
            elif d["area"] / (H * W) < cfg["min_area_ratio"]:
                stage = "너무 작음"
            elif kind != "crack":
                stage = "덜 길쭉함"
            elif _touches_border(d["bbox"], W, H, cfg):
                stage = "위쪽 가장자리"
            elif valley_score(smooth, pts, d["width"], cfg) < cfg["valley_min_ratio"]:
                stage = "양쪽 확인"
            else:
                stage = "찾음"
            out.append((_xyxy(d["bbox"]), stage, d))
    return out


def run(ratio, limit, lo_ratio=None):
    name, _, images = list_images("rdd_dev")
    images = images[:limit] if limit else images
    pcfg = validate_config({"roi": "auto"})
    dcfg = {**DEFAULT_CFG, "valley_check": True, "valley_min_ratio": ratio}
    if lo_ratio is not None:
        dcfg["crack_lo_ratio"] = lo_ratio
    lost = Counter(); lost_by = defaultdict(Counter); n_gt = Counter()
    fp = Counter(); fp_len = []; n_img = 0
    samples = defaultdict(list)
    rng = np.random.default_rng(0)
    for p in images:
        ann = json.loads((RDD_DIR / "ann" / f"{p.name}.json").read_text(encoding="utf-8"))
        raw = [(TYPES[o["classTitle"]], *o["points"]["exterior"][0], *o["points"]["exterior"][1])
               for o in ann["objects"] if o["classTitle"] in TYPES]
        img = imread(p)
        ref, geo = geometry_preprocess(img, pcfg, p.name)
        H, W = ref.shape[:2]
        n_img += 1
        boxes = [("crack", min(a[1], a[3]), min(a[2], a[4]), max(a[1], a[3]), max(a[2], a[4])) for a in raw]
        kept = transform_gt(boxes, geo)
        cand = candidates(ref, dcfg)
        # ① 놓친 균열 — 잘린 정답이 있는 사진은 상자 순서를 맞출 수 없어 그 사진의 정답 전체를 "ROI 밖"으로 센다 (드묾)
        types = [a[0] for a in raw]
        gt_xyxy = [tuple(g[1:]) for g in kept]
        for t in types:
            n_gt[t] += 1
        if len(gt_xyxy) != len(types):
            for t in types:
                lost["ROI 밖 (잘림)"] += 1
                lost_by[t]["ROI 밖 (잘림)"] += 1
        else:
            for gb, t in zip(gt_xyxy, types):
                inner = [(b, s) for b, s, _ in cand if inside(b, gb)]
                best = max((s for _, s in inner), key=STAGES.index) if inner else "후보 없음"
                lost[best] += 1
                lost_by[t][best] += 1
                if best != "찾음" and len(samples[f"lost_{best}"]) < 12 and rng.random() < 0.5:
                    samples[f"lost_{best}"].append((ref, gb, [b for b, _ in inner]))
        # ② 가짜
        road, paint = road_mask(ref), paint_mask(ref)
        L = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[..., 0]
        bg = flatten_background(L, {"ksize": 61})
        grad = cv2.GaussianBlur(np.abs(cv2.Sobel(bg, cv2.CV_32F, 0, 1)) + np.abs(cv2.Sobel(bg, cv2.CV_32F, 1, 0)), (0, 0), 3)
        for b, s, d in cand:
            if s != "찾음" or any(inside(b, g) for g in gt_xyxy):
                continue
            x1, y1, x2, y2 = (int(v) for v in b)
            cx, cy = min((x1 + x2) // 2, W - 1), min((y1 + y2) // 2, H - 1)
            xa, ya, xb, yb = max(0, x1 - PAD), max(0, y1 - PAD), min(W, x2 + PAD), min(H, y2 + PAD)
            if any(_inter(b, g) > 0 for g in gt_xyxy):
                src = "정답 걸침"
            elif road[y1:y2 + 1, x1:x2 + 1].mean() < 0.5:
                src = "노면 밖"
            elif paint[ya:yb, xa:xb].mean() >= 0.03:
                src = "페인트 옆"
            elif grad[cy, cx] >= 6:
                src = "그림자 · 조명 경계"
            elif d["length"] < 40:
                src = "노면 위 · 짧은 조각"
            else:
                src = "노면 위 · 긴 것"
            fp[src] += 1
            fp_len.append(d["length"])
            if len(samples[f"fp_{src}"]) < 12 and rng.random() < 0.15:
                samples[f"fp_{src}"].append((ref, b, []))
    return lost, lost_by, n_gt, fp, n_img, samples


def sheet(items, title, path, T=220):
    tiles = []
    for ref, box, inner in items:
        x1, y1, x2, y2 = (int(v) for v in box)
        m = 30
        a = ref.copy()
        cv2.rectangle(a, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for b in inner:
            cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
        c = a[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
        s = T / max(c.shape[:2])
        c = cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))))
        t = np.full((T, T, 3), 255, np.uint8)
        t[:c.shape[0], :c.shape[1]] = c
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.full((T, T, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    img = np.vstack(rows)
    canvas = np.full((img.shape[0] + 30, img.shape[1], 3), 255, np.uint8)
    canvas[30:] = img
    cv2.putText(canvas, title, (6, 22), 0, 0.6, (0, 0, 0), 1)
    imwrite(path, canvas)


def report(lost, lost_by, n_gt, fp, n_img, ratio):
    total = sum(lost.values())
    lines = [f"# 진단 — ROI auto → D1 + 양쪽 확인 {ratio} (개발 세트)\n",
             "## ① 진짜 균열을 어디서 놓치나 — 판정 단계별\n",
             "| 단계 (이 단계에서 버려짐) | 정답 수 | 비율 |", "|---|---|---|"]
    for s in ["ROI 밖 (잘림)"] + STAGES:
        if lost[s]:
            lines.append(f"| {s} | {lost[s]} | {lost[s] / total:.1%} |")
    lines += ["\n### 종류별\n", "| 종류 | 정답 | " + " | ".join(STAGES) + " |", "|---|---|" + "---|" * len(STAGES)]
    for t in TYPES.values():
        tot = sum(lost_by[t].values())
        if tot:
            lines.append(f"| {t} | {tot} | " + " | ".join(f"{lost_by[t][s] / tot:.0%}" for s in STAGES) + " |")
    ftot = sum(fp.values())
    lines += [f"\n## ② 남은 가짜는 무엇인가 — 사진당 {ftot / n_img:.1f}개\n",
              "| 출처 | 사진당 | 비율 |", "|---|---|---|"]
    for s in FP_SOURCES:
        lines.append(f"| {s} | {fp[s] / n_img:.2f} | {fp[s] / max(ftot, 1):.1%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--valley-ratio", type=float, default=0.35)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--crack-lo-ratio", type=float, help="D1 이중 임계값 약한 기준 비율 (기본 0.5)")
    a = ap.parse_args()
    lost, lost_by, n_gt, fp, n_img, samples = run(a.valley_ratio, a.limit, a.crack_lo_ratio)
    out = OUTPUT_DIR / "analysis" / ("diagnose_d1" + (f"_lo{a.crack_lo_ratio}" if a.crack_lo_ratio else ""))
    out.mkdir(parents=True, exist_ok=True)
    md = report(lost, lost_by, n_gt, fp, n_img, a.valley_ratio)
    (out / "diagnose_d1.md").write_text(md, encoding="utf-8")
    for k, items in samples.items():
        if items:
            sheet(items, k, out / f"{k.replace(' ', '').replace('·', '_')}.jpg")
    print(md)
    print(f"→ {out}")
