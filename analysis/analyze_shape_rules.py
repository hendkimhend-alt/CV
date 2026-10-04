"""
④ 판정 규칙의 기준 숫자(세장비·면적)를 RDD 라벨로 정한다.

방법:
1. RDD2020_train 라벨 사진을 파이프라인과 똑같이 준비 (흑백 → 긴 변 1024 → 하단 절반 ROI), 라벨 박스도 같이 변환
2. detect.py의 D1 마스크(균열 branch / 포트홀 branch)로 덩어리를 뽑고, detect.py의 shape_features()로 잰다
   → 라벨 박스 모양이 아니라 "실제로 검출되는 덩어리" 모양으로 기준을 정함
3. 덩어리 픽셀의 50% 이상이 같은 종류의 정답 박스 안이면 '안', 아니면 '밖'
   (균열 branch ↔ longitudinal/transverse/alligator, 포트홀 branch ↔ pothole)
4. 기준 숫자 후보를 훑으며 남는 덩어리의 정밀도(안 비율)와 박스 재현율(덩어리가 하나라도 걸린 정답 박스 비율) 계산
주의: 결과는 ① 표시하기 파라미터(DEFAULT_CFG)에 의존 → ①을 튜닝하면 다시 돌려야 함.
한계: 박스 안 잡음 덩어리까지 '안'으로 세서 개선이 묻힘 → 검출기 평가는 eval_detector.py(박스 단위)를 쓴다.
출력: out/shape_rules.md, out/shape_blobs.csv
① 파라미터를 바꿔 보려면: python analysis/analyze_shape_rules.py '{"blackhat_ksize": 7}'  → out/shape_rules_custom.md
"""
import csv, glob, json, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import OUTPUT_DIR, PROVIDED_DIR, RDD_DIR, imread, imwrite
from detect import DEFAULT_CFG, _components, _crack_mask, _pothole_mask, shape_features

ROOT = str(RDD_DIR)
OUT = str(OUTPUT_DIR / "analysis")
os.makedirs(OUT, exist_ok=True)
LONG_SIDE = 1024
GT_CLASSES = {"crack": {"longitudinal crack", "transverse crack", "alligator crack"}, "pothole": {"pothole"}}
MIN_AREA_RATIO = 0.00005   # 이보다 작은 덩어리는 분석에서 제외 (어차피 잡음, 속도용)
OVERRIDE = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
DEFAULT_CFG.update(OVERRIDE)
SUFFIX = "_custom" if OVERRIDE else ""


def prepare(gray, boxes):
    h, w = gray.shape
    s = LONG_SIDE / max(h, w)
    gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    top = gray.shape[0] // 2
    roi = gray[top:, :]
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        x1, x2, y1, y2 = x1 * s, x2 * s, y1 * s - top, y2 * s - top
        x1, x2 = max(0, x1), min(roi.shape[1], x2)
        y1, y2 = max(0, y1), min(roi.shape[0], y2)
        if x2 > x1 and y2 > y1:
            out.append((cls, int(x1), int(y1), int(x2), int(y2)))
    return roi, out


blobs = []
n_gt = {"crack": 0, "pothole": 0}
for ann in sorted(glob.glob(f"{ROOT}/ann/*.json")):
    d = json.load(open(ann))
    raw = [(o["classTitle"], *o["points"]["exterior"][0], *o["points"]["exterior"][1]) for o in d["objects"]]
    raw = [(c, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)) for c, x1, y1, x2, y2 in raw]
    if not any(c in GT_CLASSES["crack"] | GT_CLASSES["pothole"] for c, *_ in raw):
        continue
    name = os.path.basename(ann)[:-5]
    gray = imread(f"{ROOT}/img/{name}", cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue
    roi, boxes = prepare(gray, raw)
    H, W = roi.shape
    img_area = H * W

    for branch, mask_fn in [("crack", lambda g, c: _crack_mask(g, c)[0]), ("pothole", _pothole_mask)]:
        gt = [(i, b) for i, b in enumerate(boxes) if b[0] in GT_CLASSES[branch]]
        n_gt[branch] += len(gt)
        union = np.zeros((H, W), bool)
        box_id = np.full((H, W), -1, np.int32)
        for i, (_, x1, y1, x2, y2) in gt:
            union[y1:y2, x1:x2] = True
            box_id[y1:y2, x1:x2] = i
        for pts in _components(mask_fn(roi, DEFAULT_CFG), DEFAULT_CFG["group_ksize"]):
            f = shape_features(pts)
            if f["area"] / img_area < MIN_AREA_RATIO:
                continue
            frac = union[pts[:, 1], pts[:, 0]].mean()
            ids = box_id[pts[:, 1], pts[:, 0]]
            ids = ids[ids >= 0]
            blobs.append({"image": name, "branch": branch, "area_ratio": f["area"] / img_area,
                          "elong": f["elong"], "inside": frac >= 0.5,
                          "box": f"{name}#{np.bincount(ids).argmax()}" if len(ids) and frac >= 0.5 else ""})

with open(f"{OUT}/shape_blobs{SUFFIX}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=blobs[0].keys())
    w.writeheader()
    w.writerows(blobs)


def pct(v, qs=(10, 25, 50, 75, 90)):
    return " / ".join(f"{np.percentile(v, q):.4g}" for q in qs)


def sweep(bs, keep, n_boxes):
    kept = [b for b in bs if keep(b)]
    if not kept:
        return 0, 0.0, 0.0, 0.0
    prec = np.mean([b["inside"] for b in kept])
    rec = len({b["box"] for b in kept if b["box"]}) / max(n_boxes, 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return len(kept), prec, rec, f1


lines = ["# ④ 판정 규칙 기준 숫자 — RDD 덩어리 분포로 정하기\n",
         f"RDD2020_train 라벨 사진에 D1 마스크(현재 `DEFAULT_CFG`)를 돌려 나온 덩어리를 정답 박스 안/밖으로 나눔. "
         f"정답 박스: 균열 {n_gt['crack']}개, 포트홀 {n_gt['pothole']}개 (ROI 안). 재현: `python analysis/analyze_shape_rules.py`\n",
         "- 정밀도 = 남은 덩어리 중 정답 박스 안 비율 / 박스 재현율 = 덩어리가 하나라도 걸린 정답 박스 비율",
         "- 면적 비율은 ROI(하단 절반) 면적 기준. **① 파라미터를 바꾸면 다시 돌려야 함**",
         f"- ① 덮어쓴 값: `{json.dumps(OVERRIDE)}`\n" if OVERRIDE else ""]

for branch in ["crack", "pothole"]:
    bs = [b for b in blobs if b["branch"] == branch]
    ins = [b for b in bs if b["inside"]]
    out = [b for b in bs if not b["inside"]]
    lines += [f"## {branch} branch — 덩어리 {len(bs)}개 (안 {len(ins)} / 밖 {len(out)})\n",
              "| | 세장비 p10 / p25 / p50 / p75 / p90 | 면적 비율 p10 / p25 / p50 / p75 / p90 |", "|---|---|---|",
              f"| 정답 박스 안 | {pct([b['elong'] for b in ins])} | {pct([b['area_ratio'] for b in ins])} |",
              f"| 밖 | {pct([b['elong'] for b in out])} | {pct([b['area_ratio'] for b in out])} |\n"]

    areas = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
    if branch == "crack":
        elongs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
        keep = lambda a, e: (lambda b: b["area_ratio"] >= a and b["elong"] >= e)
        cur = (DEFAULT_CFG["min_area_ratio"], DEFAULT_CFG["crack_min_elong"])
        lines.append("격자: 행 = 최소 면적 비율, 열 = 최소 세장비. 칸 = 정밀도 / 박스 재현율 / F1\n")
    else:
        elongs = [2.0, 3.0, 4.0, 6.0, 100.0]
        keep = lambda a, e: (lambda b: b["area_ratio"] >= a and b["elong"] < e)
        cur = (DEFAULT_CFG["pothole_min_area_ratio"], DEFAULT_CFG["pothole_max_elong"])
        lines.append("격자: 행 = 최소 면적 비율, 열 = 최대 세장비(미만). 칸 = 정밀도 / 박스 재현율 / F1\n")
    lines += ["| 면적 \\ 세장비 | " + " | ".join(f"{e:g}" for e in elongs) + " |", "|---|" + "---|" * len(elongs)]
    best = None
    for a in areas:
        cells = []
        for e in elongs:
            n, p, r, f1 = sweep(bs, keep(a, e), n_gt[branch])
            mark = "**" if (a, e) == cur else ""
            cells.append(f"{mark}{p:.2f} / {r:.2f} / {f1:.2f}{mark}")
            if best is None or f1 > best[0]:
                best = (f1, a, e, p, r, n)
        lines.append(f"| {a:g} | " + " | ".join(cells) + " |")
    n, p, r, f1 = sweep(bs, keep(*cur), n_gt[branch])
    lines += [f"\n- 현재 값 (면적 {cur[0]:g}, 세장비 {cur[1]:g}): 정밀도 {p:.2f} / 재현율 {r:.2f} / F1 {f1:.2f}, 남는 덩어리 {n}개",
              f"- F1 최고 (면적 {best[1]:g}, 세장비 {best[2]:g}): 정밀도 {best[3]:.2f} / 재현율 {best[4]:.2f} / F1 {best[0]:.2f}, 남는 덩어리 {best[5]}개\n"]

open(f"{OUT}/shape_rules{SUFFIX}.md", "w").write("\n".join(lines))
print("\n".join(lines))
