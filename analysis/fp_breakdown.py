"""
오검출(정답 박스에 안 걸친 후보)을 출처별로 나눈다. RDD 라벨 사용, 검출기 D1.

출처는 라벨이 없어 박스 주변으로 추정한다 (위에서부터 먼저 걸리는 것):
  풀·나무      박스 안 초록 픽셀(HSV H 30~90, S>50, V>40) ≥ 30%   — 노면 밖
  원경        박스가 노면 영역 위쪽 25% 안                        — 차량·인도·건물
  차선·밝은 표시 박스 주변(±8px) 밝기 > 200 픽셀 ≥ 5%               — 차선 가장자리·반사
  그림자·조도 경계 박스 주변(±8px)의 큰 범위 밝기(σ=12 블러) 최대−최소 ≥ 40
  노면 위      나머지 — 골재 질감, 이음매, 라벨 안 된 실제 손상 등
주의: RDD 라벨에 빠진 손상이 있어 "오검출" 중 일부는 실제 손상일 수 있음 (주로 '노면 위'에 섞임).
출력: outputs/analysis/fp_breakdown.md, fp_samples_<출처>.jpg (출처별 샘플로 규칙 검증용)
실행: python analysis/fp_breakdown.py [--limit N]
"""
import argparse, os, random, sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data import list_images, load_gt
from detect import detect
from evaluate import _overlaps, transform_gt
from metrics import measure_quality
from paths import OUTPUT_DIR, imread, imwrite
from preprocess import classify_quality, geometry_preprocess, preprocess_condition, validate_config

SOURCES = ["풀·나무", "원경", "차선·밝은 표시", "그림자·조도 경계", "노면 위"]
FILE_TAG = {"풀·나무": "vegetation", "원경": "far", "차선·밝은 표시": "bright", "그림자·조도 경계": "shadow", "노면 위": "road"}
PAD = 8


def classify_fp(bbox, green, gray, illum, H, W):
    x, y, w, h = bbox
    if green[y:y + h, x:x + w].mean() >= 0.3:
        return "풀·나무"
    if y + h <= 0.25 * H:
        return "원경"
    x1, y1, x2, y2 = max(0, x - PAD), max(0, y - PAD), min(W, x + w + PAD), min(H, y + h + PAD)
    if (gray[y1:y2, x1:x2] > 200).mean() >= 0.05:
        return "차선·밝은 표시"
    region = illum[y1:y2, x1:x2]
    if float(region.max()) - float(region.min()) >= 40:
        return "그림자·조도 경계"
    return "노면 위"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    name, _, images = list_images("rdd")
    images = images[:args.limit] if args.limit else images
    gt_loader = load_gt("rdd", name)
    cfg = validate_config({})
    conds = ["P0_reference", "P1+"]
    counts = {c: {k: Counter() for k in ("crack", "pothole")} for c in conds}
    totals = {c: Counter() for c in conds}
    samples = defaultdict(list)          # (출처) -> 크롭들 (P1+ 균열 오검출에서)
    random.seed(0)

    for path in images:
        img = imread(path)
        ref, geom = geometry_preprocess(img, cfg, path.name)
        tags = classify_quality(measure_quality(ref))
        gt = transform_gt(gt_loader(path) or [], geom)
        for cond in conds:
            fixed, _ = preprocess_condition(ref, cfg, cond, tags)
            gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(fixed, cv2.COLOR_BGR2HSV)
            green = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)
            illum = cv2.GaussianBlur(gray, (0, 0), 12)
            H, W = gray.shape
            for d in detect(fixed):
                kind = d["type"]
                totals[cond][kind] += 1
                if any(_overlaps(d["bbox"], g) for g in gt if g[0] == kind):
                    continue
                src = classify_fp(d["bbox"], green, gray, illum, H, W)
                counts[cond][kind][src] += 1
                if cond == "P1+" and kind == "crack" and random.random() < 0.05 and len(samples[src]) < 24:
                    x, y, w, h = d["bbox"]
                    x1, y1 = max(0, x - 20), max(0, y - 20)
                    crop = fixed[y1:min(H, y + h + 20), x1:min(W, x + w + 20)].copy()
                    cv2.rectangle(crop, (x - x1, y - y1), (x - x1 + w, y - y1 + h), (0, 0, 255), 1)
                    samples[src].append(cv2.resize(crop, (160, 120)))

    n = len(images)
    lines = [f"# 오검출 출처별 분해 (RDD {n}장, D1)\n",
             "오검출 = 같은 종류 정답 박스에 걸치지 않은 후보. 출처는 박스 주변으로 추정 (규칙은 스크립트 설명 참고).\n"]
    for kind, label in [("crack", "균열"), ("pothole", "포트홀")]:
        lines += [f"## {label}\n", "| 출처 | P0 사진당 | P0 비율 | P1+ 사진당 | P1+ 비율 | P0→P1+ 증가 |", "|---|---|---|---|---|---|"]
        f0, f1 = sum(counts["P0_reference"][kind].values()), sum(counts["P1+"][kind].values())
        for s in SOURCES:
            a, b = counts["P0_reference"][kind][s], counts["P1+"][kind][s]
            lines.append(f"| {s} | {a / n:.2f} | {a / max(f0, 1) * 100:.0f}% | {b / n:.2f} | {b / max(f1, 1) * 100:.0f}% | {(b - a) / n:+.2f} |")
        lines.append(f"| **합계** | **{f0 / n:.2f}** | | **{f1 / n:.2f}** | | **{(f1 - f0) / n:+.2f}** |")
        lines.append(f"\n후보 전체: P0 {totals['P0_reference'][kind] / n:.1f}/장, P1+ {totals['P1+'][kind] / n:.1f}/장\n")
    out = OUTPUT_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fp_breakdown.md").write_text("\n".join(lines), encoding="utf-8")
    for s, crops in samples.items():
        crops += [np.full((120, 160, 3), 255, np.uint8)] * (-len(crops) % 8)
        grid = np.vstack([np.hstack(crops[i:i + 8]) for i in range(0, len(crops), 8)])
        imwrite(out / f"fp_samples_{FILE_TAG[s]}.jpg", grid)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
