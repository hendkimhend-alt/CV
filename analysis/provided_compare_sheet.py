"""
제공 13장: 정답(사용자 초안) vs 우리 검출 결과를 사진마다 나란히 한 장에 (10/12) — 측정은 provided_check.py와 같음.
왼쪽 = 정답 (빨강 균열 · 주황 포트홀, data/road_damage_annotated) · 오른쪽 = F + 흐림 대응의 균열 후보
  (초록 굵게 = 정답 안 맞힘 · 빨강 = 정답 밖 가짜) + 정답 균열 테두리(흰 점선 대신 흰 얇은 선)
실행: python analysis/provided_compare_sheet.py → outputs/provided_check/compare_sheet.jpg
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import provided_check as PC  # noqa: E402
import run_road_pipeline as R  # noqa: E402
import visual_check as V  # noqa: E402
from data import list_images  # noqa: E402
from evaluate import _area, _inter  # noqa: E402
from paths import OUTPUT_DIR, imread  # noqa: E402

FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


def main():
    ann = {im["source"]: [(b["class"], *b["bbox_xyxy"]) for b in im["boxes"]] for im in json.load(open(PC.ANN))["images"]}
    pre_cfg, det = R.load_config(R.PRE_CFG), R.load_detection_config(R.DET_CFG)
    cfg = {**R.DEFAULT_CFG, **det["detect_cfg"], "line_hi_abs": V.HI}
    _, _, images = list_images("provided")
    TW, rows = 560, []
    for path in images:
        raw = imread(path)
        t = max(2, raw.shape[1] // 280)
        gts = ann.get(path.name, [])
        left = raw.copy()
        for c, x1, y1, x2, y2 in gts:
            cv2.rectangle(left, (x1, y1), (x2, y2), (0, 0, 255) if c == "crack" else (0, 165, 255), t + 1)
        right = raw.copy()
        cg = [g[1:] for g in gts if g[0] == "crack"]
        for x1, y1, x2, y2 in cg:
            cv2.rectangle(right, (x1, y1), (x2, y2), (255, 255, 255), t)
        meta, img, mask, geo = R.prepare(path, pre_cfg, det, None)
        n_hit = n_fake = 0
        if img is not None:
            tags = R.runtime_tags(path.name, img, mask, geo, PC.BLUR, lambda n: [])
            proc, _ = R.apply_extra(img, mask, PC.BLUR, tags)
            for d in R.detect(proc, cfg):
                if d["type"] != "crack" or R.road_overlap(mask, d["bbox"]) < 0.5:
                    continue
                x, y, w, h = R.to_original_bbox(d["bbox"], geo)
                b = (x, y, x + w, y + h)
                hit = any(_inter(b, g) / max(_area(b), 1) >= 0.5 for g in cg)
                n_hit += hit; n_fake += not hit
                cv2.rectangle(right, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0) if hit else (0, 0, 255), t * (2 if hit else 1))
        found = sum(any(_inter((x, y, x + w, y + h), g) / max(w * h, 1) >= 0.5 for (x, y, w, h) in []) for g in cg)
        h = int(TW * raw.shape[0] / raw.shape[1])
        rows.append((path.stem, [cv2.resize(v, (TW, h), interpolation=cv2.INTER_AREA) for v in (left, right)],
                     len(cg), sum(g[0] == "pothole" for g in gts), n_hit, n_fake, img is None))
    cols = 2                                           # 사진 두 장씩 (각 사진 = 정답 · 검출 나란히)
    W = 40 + cols * (2 * TW + 60)
    heights = [max(r[1][0].shape[0] for r in rows[i:i + cols]) + 70 for i in range(0, len(rows), cols)]
    canvas = Image.new("RGB", (W, 150 + sum(heights)), "white")
    d = ImageDraw.Draw(canvas)
    F = lambda s, i=0: ImageFont.truetype(FONT, s, index=i)
    d.text((40, 18), "제공 13장 — 정답(사용자 초안) vs 우리 검출 (F + 흐림 대응) · 균열만 채점", font=F(32, 6), fill=(15, 23, 42))
    d.text((40, 66), "각 사진 왼쪽 = 정답 (빨강 균열 · 주황 포트홀)  ·  오른쪽 = 우리 균열 후보 (초록 굵게 = 정답 균열을 맞힘 · 빨강 = 가짜) + 흰 선 = 정답 균열 위치", font=F(20), fill=(70, 70, 70))
    d.text((40, 100), "합계: 정답 균열 25개 중 9개 찾음 (R 0.360) · P 0.437 · 가짜 3.1/장 — 정답은 AI로 그린 초안(검수 전)", font=F(20, 6), fill=(180, 30, 30))
    y = 150
    for i in range(0, len(rows), cols):
        for j, (stem, (l, r), nc, npot, nh, nf, failed) in enumerate(rows[i:i + cols]):
            x = 40 + j * (2 * TW + 60)
            canvas.paste(Image.fromarray(cv2.cvtColor(l, cv2.COLOR_BGR2RGB)), (x, y))
            canvas.paste(Image.fromarray(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)), (x + TW + 8, y))
            cap = f"{stem} · 정답 균열 {nc}" + (f" · 포트홀 {npot}" if npot else "") + f"  →  맞힌 후보 {nh} · 가짜 {nf}" + (" · 전처리 실패" if failed else "")
            d.text((x, y + l.shape[0] + 8), cap, font=F(19, 6), fill=(30, 30, 30))
        y += heights[i // cols]
    out = OUTPUT_DIR / "provided_check" / "compare_sheet.jpg"
    canvas.save(out, quality=90)
    print(out)


if __name__ == "__main__":
    main()
