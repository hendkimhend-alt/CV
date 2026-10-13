"""
육안 확인용 — 최종 구성 F로 제공 13장 · 직접 촬영분을 돌려 후보 박스만 그린다 (성능 측정 안 함, 10/12 사용자 요청).

F = 팀원 전처리 v1.1 (노면 마스크 · 조건부 감마) → 연결 → 가우시안 σ0.8 → 막대 닫힘 7px → CLAHE 1.0 (노면 안)
    → 검출기 p2bd (강한 기준 132.172, 개발 세트에서 맞춘 값) → 도로 위 판정
그림: 노면 마스크(옅은 파랑 칠 + 경계선) · 균열 후보(빨강) · 포트홀 후보(주황) · 정답은 그리지 않음 · 범례는 그림 위 띠
실행: python analysis/visual_check.py → outputs/visual_check/<데이터>/*.jpg + 한 장 모음
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import run_road_pipeline as R  # noqa: E402
from data import list_images  # noqa: E402
from paths import OUTPUT_DIR, imread  # noqa: E402

EXTRA = {"gaussian": {"ksize": 3, "sigma": 0.8}, "line_close": {"length": 7, "angles": 12},
         "clahe": {"clip": 1.0, "tile": 8}, "order": ["gaussian", "line_close", "clahe"]}
HI = 132.17221069335938          # F (`t_c10_post` run_20261011_072601)의 맞춘 강한 기준
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
LEGEND = [((255, 200, 0), "옅은 파랑 칠 + 파란 선 = 전처리가 '도로'로 본 영역 (검출은 이 안의 후보만 남김)"), ((0, 0, 255), "균열 후보"), ((0, 150, 255), "포트홀 후보")]


def legend_bar(width, title, size=22):
    """그림 위에 붙일 범례 띠 (BGR)."""
    bar = Image.new("RGB", (width, size * 2 + 30), "white")
    d = ImageDraw.Draw(bar)
    d.text((12, 8), title, font=ImageFont.truetype(FONT, size, index=6), fill=(15, 23, 42))
    x = 12
    for bgr, text in LEGEND:
        d.rectangle([x, size + 22, x + 34, size + 40], outline=bgr[::-1], width=4)
        d.text((x + 44, size + 16), text, font=ImageFont.truetype(FONT, size - 4), fill=(40, 40, 40))
        x += 60 + int(d.textlength(text, font=ImageFont.truetype(FONT, size - 4)))
    return cv2.cvtColor(np.array(bar), cv2.COLOR_RGB2BGR)


def run(dataset):
    pre_cfg, det = R.load_config(R.PRE_CFG), R.load_detection_config(R.DET_CFG)
    cfg = {**R.DEFAULT_CFG, **det["detect_cfg"], "line_hi_abs": HI}
    _, _, images = list_images(dataset)
    out = OUTPUT_DIR / "visual_check" / dataset
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for path in images:
        meta, img, mask, geo = R.prepare(path, pre_cfg, det, None)
        vis = imread(path)
        t = max(2, vis.shape[1] // 400)
        n = {"crack": 0, "pothole": 0}
        if img is not None:
            proc, _ = R.apply_extra(img, mask, EXTRA, set())
            # 노면 마스크 경계를 원본 좌표로
            full = np.zeros(vis.shape[:2], np.uint8)
            m = cv2.resize(mask, (geo["crop_width"], geo["crop_height"]), interpolation=cv2.INTER_NEAREST)
            full[geo["crop_y"]:geo["crop_y"] + geo["crop_height"], geo["crop_x"]:geo["crop_x"] + geo["crop_width"]] = m
            tint = vis.copy()
            tint[full > 0] = (255, 200, 0)
            vis = cv2.addWeighted(vis, 0.78, tint, 0.22, 0)          # 도로로 본 영역을 옅게 칠함
            cs, _ = cv2.findContours((full > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (255, 200, 0), t)
            for d in R.detect(proc, cfg):
                if R.road_overlap(mask, d["bbox"]) >= det["integration"]["min_road_overlap"]:
                    x, y, w, h = R.to_original_bbox(d["bbox"], geo)
                    cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255) if d["type"] == "crack" else (0, 150, 255), t)
                    n[d["type"]] += 1
        label = f"{path.stem}  crack {n['crack']} / pothole {n['pothole']}" + ("" if img is not None else "  (preprocess failed)")
        bar = legend_bar(vis.shape[1], f"{path.stem} — 균열 후보 {n['crack']}개 · 포트홀 후보 {n['pothole']}개" + ("" if img is not None else " · 전처리 실패 (노면 마스크 없음)"),
                         size=max(18, vis.shape[1] // 50))
        cv2.imwrite(str(out / f"{path.stem}.jpg"), np.vstack([bar, vis]))
        th = cv2.resize(vis, (480, int(480 * vis.shape[0] / vis.shape[1])), interpolation=cv2.INTER_AREA)
        th = cv2.copyMakeBorder(th, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(th, label, (6, th.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(cv2.resize(th, (480, 400)) if th.shape[0] > 400 else cv2.copyMakeBorder(th, 0, 400 - th.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        print(f"  {label}", flush=True)
    cols = 4
    rows = [np.hstack(tiles[i:i + cols] + [np.full_like(tiles[0], 255)] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    title = {"provided": "제공 13장", "captured": "직접 촬영"}.get(dataset, dataset) + " — 최종 구성 F로 돌린 결과 (박스만, 성능 측정 안 함)"
    cv2.imwrite(str(out / "_sheet.jpg"), np.vstack([legend_bar(sheet.shape[1], title, 30), sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"→ {out}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ("provided", "captured"):
        run(ds)
