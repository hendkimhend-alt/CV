"""
균열 진단 그림 — 사진마다 "처음 찾은 균열 흔적"과 "버려진 후보 · 이유"를 한 장에 (측정 안 함, 10/12 사용자 요청).

최종 구성 F (analysis/visual_check.py와 같음)로 돌리고, 균열 갈래만 그린다.
  노면 마스크 = 옅은 파랑 칠 + 파란 선 (ROI)
  처음 찾은 균열 흔적 = 노란 선 (찾기 → 잇기 뒤 1px 흔적 — 이중 임계값을 통과한 것)
  후보 덩어리 (흔적을 이은 덩어리) = 박스, 버린 이유마다 색 — 남은 균열은 굵은 빨강
버린 이유는 detect()의 판정 순서 그대로: 길이 < 45 → 세장비 < 3 → 사진 위 테두리에 닿음 → 양쪽 확인 < 0.35
  → 노면 단서(구조 세기 · 정렬 · 색) → 깊이 < 0.267 → (검출 뒤) 도로 위 판정 (박스의 50% 미만이 노면)
실행: python analysis/crack_diagnosis.py provided → outputs/crack_diagnosis/provided/*.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import run_road_pipeline as R  # noqa: E402
import visual_check as V  # noqa: E402
from data import list_images  # noqa: E402
from paths import OUTPUT_DIR, imread  # noqa: E402

LINE_TRACE = R.detect.__globals__["line_trace"]
TOUCH = R.detect.__globals__["_touches_border"]
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
# 이유 → (BGR 색, 설명)
REASONS = {
    "남음": ((0, 0, 255), "최종 균열 (남음)"),
    "짧음": ((160, 160, 160), "길이 < 45px"),
    "세장비": ((60, 120, 200), "길쭉하지 않음 (세장비 < 3)"),
    "테두리": ((0, 90, 160), "사진 위 테두리에 닿음"),
    "양쪽 확인": ((255, 120, 0), "양쪽 확인 < 0.35 (한쪽만 어두움: 그림자 · 차선 경계)"),
    "노면 단서": ((200, 0, 200), "노면 단서 (노면 밖처럼 보임)"),
    "깊이": ((150, 0, 255), "깊이 < 0.267 (페인트 · 노면 밖)"),
    "도로 밖": ((90, 90, 90), "도로 위 판정 (박스의 50% 미만이 노면)"),
}


def reason(d, cfg, W, H, mask):
    if d["type"] == "crack":
        return "남음" if R.road_overlap(mask, d["bbox"]) >= 0.5 else "도로 밖"
    if d["length"] < cfg["crack_min_length"]:
        return "짧음"
    if d["elong"] < cfg["crack_min_elong"] or d["contrast"] < cfg["crack_min_contrast"]:
        return "세장비"
    if TOUCH(d["bbox"], W, H, cfg):
        return "테두리"
    if d.get("valley") is not None and d["valley"] < cfg["valley_min_ratio"]:
        return "양쪽 확인"
    cv = d.get("cues") or {}
    if "depth" in cv and cv["depth"] < cfg["cue_filter"].get("depth_min", -1e9):
        return "깊이"
    return "노면 단서"


def legend(width, title, counts, size):
    f, fb = ImageFont.truetype(FONT, size), ImageFont.truetype(FONT, size, index=6)
    bar = Image.new("RGB", (width, size * 2 + 20 + (size + 12) * ((len(REASONS) + 1) // 2) + 50), "white")
    d = ImageDraw.Draw(bar)
    d.text((12, 8), title, font=fb, fill=(15, 23, 42))
    y = size + 22
    d.rectangle([12, y + 4, 48, y + size - 2], fill=(200, 235, 255), outline=(0, 200, 255), width=3)
    d.text((58, y), "옅은 파랑 = 노면 마스크(ROI)  ·  노란 선 = 처음 찾은 균열 흔적 (찾기 → 잇기 뒤)", font=f, fill=(40, 40, 40))
    y += size + 14
    half = width // 2
    for i, (k, (bgr, text)) in enumerate(REASONS.items()):
        x = 12 + (i % 2) * half
        yy = y + (i // 2) * (size + 12)
        d.rectangle([x, yy + 4, x + 36, yy + size - 2], outline=bgr[::-1], width=4 if k == "남음" else 3)
        d.text((x + 46, yy), f"{text}: {counts.get(k, 0)}개", font=fb if k == "남음" else f, fill=(40, 40, 40))
    return cv2.cvtColor(np.array(bar), cv2.COLOR_RGB2BGR)


def run(dataset):
    pre_cfg, det = R.load_config(R.PRE_CFG), R.load_detection_config(R.DET_CFG)
    cfg = {**R.DEFAULT_CFG, **det["detect_cfg"], "line_hi_abs": V.HI}
    _, _, images = list_images(dataset)
    out = OUTPUT_DIR / "crack_diagnosis" / dataset
    out.mkdir(parents=True, exist_ok=True)
    for path in images:
        meta, img, mask, geo = R.prepare(path, pre_cfg, det, None)
        if img is None:
            vis = imread(path)
            bar = legend(vis.shape[1], f"{path.stem} — 전처리 실패 (노면 마스크 없음) → 검출 안 함", {}, max(18, vis.shape[1] // 60))
            cv2.imwrite(str(out / f"{path.stem}.jpg"), np.vstack([bar, vis]))
            print(f"  {path.stem}: 전처리 실패")
            continue
        proc, _ = R.apply_extra(img, mask, V.EXTRA, set())
        H, W = proc.shape[:2]
        trace, _, _ = LINE_TRACE(R._to_gray(proc), cfg)
        vis = proc.copy()
        tint = vis.copy(); tint[mask > 0] = (255, 200, 0)
        vis = cv2.addWeighted(vis, 0.8, tint, 0.2, 0)
        cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (255, 200, 0), 2)
        vis[cv2.dilate(trace, np.ones((2, 2), np.uint8)) > 0] = (0, 255, 255)
        counts, kept = {}, []
        for d in R.detect(proc, {**cfg, "keep_noise": True}):
            if d.get("branch") != "crack":
                continue
            r = reason(d, cfg, W, H, mask)
            counts[r] = counts.get(r, 0) + 1
            if r == "짧음":
                continue                                   # 짧은 흔적은 노란 선으로만 (박스가 너무 많아짐)
            x, y, w, h = (int(v) for v in d["bbox"])
            if r == "남음":
                kept.append((x, y, w, h))
            else:
                cv2.rectangle(vis, (x, y), (x + w, y + h), REASONS[r][0], 2)
        for x, y, w, h in kept:                               # 남은 균열은 맨 위에 굵게
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
        title = f"{path.stem} — 최종 균열 {counts.get('남음', 0)}개 · 처음 찾은 후보 덩어리 {sum(counts.values())}개 (검출기 입력, 배율 {geo['scale_x']:.2f})"
        cv2.imwrite(str(out / f"{path.stem}.jpg"), np.vstack([legend(W, title, counts, max(18, W // 55)), vis]))
        print(f"  {path.stem}: " + " · ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"→ {out}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ("provided",):
        run(ds)
