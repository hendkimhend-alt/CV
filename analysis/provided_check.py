"""
제공 13장 확인 — 사용자가 만든 초안 정답(data/road_damage_annotated/annotations.json, 검수 전)으로 균열 in50 채점 (10/12).

설정을 고르는 데 쓰지 않는다 (마지막 확인용 데이터). 쓸 때마다 results/experiments.md "제공 · 촬영 데이터 확인 기록"에 남긴다.
구성: F (최종) · F + 흐림 대응 (아주 흐림 < 2.76 → 질감 처리 건너뜀 + 언샤프 3.0 · σ3)
그림: outputs/provided_check/<구성>/<사진>.jpg — 초록 = 정답 균열(초안) · 빨강 = 균열 후보 (맞힘 굵게)
실행: python analysis/provided_check.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import run_road_pipeline as R  # noqa: E402
import visual_check as V  # noqa: E402
from data import list_images  # noqa: E402
from evaluate import _area, _inter, evaluate, prf_in50  # noqa: E402
from paths import OUTPUT_DIR, imread  # noqa: E402

ANN = ROOT / "data" / "road_damage_annotated" / "annotations.json"
BLUR = {"gaussian": {"ksize": 3, "sigma": 0.8, "apply_to": "not_blur_rt", "blur_thr": 2.76},
        "line_close": {"length": 7, "angles": 12, "apply_to": "not_blur_rt"},
        "unsharp": {"amount": 3.0, "sigma": 3.0, "apply_to": "blur_rt"}, "clahe": {"clip": 1.0, "tile": 8},
        "order": ["gaussian", "line_close", "unsharp", "clahe"]}
CONFIGS = {"F": V.EXTRA, "F+흐림대응": BLUR}


def main():
    ann = {im["source"]: [(b["class"], *b["bbox_xyxy"]) for b in im["boxes"]] for im in json.load(open(ANN))["images"]}
    pre_cfg, det = R.load_config(R.PRE_CFG), R.load_detection_config(R.DET_CFG)
    cfg = {**R.DEFAULT_CFG, **det["detect_cfg"], "line_hi_abs": V.HI}
    _, _, images = list_images("provided")
    for name, extra in CONFIGS.items():
        out = OUTPUT_DIR / "provided_check" / name
        out.mkdir(parents=True, exist_ok=True)
        tot = {"tp": 0, "fp": 0, "fn": 0, "n": 0}
        print(f"[{name}]")
        for path in images:
            gts = [g for g in ann.get(path.name, []) if g[0] == "crack"]
            meta, img, mask, geo = R.prepare(path, pre_cfg, det, None)
            dets = []
            if img is not None:
                tags = R.runtime_tags(path.name, img, mask, geo, extra, lambda n: [])
                proc, _ = R.apply_extra(img, mask, extra, tags)
                for d in R.detect(proc, cfg):
                    if d["type"] == "crack" and R.road_overlap(mask, d["bbox"]) >= det["integration"]["min_road_overlap"]:
                        dets.append({"type": "crack", "bbox": list(R.to_original_bbox(d["bbox"], geo))})
            e = evaluate(dets, gts, None)["crack"]["in50"]
            for k in ("tp", "fp", "fn"):
                tot[k] += e[k]
            tot["n"] += len(dets)
            vis = imread(path)
            t = max(2, vis.shape[1] // 300)
            for g in gts:
                cv2.rectangle(vis, (int(g[1]), int(g[2])), (int(g[3]), int(g[4])), (0, 200, 0), t + 1)
            for d in dets:
                x, y, w, h = d["bbox"]
                b = (x, y, x + w, y + h)
                hit = any(_inter(b, g[1:]) / max(_area(b), 1) >= 0.5 for g in gts)
                cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), t * (2 if hit else 1))
            cv2.imwrite(str(out / f"{path.stem}.jpg"), vis)
            print(f"  {path.stem:24} 정답 균열 {len(gts)} · 찾음 {e['tp']} · 후보 {len(dets)} · 가짜 {e['fp']}")
        p, r, f1 = prf_in50(tot["tp"], tot["fp"], tot["fn"], tot["n"])
        print(f"  → 합계 정답 {tot['tp'] + tot['fn']} · R {r:.3f} · P {p:.3f} · F1 {f1:.3f} · 가짜/장 {tot['fp'] / len(images):.1f}")


if __name__ == "__main__":
    main()
