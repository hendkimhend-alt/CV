"""
골재 질감 가짜의 특성 — B의 가짜 표본(results/analysis/b_fakes.csv, 사람이 종류를 붙인 150개) 중 노면 질감(E)과
진짜 균열 조각(정답 박스 안 후보, 무작위 150개)을 검출기 입력 척도(긴 변 1024)에서 비교한다 (개발 세트 · 분석만, 10/10).

재는 것 (후보 하나마다, detect()가 내는 값): 길이(px, 1px 줄기 길이) · 폭 · 세장비 · 진하기 · 사진 안 높이(0 위 ~ 1 아래)
사진마다: 연결 단계의 확대 배율(det_scale = 검출기 입력 / 원본 노면 영역, 1보다 크면 키움) ↔ 가짜/장
→ 골재 질감을 지우는 전처리의 크기(막대 길이 등)를 데이터로 정하는 근거.

실행: python analysis/texture_traits.py --run <B 실행 (results.csv에 det_scale 있는 것)> --cache <run_road_pipeline 캐시>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import run_road_pipeline as R  # noqa: E402  (CV-main 실행 함수 + 맨 위 src 채점 함수)
from data import list_images, load_gt  # noqa: E402
from evaluate import _area, _inter, _xyxy  # noqa: E402
from paths import OUTPUT_DIR  # noqa: E402

OUT = ROOT / "results" / "analysis" / "texture_traits.csv"
SEED = 20261010


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", required=True, type=Path)
    a = ap.parse_args()
    run = OUTPUT_DIR / "pipeline" / a.run
    rows = {r["image"]: r for r in csv.DictReader(open(run / "results.csv", encoding="utf-8-sig"))}
    dets = json.load(open(run / "detections.json"))
    fakes = list(csv.DictReader(open(ROOT / "results" / "analysis" / "b_fakes.csv", encoding="utf-8-sig")))
    name, _, images = list_images("rdd_dev")
    P = {p.name: p for p in images}
    gt_of = load_gt("rdd_dev", name)
    # 진짜 균열 조각: 정답 박스 안(in50) 균열 후보 중 무작위 150개
    inside = []
    for n, ds in dets.items():
        gts = [g[1:] for g in (gt_of(P[n]) or []) if g[0] == "crack"]
        for d in ds:
            b = _xyxy(d["bbox"])
            if d["type"] == "crack" and any(_inter(b, g) / max(_area(b), 1) >= 0.5 for g in gts):
                inside.append({"image": n, "bbox": d["bbox"]})
    pick = [inside[i] for i in np.random.default_rng(SEED).choice(len(inside), 150, replace=False)]
    targets = [("E", f["image"], [float(f["x1"]), float(f["y1"]), float(f["x2"]) - float(f["x1"]), float(f["y2"]) - float(f["y1"])])
               for f in fakes if f["kind"] == "E" and f["type"] == "crack"]
    targets += [("crack", t["image"], t["bbox"]) for t in pick]

    pre_cfg, det = R.load_config(R.PRE_CFG), R.load_detection_config(R.DET_CFG)
    cfg = {**R.DEFAULT_CFG, **det["detect_cfg"]}
    cache_det, out = {}, []
    for kind, n, bb in targets:
        if n not in cache_det:
            meta, img, mask, geo = R.prepare(P[n], pre_cfg, det, a.cache)
            cache_det[n] = (img.shape[0], geo, [d for d in R.detect(img, cfg)
                                                if R.road_overlap(mask, d["bbox"]) >= det["integration"]["min_road_overlap"]])
        h, geo, ds = cache_det[n]
        best = min(ds, key=lambda d: sum(abs(u - v) for u, v in zip(R.to_original_bbox(d["bbox"], geo), bb)))
        err = sum(abs(u - v) for u, v in zip(R.to_original_bbox(best["bbox"], geo), bb))
        x, y, w, hh = best["bbox"]
        out.append({"kind": kind, "image": n, "match_err": round(err, 1), "length": round(best["length"], 1),
                    "width": round(best["width"], 2), "elong": round(best["elong"], 2), "contrast": round(best["contrast"], 2),
                    "y_rel": round((y + hh / 2) / h, 3), "det_scale": geo["scale_x"],
                    "group": rows[n]["group"], "fakes_in_image": int(rows[n]["crack_in50_fp"])})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"→ {OUT.relative_to(ROOT)} · 맞춤 오차 > 3px: {sum(o['match_err'] > 3 for o in out)}개")
    for k in ("E", "crack"):
        s = [o for o in out if o["kind"] == k]
        q = lambda key: np.percentile([o[key] for o in s], [25, 50, 75])
        print(f"{k:6} n={len(s)} 길이 {q('length').round(1)} 폭 {q('width').round(2)} 세장비 {q('elong').round(1)} "
              f"진하기 {q('contrast').round(1)} 높이 {q('y_rel').round(2)} 확대 {q('det_scale').round(2)}")
    # 사진마다: 확대 배율 ↔ 가짜/장
    sc = [(float(r["det_scale"]), int(r["crack_in50_fp"])) for r in rows.values() if r.get("det_scale")]
    for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99)):
        v = [f for s, f in sc if lo <= s < hi]
        if v:
            print(f"확대 {lo}~{hi}: 사진 {len(v)} · 균열 가짜/장 {np.mean(v):.1f} (중앙 {np.median(v):.0f})")


if __name__ == "__main__":
    main()
