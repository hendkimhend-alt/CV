"""
특징점 추출 · 매칭으로 전처리를 검사한다 (측정만 — 검출 결과는 바꾸지 않음, 개발 세트, 10/12).

과제 비교 항목 "특징점 수"(전처리 전 · 후)를 채우고, 보정이 노면 구조를 유지했는지 · 없던 무늬를 만들었는지 본다.
단계 (검출기 입력 = 팀원 전처리 + 연결 뒤, 노면 마스크 안에서만):
  s0 원본       — B의 검출기 입력 (보정 없음)
  s1 질감 처리  — 가우시안 σ0.8 → 막대 닫힘 7px
  s2 최종       — s1 → CLAHE 1.0 (지금 채택 구성)
사진마다: SIFT 특징점 수 · s0과 매칭(비율 검사 0.75 + 같은 자리 2px 이내 = 같은 점) → 다시 잡힌 비율 · 새로 생긴 점 수
          · 균열 정답 박스 안 / 밖(노면) 특징점 밀도 (1000px²당)

실행: python analysis/keypoint_check.py --cache <run_road_pipeline 캐시> → results/analysis/keypoint_check.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import run_road_pipeline as R  # noqa: E402
from data import list_images, load_gt  # noqa: E402

OUT = ROOT / "results" / "analysis" / "keypoint_check.csv"
TEX = {"gaussian": {"ksize": 3, "sigma": 0.8}, "line_close": {"length": 7, "angles": 12}, "order": ["gaussian", "line_close"]}
FINAL = {**TEX, "clahe": {"clip": 1.0, "tile": 8}, "order": ["gaussian", "line_close", "clahe"]}
RATIO, SAME_PX = 0.75, 2.0


def sift_in(img, mask, sift):
    return sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (mask > 0).astype(np.uint8) * 255)


def match(k0, d0, k1, d1, bf):
    """→ (같은 점으로 다시 잡힌 수, 새로 생긴 점 수). 같은 점 = 설명자 비율 검사 통과 + 위치 2px 이내 (두 영상은 정렬돼 있음)."""
    if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
        return 0, len(k1)
    same = 0
    for m in bf.knnMatch(d1, d0, k=2):
        if len(m) == 2 and m[0].distance < RATIO * m[1].distance:
            p, q = np.array(k1[m[0].queryIdx].pt), np.array(k0[m[0].trainIdx].pt)
            if np.hypot(*(p - q)) <= SAME_PX:
                same += 1
    p0 = np.array([k.pt for k in k0])
    new = sum(np.min(np.hypot(*(p0 - np.array(k.pt)).T)) > SAME_PX for k in k1)
    return same, int(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    pre_cfg, det = R.load_config(R.PRE_CFG), R.load_detection_config(R.DET_CFG)
    name, _, images = list_images("rdd_dev")
    images = images[:a.limit] if a.limit else images
    gt_of = load_gt("rdd_dev", name)
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2)
    group = {r["image"]: r["group"] for r in csv.DictReader(open(R.SPLIT_CSV, encoding="utf-8"))}
    rows = []
    for i, path in enumerate(images, 1):
        meta, img, mask, geo = R.prepare(path, pre_cfg, det, a.cache)
        if img is None:
            continue
        road = mask > 0
        gtm = np.zeros(road.shape, bool)
        for g in gt_of(path) or []:
            if g[0] != "crack":
                continue
            x1, y1 = (g[1] - geo["crop_x"]) * geo["scale_x"], (g[2] - geo["crop_y"]) * geo["scale_y"]
            x2, y2 = (g[3] - geo["crop_x"]) * geo["scale_x"], (g[4] - geo["crop_y"]) * geo["scale_y"]
            gtm[max(int(y1), 0):max(int(y2), 0), max(int(x1), 0):max(int(x2), 0)] = True
        gt_area, bg_area = np.count_nonzero(gtm & road), np.count_nonzero(~gtm & road)
        stages = {"s0": img, "s1": R.apply_extra(img, mask, TEX, set())[0], "s2": R.apply_extra(img, mask, FINAL, set())[0]}
        kd = {s: sift_in(v, mask, sift) for s, v in stages.items()}
        row = {"image": path.name, "group": group.get(path.name, ""), "road_px": int(road.sum())}
        for s, (k, d) in kd.items():
            pts = np.array([kp.pt for kp in k]).astype(int) if k else np.zeros((0, 2), int)
            in_gt = int(gtm[pts[:, 1], pts[:, 0]].sum()) if len(pts) else 0
            row[f"{s}_kp"] = len(k)
            row[f"{s}_kp_gt_per_1k"] = round(in_gt / gt_area * 1000, 3) if gt_area else None
            row[f"{s}_kp_bg_per_1k"] = round((len(k) - in_gt) / bg_area * 1000, 3) if bg_area else None
        for s in ("s1", "s2"):
            same, new = match(*kd["s0"], *kd[s], bf)
            row[f"{s}_same"], row[f"{s}_new"] = same, new
        rows.append(row)
        if i % 100 == 0:
            print(f"  [{i}/{len(images)}]", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tot = lambda k: sum(r[k] for r in rows)
    mean = lambda k: np.mean([r[k] for r in rows if r[k] is not None])
    print(f"→ {OUT.relative_to(ROOT)} · 사진 {len(rows)}장")
    for s, lab in (("s0", "원본"), ("s1", "질감 처리"), ("s2", "최종 (+CLAHE)")):
        line = f"{lab:12} 특징점 {tot(s + '_kp') / len(rows):7.1f}/장 · 균열 박스 안 {mean(s + '_kp_gt_per_1k'):.3f} · 밖(노면) {mean(s + '_kp_bg_per_1k'):.3f} (1000px²당)"
        if s != "s0":
            line += f" · 원본 점이 다시 잡힘 {tot(s + '_same') / tot('s0_kp'):.1%} · 새로 생긴 점 {tot(s + '_new') / tot(s + '_kp'):.1%}"
        print(line)


if __name__ == "__main__":
    main()
