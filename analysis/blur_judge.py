"""
흐린 사진 판단 — 파이프라인이 사진을 보고 직접 정하는 선명도 기준 (개발 세트, 10/12).

선명도 = 노면 마스크(5×5로 깎음) 안 회색의 Laplacian 분산, 사진 전체 긴 변을 1024로 맞춘 척도에서 잼
  (팀원 전처리의 품질 측정과 같은 척도 · 1차 흐림 분류 "선명도 < 50"과 같은 지표)
  검출기 입력(노면을 잘라 긴 변 1024)에서 원래 척도로 되돌려 잼: 배율 = (1024 / 원본 긴 변) / 검출기 배율
기준값: 정답(균열 박스)을 보지 않고, 1차 흐림 라벨(labels/split_rdd.csv의 blur — 사진 품질만으로 만든 분류)과
  가장 잘 맞는 값 → run_road_pipeline --extra의 "blur_thr"
결과 (10/12): 일치율 최대 5.7 (흐림 라벨의 41%만 잡음) · 잡는 비율 − 오판 비율 최대 47 (흐림 라벨 100% · 정상 오판 25%)
  → 1차 기준값 50과 거의 같아 50을 그대로 씀 (201 ~ 207장이 흐림으로 판단됨)

실행: python analysis/blur_judge.py --cache <run_road_pipeline 캐시>
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
from data import list_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    a = ap.parse_args()
    pre_cfg, det = R.load_config(R.PRE_CFG), R.load_detection_config(R.DET_CFG)
    _, _, images = list_images("rdd_dev")
    label = {r["image"]: "blur" in r["group"].split(";") for r in csv.DictReader(open(R.SPLIT_CSV, encoding="utf-8"))}
    v, y = [], []
    for p in images:
        meta, img, mask, geo = R.prepare(p, pre_cfg, det, a.cache)
        if img is None:
            continue
        v.append(R.sharpness(img, mask, geo)); y.append(label.get(p.name, False))
    v, y = np.array(v), np.array(y)
    best = max(np.unique(np.round(v, 1)), key=lambda t: np.mean((v < t) == y))
    print(f"사진 {len(v)} · 1차 흐림 라벨 {y.sum()}장 · 선명도 중앙값: 흐림 {np.median(v[y]):.1f} / 나머지 {np.median(v[~y]):.1f}")
    for t in sorted({best, 20.0, 50.0, 100.0}):
        pred = v < t
        print(f"  기준 {t:7.1f}: 흐림으로 판단 {pred.sum():3}장 · 라벨과 일치 {np.mean(pred == y):.1%} · 라벨 흐림 중 잡음 {np.mean(pred[y]):.0%} · 라벨 정상인데 흐림 판단 {np.mean(pred[~y]):.0%}")
    ts = np.unique(np.round(v, 1))
    youden = max(ts, key=lambda t: np.mean((v < t)[y]) - np.mean((v < t)[~y]))
    pred = v < youden
    print(f"→ 일치율 최대 {best} · 흐림 잡는 비율 − 정상 오판 비율 최대 {youden} (잡음 {np.mean(pred[y]):.0%} · 오판 {np.mean(pred[~y]):.0%} · 흐림 판단 {pred.sum()}장)")


if __name__ == "__main__":
    main()
