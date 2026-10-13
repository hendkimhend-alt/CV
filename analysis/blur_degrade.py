"""
인공 흐림 평가셋 — 개발 세트 사진에 촬영 흐림을 일부러 걸어 "아주 흐린 사진"을 만든다 (10/12).

개발 세트에는 아주 흐린 사진이 없다 (가장 흐린 것도 낮은 해상도 · 압축 · 뿌연 대기 정도). 제공 13장 중 흐림 4장은
1차 계획서에서 선명도(원본 해상도 사진 전체의 Laplacian 분산) 3 ~ 6 — 이 사진들은 마지막 확인용이라 실험에 쓰지 않고,
개발 세트에 흐림을 걸어 같은 선명도 수준을 만든다. 흐림은 위치를 바꾸지 않으므로 RDD 정답 박스를 그대로 쓴다.

흐림 종류 (원본 사진에, 팀원 전처리보다 앞 — 촬영할 때 생기는 흐림):
  {"kind": "gauss",  "k": …}  초점 흐림: 가우시안 σ = k × (긴 변 / 1024)
  {"kind": "motion", "k": …}  흔들림: 길이 L = k × (긴 변 / 1024) 의 직선 커널, 방향은 사진마다 고정 난수 (seed = 파일 이름)
  → 해상도가 달라도 같은 정도로 흐려지게 긴 변에 비례
  {"kind": "mixed", "frac": 0.25, "ks": [2, 3.5]}  섞인 흐림: 사진의 frac만 (파일 이름 해시로 고정) 가우시안, k는 ks 중 하나
  → 실제처럼 "대부분 선명 · 일부 흐림"이고 강한 기준은 선명한 사진에 맞춘 값 그대로일 때를 재현
세기 고르기: python analysis/blur_degrade.py --calibrate → 개발 세트 표본에서 선명도 중앙값이 1차 흐림 4장 범위(3 ~ 6)의
  가운데(약 4.5)가 되는 k (정답 안 봄)
쓰는 곳: python analysis/run_road_pipeline.py --degrade '{"kind": "gauss", "k": …}' --cache <흐림 전용 캐시> …
"""
import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
TARGET = 4.5     # 1차 계획서 제공 흐림 4장의 선명도 3 ~ 6 의 가운데


def blurred_in_mixed(name, spec):
    """섞인 흐림에서 이 사진이 흐려지는가 · 어떤 k인가 (파일 이름 해시로 고정)."""
    h = int(hashlib.md5(("mixed:" + name).encode()).hexdigest(), 16)
    if (h % 10000) / 10000 >= spec["frac"]:
        return None
    return spec["ks"][(h // 10000) % len(spec["ks"])]


def degrade(img, spec, name=""):
    if not spec:
        return img
    if spec["kind"] == "mixed":
        k = blurred_in_mixed(name, spec)
        return img if k is None else degrade(img, {"kind": "gauss", "k": k}, name)
    s = spec["k"] * max(img.shape[:2]) / 1024.0
    if spec["kind"] == "gauss":
        return cv2.GaussianBlur(img, (0, 0), s, borderType=cv2.BORDER_REFLECT_101)
    if spec["kind"] == "motion":
        L = max(3, int(round(s)) | 1)
        ang = int(hashlib.md5(name.encode()).hexdigest(), 16) % 180
        k = np.zeros((L, L), np.float32)
        c = (L - 1) / 2
        dx, dy = np.cos(np.radians(ang)) * c, np.sin(np.radians(ang)) * c
        cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1.0, 1)
        return cv2.filter2D(img, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT_101)
    raise ValueError(f"error:모르는 흐림 {spec}")


def sharpness_1st(img):
    """1차 계획서의 선명도: 원본 해상도 사진 전체 회색의 Laplacian 분산 (과제1/analysis/analyze_dataset.py)."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    from data import list_images
    from paths import imread
    _, _, ims = list_images("rdd_dev")
    pick = [ims[i] for i in np.random.default_rng(20261012).choice(len(ims), a.n, replace=False)]
    imgs = [(p.name, imread(p)) for p in pick]
    print(f"개발 표본 {len(imgs)}장 · 흐림 전 선명도 중앙값 {np.median([sharpness_1st(i) for _, i in imgs]):.1f} (1차 흐림 4장 목표 3 ~ 6)")
    for kind, ks in (("gauss", (1, 1.5, 2, 3, 4, 5)), ("motion", (5, 9, 13, 17, 21, 25))):
        res = {k: float(np.median([sharpness_1st(degrade(i, {"kind": kind, "k": k}, n)) for n, i in imgs])) for k in ks}
        best = min(res, key=lambda k: abs(res[k] - TARGET))
        print(f"  {kind:6} " + " · ".join(f"k {k}: {v:.1f}" for k, v in res.items()) + f"  → 목표 {TARGET}에 가장 가까운 k = {best}")


if __name__ == "__main__":
    main()
