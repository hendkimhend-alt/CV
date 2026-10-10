"""테스트 공통 도구 — src 경로 등록, 기본 설정, 합성 도로 장면."""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing.config import load_config  # noqa: E402

RDD_IMG = ROOT / "data" / "RDD2020_train" / "train" / "img"
_BASE = load_config()


def config(**overrides):
    """기본 설정 사본. overrides: {"gamma.mode": "off", ...}."""
    cfg = copy.deepcopy(_BASE)
    for key, value in overrides.items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg


def synthetic_scene(seed=0, height=240, width=320, road_gray=95, crack=True):
    """위 35% 하늘(매끈한 하늘색) · 아래 왼쪽 15% 풀(초록, 거친 질감) · 나머지 아스팔트(회색 + 질감).

    → (BGR 영상, 영역 사전 {"sky", "grass", "road"} bool 마스크)
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), np.float64)
    sky_rows = int(height * 0.35)
    grass_cols = int(width * 0.15)
    regions = {name: np.zeros((height, width), bool) for name in ("sky", "grass", "road")}
    regions["sky"][:sky_rows] = True
    regions["grass"][sky_rows:, :grass_cols] = True
    regions["road"][sky_rows:, grass_cols:] = True
    img[regions["sky"]] = (235, 205, 160)
    img[regions["sky"]] += rng.normal(0, 0.8, (regions["sky"].sum(), 1))
    img[regions["grass"]] = (50, 140, 60)
    img[regions["grass"]] += rng.normal(0, 30, (regions["grass"].sum(), 3))
    img[regions["road"]] = road_gray
    img[regions["road"]] += rng.normal(0, 6, (regions["road"].sum(), 1))
    if crack:  # 노면 안쪽(경계에 닿지 않음)의 어두운 가는 균열
        y0, y1 = sky_rows + 40, height - 30
        for y in range(y0, y1):
            x = grass_cols + 100 + int(10 * np.sin(y / 9.0))
            img[y, x:x + 2] = road_gray - 45
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), regions
