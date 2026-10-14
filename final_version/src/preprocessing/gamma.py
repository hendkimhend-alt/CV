"""노면이 어두울 때만 노면 안에 Gamma 보정을 건다. out = 255 * (in / 255) ^ gamma

밝기 평균이 dark_mean_below보다 낮고 어두운 화소 비율이 dark_ratio_above보다 높으면 어둡다고 본다.
후보 gamma를 약한 것부터 써 보고 어두움이 풀리는 첫 값을 고른다. 포화가 심해지면 그 전 값에서 멈춘다.
"""
from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=32)
def gamma_lut(gamma):
    lut = np.rint(255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** gamma).astype(np.uint8)
    lut.setflags(write=False)
    return lut


def apply_gamma(img, gamma, mask):
    corrected = cv2.LUT(img, gamma_lut(float(gamma)))
    out = img.copy()
    road = mask > 0
    out[road] = corrected[road]
    return out


def analysis_mask(road_mask, cfg):
    # 노면 가장자리는 빼고 잰다. 남는 게 너무 적으면 Gamma 판단을 안 한다
    k = cfg["erode_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    analysis = cv2.erode(road_mask.copy(), kernel, iterations=1)
    analysis = np.where(analysis > 0, 255, 0).astype(np.uint8)
    pixels = int(np.count_nonzero(analysis))
    road_pixels = int(np.count_nonzero(road_mask))
    ratio = pixels / road_pixels if road_pixels else 0.0
    if pixels < cfg["min_pixels"] or ratio < cfg["min_ratio_of_road"]:
        return None
    return analysis


def measure(img, analysis, cfg):
    scale = cfg["measure_long_side"] / max(img.shape[:2])
    if abs(scale - 1.0) > 1e-12:
        size = (max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        analysis = cv2.resize(analysis, size, interpolation=cv2.INTER_NEAREST)
    valid = analysis > 0
    if not valid.any():
        return None
    pixels = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[valid]
    return {"gray_mean": float(pixels.mean()),
            "dark_ratio": float(np.mean(pixels < cfg["dark_pixel_value"])),
            "saturation_ratio": float(np.mean((pixels == 0) | (pixels == 255)))}


def is_dark(m, cfg):
    return m["gray_mean"] < cfg["dark_mean_below"] and m["dark_ratio"] > cfg["dark_ratio_above"]


def is_resolved(m, cfg):
    # not is_dark가 아님. 두 조건이 둘 다 풀려야 멈춘다
    return m["gray_mean"] >= cfg["dark_mean_below"] and m["dark_ratio"] <= cfg["dark_ratio_above"]


def guard_triggered(before, after, cfg):
    sat = after["saturation_ratio"]
    return sat > cfg["max_saturation_after"] and sat > before["saturation_ratio"]


def correct_gamma(image, road_mask, cfg):
    analysis = analysis_mask(road_mask, cfg)
    if analysis is None:
        return image
    before = measure(image, analysis, cfg)
    if before is None or not is_dark(before, cfg):
        return image

    best = image
    for g in sorted({float(c) for c in cfg["candidates"]}, reverse=True):
        img = apply_gamma(image, g, road_mask)
        after = measure(img, analysis, cfg)
        if guard_triggered(before, after, cfg):    # 더 작은 gamma는 더 밝으니 볼 필요 없음
            break
        best = img
        if is_resolved(after, cfg):
            break
    return best
