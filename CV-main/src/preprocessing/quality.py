"""도로 영역 안 영상 품질 지표 (밝기, 어두운/밝은 화소, 포화, 선명도, 노이즈)."""
import math

import cv2
import numpy as np

METRIC_NAMES = ("gray_mean", "gray_std", "block_mean_std_4x4", "dark_ratio", "bright_ratio",
                "saturation_ratio", "laplacian_variance", "noise_sigma")
# Immerkær(1996) 노이즈 추정 커널. 노면 질감 · 균열도 응답에 들어가므로 값이 크다고 꼭 잡음은 아니다.
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
NOISE_FACTOR = math.sqrt(math.pi / 2)
GRAYSCALE = "cv2.COLOR_BGR2GRAY (0.299R+0.587G+0.114B), uint8 0-255"


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _empty(include_hsv):
    values = {name: None for name in METRIC_NAMES}
    if include_hsv:
        values["hsv_v_mean"] = None
    return values


def _measure_full(img, gray, dark_threshold, bright_threshold, block_grid, include_hsv):
    height, width = gray.shape
    block_std = None
    if height >= block_grid and width >= block_grid:
        blocks = []
        for rows in np.array_split(gray, block_grid, axis=0):
            for block in np.array_split(rows, block_grid, axis=1):
                blocks.append(float(block.mean()))
        block_std = float(np.std(blocks, ddof=0))

    noise = None
    if height >= 3 and width >= 3:
        response = cv2.filter2D(gray, cv2.CV_16S, NOISE_KERNEL)
        total = np.abs(response[1:-1, 1:-1]).sum(dtype=np.int64)
        noise = float(NOISE_FACTOR * total / (6 * (width - 2) * (height - 2)))

    laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=1, borderType=cv2.BORDER_REFLECT_101)
    values = {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std(ddof=0)),
        "block_mean_std_4x4": block_std,
        "dark_ratio": float(np.mean(gray < dark_threshold)),
        "bright_ratio": float(np.mean(gray > bright_threshold)),
        "saturation_ratio": float(np.mean((gray == 0) | (gray == 255))),
        "laplacian_variance": float(laplacian.var()),
        "noise_sigma": noise,
    }
    if include_hsv:
        values["hsv_v_mean"] = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
    return values


def _block_std(gray, valid, block_grid, min_block_pixels):
    """마스크 외접 영역을 block_grid×block_grid로 나눈 블록 평균들의 표준편차."""
    ys, xs = np.nonzero(valid)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    gray_box, valid_box = gray[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    if gray_box.shape[0] < block_grid or gray_box.shape[1] < block_grid:
        return None
    means = []
    for rows, rows_valid in zip(np.array_split(gray_box, block_grid, axis=0),
                                np.array_split(valid_box, block_grid, axis=0)):
        for block, block_valid in zip(np.array_split(rows, block_grid, axis=1),
                                      np.array_split(rows_valid, block_grid, axis=1)):
            if np.count_nonzero(block_valid) >= min_block_pixels:
                means.append(float(block[block_valid].mean()))
    return float(np.std(means, ddof=0)) if len(means) >= 2 else None


def measure_quality(img, mask=None, *, support_mask=None, dark_threshold=40, bright_threshold=215,
                    block_grid=4, min_block_pixels=1, include_hsv=False):
    """mask가 없으면 영상 전체, 있으면 mask 안 화소만 잰다."""
    gray = to_gray(img)
    if mask is None:
        return _measure_full(img, gray, dark_threshold, bright_threshold, block_grid, include_hsv)
    valid = np.asarray(mask) > 0
    if valid.shape != gray.shape:
        raise ValueError("품질 측정 마스크 크기가 영상과 다름")
    if not valid.any():
        return _empty(include_hsv)

    pixels = gray[valid]
    values = {
        "gray_mean": float(pixels.mean()),
        "gray_std": float(pixels.std(ddof=0)),
        "block_mean_std_4x4": _block_std(gray, valid, block_grid, min_block_pixels),
        "dark_ratio": float(np.mean(pixels < dark_threshold)),
        "bright_ratio": float(np.mean(pixels > bright_threshold)),
        "saturation_ratio": float(np.mean((pixels == 0) | (pixels == 255))),
        "laplacian_variance": None,
        "noise_sigma": None,
    }
    # 3×3 필터 값은 창 전체가 도로 안인 화소에서만 집계
    support = valid if support_mask is None else valid & (np.asarray(support_mask) > 0)
    if support.any():
        laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=1, borderType=cv2.BORDER_REFLECT_101)
        values["laplacian_variance"] = float(laplacian[support].var())
    height, width = gray.shape
    if height >= 3 and width >= 3:
        inner = support[1:-1, 1:-1]
        if inner.any():
            response = cv2.filter2D(gray, cv2.CV_16S, NOISE_KERNEL)
            total = np.abs(response[1:-1, 1:-1][inner]).sum(dtype=np.int64)
            values["noise_sigma"] = float(NOISE_FACTOR * total / (6 * np.count_nonzero(inner)))
    if include_hsv:
        values["hsv_v_mean"] = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2][valid].mean())
    return values


def measurement_scale(shape, long_side):
    if long_side is None:
        return 1.0
    return long_side / max(shape[:2])


def measure_in_masks(img, analysis_mask, road_mask, cfg_quality):
    """긴 변 measurement_long_side로 맞춘 사본에서 analysis_mask 안을 잰다. → (지표, 측정 정보)"""
    scale = measurement_scale(img.shape, cfg_quality["measurement_long_side"])
    height, width = img.shape[:2]
    method = "none"
    if abs(scale - 1.0) > 1e-12:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        method = "INTER_AREA" if scale < 1 else "INTER_CUBIC"
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        analysis_mask = cv2.resize(analysis_mask, size, interpolation=cv2.INTER_NEAREST)
        road_mask = cv2.resize(road_mask, size, interpolation=cv2.INTER_NEAREST)

    support = cv2.erode((road_mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    values = measure_quality(img, analysis_mask, support_mask=support,
                             dark_threshold=cfg_quality["dark_threshold"],
                             bright_threshold=cfg_quality["bright_threshold"],
                             block_grid=cfg_quality["block_grid"],
                             min_block_pixels=cfg_quality["min_block_pixels"])
    info = {"measurement_long_side": cfg_quality["measurement_long_side"], "scale": round(scale, 6),
            "measured_size": [img.shape[1], img.shape[0]], "resize": method, "grayscale": GRAYSCALE,
            "intensity_scale": "0-255", "dark_threshold": cfg_quality["dark_threshold"],
            "bright_threshold": cfg_quality["bright_threshold"]}
    return values, info
