"""자동 Road Mask: 격자 특징 → 시드 → Region Growing → 정리 → 원본 크기로 복원."""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .grid_features import compute_feature_maps, compute_grid_features
from .image_io import to_binary_mask, validate_image
from .mask_refine import refine_mask
from .region_growing import grow_regions
from .seeds import score_cells, select_seeds

METHOD = "grid_lab_texture_seeds+region_growing_v1"


@dataclass
class RoadMaskResult:
    mask: np.ndarray          # 원본 크기 uint8 0/255
    seeds: list               # 시드 기록 (원본 좌표)
    work_scale: float         # 작업 영상 / 원본
    work_size: tuple          # (width, height)
    grid_shape: tuple         # (rows, cols)
    warnings: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


def work_image(image, long_side):
    """긴 변을 long_side로 줄인다 (작은 영상은 그대로)."""
    height, width = image.shape[:2]
    scale = 1.0 if long_side is None else min(1.0, long_side / max(height, width))
    if scale >= 1.0:
        return image, 1.0
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def upsample_mask(mask_work, size, threshold):
    if mask_work.shape[::-1] == tuple(size):
        return mask_work.copy()
    up = cv2.resize(mask_work.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    return up >= threshold


def extract_road_mask(image, cfg_roi):
    validate_image(image)
    height, width = image.shape[:2]
    lightness_weight = cfg_roi["features"]["lightness_weight"]
    timings = {}

    t0 = time.perf_counter()
    work, scale = work_image(image, cfg_roi["work_long_side"])
    maps = compute_feature_maps(work, cfg_roi["features"])
    grid = compute_grid_features(maps, cfg_roi["grid"], cfg_roi["features"])
    timings["features"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scores, components = score_cells(grid, cfg_roi["seeds"], lightness_weight)
    seeds, warnings = select_seeds(grid, scores, components, cfg_roi["seeds"], cfg_roi["grid"])
    timings["seeds"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    grown, growth = grow_regions(maps, grid, seeds, cfg_roi["region_growing"], lightness_weight)
    timings["region_growing"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    refined = refine_mask(grown, cfg_roi["refine"])
    full = upsample_mask(refined, (width, height), cfg_roi["upsample_threshold"])
    timings["refine"] = (time.perf_counter() - t0) * 1000

    seed_records = []
    for seed, record in zip(seeds, growth):
        seed_records.append({
            "row": seed.row, "col": seed.col,
            "x": round(seed.x / scale, 2), "y": round(seed.y / scale, 2),
            "score": round(seed.score, 4), "components": seed.components,
            "grown": record["grown"], "grown_pixels_work": record["pixels"],
            "cell_coverage": record["cell_coverage"], "color_threshold": record["color_threshold"],
        })
    if seeds and not any(r["grown"] for r in growth):
        warnings.append("no_seed_grew")

    return RoadMaskResult(mask=to_binary_mask(full), seeds=seed_records, work_scale=scale,
                          work_size=(work.shape[1], work.shape[0]), grid_shape=(grid.rows, grid.cols),
                          warnings=warnings, timings_ms={k: round(v, 2) for k, v in timings.items()})
