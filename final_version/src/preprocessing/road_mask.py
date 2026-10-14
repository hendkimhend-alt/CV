"""자동 노면 마스크: 격자 특징, 시드, Region Growing, 정리를 거쳐 원본 크기로 되돌린다."""
import cv2
import numpy as np

from .grid_features import compute_feature_maps, compute_grid_features
from .mask_refine import refine_mask
from .region_growing import grow_regions
from .seeds import score_cells, select_seeds


def work_image(image, long_side):
    """긴 변이 long_side보다 크면 줄인 작업 영상과 배율을 돌려준다."""
    height, width = image.shape[:2]
    scale = min(1.0, long_side / max(height, width))
    if scale >= 1.0:
        return image, 1.0
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def upsample_mask(mask_work, size, threshold):
    """작업 영상의 마스크를 원본 크기로 키운다 (선형 보간 후 threshold 이상)."""
    if mask_work.shape[::-1] == tuple(size):
        return mask_work.copy()
    up = cv2.resize(mask_work.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    return up >= threshold


def extract_road_mask(image, cfg):
    """노면 마스크(uint8 0/255, 원본 크기)와 시드 위치(원본 좌표)를 돌려준다."""
    height, width = image.shape[:2]
    lightness_weight = cfg["features"]["lightness_weight"]

    work, scale = work_image(image, cfg["work_long_side"])
    maps = compute_feature_maps(work, cfg["features"])
    grid = compute_grid_features(maps, cfg["grid"], cfg["features"])
    scores = score_cells(grid, cfg["seeds"], lightness_weight)
    seeds = select_seeds(grid, scores, cfg["seeds"], cfg["grid"])
    grown = grow_regions(maps, grid, seeds, cfg["region_growing"], lightness_weight)
    refined = refine_mask(grown, cfg["refine"])
    full = upsample_mask(refined, (width, height), cfg["upsample_threshold"])
    if not seeds:
        # 시드가 하나도 없으면(근접 촬영 등) 화면 전체를 노면으로 놓고 계속 간다. 검증은 FAIL로 나온다
        full = np.ones((height, width), bool)

    seed_points = [(round(s.x / scale, 2), round(s.y / scale, 2)) for s in seeds]
    return np.where(full, 255, 0).astype(np.uint8), seed_points
