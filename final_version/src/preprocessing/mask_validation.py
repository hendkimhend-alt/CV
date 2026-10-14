"""자동 노면 마스크 검증. 면적, 최대 연결 요소, 구멍, 시드 일치 네 지표로 PASS/FAIL을 정한다."""
import cv2
import numpy as np

from .mask_refine import fill_holes


def mask_metrics(mask, seed_points):
    """검증 지표 네 가지: 면적 비율, 최대 연결 요소 비율, 구멍 비율, 시드 일치율."""
    road = mask > 0
    area = int(road.sum())
    metrics = {"mask_area_ratio": area / road.size, "largest_component_ratio": None,
               "hole_ratio": None, "seed_consistency": None}
    if area:
        _, _, stats, _ = cv2.connectedComponentsWithStats(road.astype(np.uint8), connectivity=8)
        metrics["largest_component_ratio"] = float(stats[1:, cv2.CC_STAT_AREA].max()) / area
        filled = fill_holes(road)
        metrics["hole_ratio"] = float(filled.sum() - area) / float(filled.sum())
    if seed_points:
        height, width = road.shape
        inside = 0
        for x, y in seed_points:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < width and 0 <= yi < height and road[yi, xi]:
                inside += 1
        metrics["seed_consistency"] = inside / len(seed_points)
    return metrics


def validate_road_mask(mask, seed_points, rules):
    """지표가 기준을 모두 넘으면 PASS. (통과 여부, 실패 이유 목록)을 돌려준다."""
    metrics = mask_metrics(mask, seed_points)
    reasons = []
    if not seed_points:
        reasons.append("no_seed")
    if not np.any(mask):
        reasons.append("empty_mask")
    for name, rule in rules.items():
        value = metrics[name]
        if value is None:
            reasons.append(f"{name}:undefined")
        elif "min" in rule and value < rule["min"]:
            reasons.append(f"{name}<{rule['min']} ({value:.4f})")
        elif "max" in rule and value > rule["max"]:
            reasons.append(f"{name}>{rule['max']} ({value:.4f})")
    return not reasons, reasons
