"""자동 Road Mask 검증: 면적 · 최대 연결 요소 · 구멍 · 시드 일치 4개 지표로 PASS / FAIL."""
import cv2
import numpy as np

from .mask_refine import fill_holes

PASS, FAIL = "PASS", "FAIL"


def compute_mask_metrics(mask, seed_points):
    road = np.asarray(mask) > 0
    area = int(road.sum())
    metrics = {"mask_area_ratio": area / road.size if road.size else None,
               "largest_component_ratio": None, "hole_ratio": None, "seed_consistency": None,
               "n_components": 0, "n_seeds": len(seed_points), "road_pixels": area}
    if area:
        count, _, stats, _ = cv2.connectedComponentsWithStats(road.astype(np.uint8), connectivity=8)
        metrics["n_components"] = count - 1
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


def validate_road_mask(metrics, cfg_validation):
    checks, reasons, warnings = [], [], []
    if metrics.get("n_seeds", 0) == 0:
        reasons.append("no_seed")
    if not metrics.get("road_pixels"):
        reasons.append("empty_mask")

    policy = cfg_validation["undefined_metric_policy"]
    for name, rule in cfg_validation["metrics"].items():
        value = metrics.get(name)
        check = {"metric": name, "enabled": rule["enabled"], "value": value,
                 "min": rule.get("min"), "max": rule.get("max"), "passed": None}
        checks.append(check)
        if not rule["enabled"]:
            continue
        if value is None:
            check["passed"] = policy != "fail"
            if policy == "fail":
                reasons.append(f"{name}:undefined")
            else:
                warnings.append(f"{name}:undefined")
            continue
        failed = []
        if rule.get("min") is not None and value < rule["min"]:
            failed.append(f"{name}<{rule['min']}")
        if rule.get("max") is not None and value > rule["max"]:
            failed.append(f"{name}>{rule['max']}")
        check["passed"] = not failed
        reasons.extend(f"{item} ({value:.4f})" for item in failed)

    status = FAIL if reasons else PASS
    return {"status": status, "checks": checks, "fail_reasons": reasons, "warnings": warnings}
