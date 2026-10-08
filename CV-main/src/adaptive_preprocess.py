"""개발 세트에서 동결한 ROI/MSR/Gamma/Gaussian 전처리. GT와 검출기는 사용하지 않는다."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .preprocess import (apply_gamma, apply_gaussian, apply_msr, geometry_preprocess,
                             validate_config, validate_image, validate_msr_config, FINAL_CONFIG_PATH)
    from .roi import apply_roi_mask, geometry_mask, validate_mask
    from .metrics import measure_quality
else:
    from preprocess import (apply_gamma, apply_gaussian, apply_msr, geometry_preprocess,
                            validate_config, validate_image, validate_msr_config, FINAL_CONFIG_PATH)
    from roi import apply_roi_mask, geometry_mask, validate_mask
    from metrics import measure_quality

CONFIG_PATH = FINAL_CONFIG_PATH


def load_adaptive_config(path=None):
    """outputs 삭제 여부와 무관하게 소스 옆의 동결 설정을 읽는다."""
    return validate_adaptive_config(json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8")))


def validate_adaptive_config(config):
    cfg = copy.deepcopy(config)
    if not isinstance(cfg, dict) or cfg.get("version") != 1:
        raise ValueError("적응형 설정 version=1이 필요합니다")
    cfg["geometry"] = validate_config(cfg["geometry"])
    if not isinstance(cfg["geometry"]["roi"], dict) or cfg["geometry"]["roi_overrides"]:
        raise ValueError("고정 사다리꼴 ROI만 허용합니다")
    cfg["msr"]["parameters"] = validate_msr_config(cfg["msr"]["parameters"])
    alpha = cfg["msr"]["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("MSR alpha는 0~1이어야 합니다")
    for stage, labels in (("gamma", ("low", "high", "dark_value", "bright_value")),
                          ("gaussian", ("low", "high", "moderate_sigma", "high_sigma"))):
        rule = cfg[stage]
        if not isinstance(rule["enabled"], bool):
            raise ValueError("단계 enabled는 bool이어야 합니다")
        for key in labels:
            value = rule[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{stage}.{key}: 유한한 음이 아닌 수가 필요합니다")
        if rule["low"] > rule["high"]:
            raise ValueError("규칙 구간 순서 오류")
    if cfg["gamma"]["dark_value"] <= 0 or cfg["gamma"]["bright_value"] <= 0:
        raise ValueError("Gamma는 양수이어야 합니다")
    if not isinstance(cfg["msr"]["enabled"], bool) or cfg["gaussian"].get("kernel") != 3:
        raise ValueError("MSR enabled는 bool, Gaussian kernel은 3이어야 합니다")
    if not 0 < cfg["gamma"]["dark_value"] <= 1 <= cfg["gamma"]["bright_value"]:
        raise ValueError("어두운 영상 Gamma <=1, 밝은 영상 Gamma >=1이 필요합니다")
    if max(cfg["gaussian"]["moderate_sigma"], cfg["gaussian"]["high_sigma"]) > 5:
        raise ValueError("Gaussian sigma 상한은 5입니다")
    return cfg


def blend_msr_lightness(reference, msr_image, alpha, mask):
    """기존 MSR 강도 실험과 동일한 L 혼합. alpha=0/1은 색 공간 왕복을 생략한다."""
    validate_image(reference)
    validate_image(msr_image)
    mask = validate_mask(mask, reference.shape)
    if reference.shape != msr_image.shape:
        raise ValueError("원본/MSR 크기 불일치")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha는 유한한 0~1 숫자")
    if alpha == 0:
        return apply_roi_mask(reference.copy(), mask)
    if alpha == 1:
        return apply_roi_mask(msr_image.copy(), mask)
    lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    original_l = lab[:, :, 0].astype(np.float32)
    msr_l = cv2.cvtColor(msr_image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    lab[:, :, 0] = np.rint((1 - alpha) * original_l + alpha * msr_l).astype(np.uint8)
    return apply_roi_mask(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), mask)


def masked_gaussian(image, mask, sigma):
    """G(I*M)/G(M). ROI 밖 검정 패딩이 유효 도로로 번지지 않도록 한다."""
    validate_image(image)
    mask = validate_mask(mask, image.shape)
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or not 0 <= sigma <= 5:
        raise ValueError("sigma는 유한한 0~5 숫자")
    if sigma == 0:
        return apply_roi_mask(image.copy(), mask)
    settings = {"kernel": 3, "sigma": float(sigma)}
    valid = (mask != 0).astype(np.float32)
    numerator = apply_gaussian(image.astype(np.float32) * valid[:, :, None], settings)
    denominator = apply_gaussian(valid, settings)
    result = numerator / np.maximum(denominator[:, :, None], 1e-8)
    return apply_roi_mask(np.rint(np.clip(result, 0, 255)).astype(np.uint8), mask)


def gamma_choice(quality, rule):
    value = quality.get("gray_mean")
    if not rule["enabled"] or value is None or not math.isfinite(value):
        return 1.0, "disabled_or_unavailable"
    if value < rule["low"]:
        return rule["dark_value"], "dark"
    if value > rule["high"]:
        return rule["bright_value"], "bright"
    return 1.0, "inside_target"


def gaussian_choice(quality, rule):
    value = quality.get("noise_sigma")
    if not rule["enabled"] or value is None or not math.isfinite(value):
        return 0.0, "disabled_or_unavailable"
    if value > rule["high"]:
        return rule["high_sigma"], "high_noise"
    if value > rule["low"]:
        return rule["moderate_sigma"], "moderate_noise"
    return 0.0, "inside_target"


def adaptive_from_reference(reference, geometry, config, stage="B3"):
    """동일 ROI 출력에서 B0/B1/B2/B3를 만든다. config는 이미 검증한 설정이다."""
    if stage not in ("B0", "B1", "B2", "B3", "FINAL"):
        raise ValueError("적응형 단계 오류")
    mask = geometry_mask(geometry)
    applied_msr = stage != "B0" and config["msr"]["enabled"] and config["msr"]["alpha"] > 0
    fixed = reference.copy()
    start = time.perf_counter()
    if applied_msr:
        msr = apply_msr(reference, config["msr"]["parameters"], mask)
        fixed = blend_msr_lightness(reference, msr, config["msr"]["alpha"], mask)
    msr_ms = (time.perf_counter() - start) * 1000 if applied_msr else 0.0
    start = time.perf_counter()
    before = measure_quality(fixed, mask)
    quality_before_ms = (time.perf_counter() - start) * 1000
    gamma, gamma_reason = gamma_choice(before, config["gamma"]) if stage in ("B2", "B3", "FINAL") else (1.0, "stage_disabled")
    start = time.perf_counter()
    if gamma != 1:
        fixed = apply_roi_mask(apply_gamma(fixed, gamma), mask)
    gamma_ms = (time.perf_counter() - start) * 1000 if gamma != 1 else 0.0
    start = time.perf_counter()
    after_gamma = measure_quality(fixed, mask) if gamma != 1 else before
    quality_after_gamma_ms = (time.perf_counter() - start) * 1000 if gamma != 1 else 0.0
    sigma, gaussian_reason = gaussian_choice(after_gamma, config["gaussian"]) if stage in ("B3", "FINAL") else (0.0, "stage_disabled")
    start = time.perf_counter()
    if sigma:
        fixed = masked_gaussian(fixed, mask, sigma)
    gaussian_ms = (time.perf_counter() - start) * 1000 if sigma else 0.0
    start = time.perf_counter()
    after = measure_quality(fixed, mask) if sigma else after_gamma
    quality_after_ms = (time.perf_counter() - start) * 1000 if sigma else 0.0
    return fixed, {"stage": stage, "msr_applied": applied_msr, "msr_parameters": config["msr"]["parameters"],
                   "msr_alpha": config["msr"]["alpha"] if applied_msr else 0.0,
                   "gamma_applied": gamma != 1, "gamma": gamma, "gamma_reason": gamma_reason,
                   "gaussian_applied": sigma > 0, "gaussian_sigma": sigma, "gaussian_reason": gaussian_reason,
                   "quality_before": before, "quality_after_gamma": after_gamma, "quality_after": after,
                   "timings_ms": {"msr_ms": msr_ms, "gamma_ms": gamma_ms, "gaussian_ms": gaussian_ms,
                                  "quality_before_ms": quality_before_ms, "quality_after_gamma_ms": quality_after_gamma_ms,
                                  "quality_after_ms": quality_after_ms}}


def adaptive_preprocess(image, config=None, stage="FINAL"):
    """(보정 BGR, 메타데이터)를 반환한다. GT 입력과 데이터셋 접근이 없다."""
    cfg = load_adaptive_config() if config is None else validate_adaptive_config(config)
    start = time.perf_counter()
    reference, geometry = geometry_preprocess(image, cfg["geometry"])
    geometry_ms = (time.perf_counter() - start) * 1000
    fixed, metadata = adaptive_from_reference(reference, geometry, cfg, stage)
    metadata.update(geometry=geometry, roi_mask=geometry_mask(geometry))
    metadata["timings_ms"]["geometry_ms"] = geometry_ms
    metadata["preprocess_ms"] = sum(metadata["timings_ms"].values())
    return fixed, metadata
