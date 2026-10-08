"""공통 보정 연산과 FINAL 전처리. 기존 P0/P1/P1+ 실행 코드는 주석으로 보존한다."""

from __future__ import annotations

import copy
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .roi import apply_roi_mask, build_trapezoid_mask, geometry_mask, validate_trapezoid
else:
    from roi import apply_roi_mask, build_trapezoid_mask, geometry_mask, validate_trapezoid

# 기존 비교 조건은 비활성화한다.
# CONDITIONS = ("P0_reference", "P1", "P1+")
CONDITIONS = ("FINAL",)
VALID_TAGS = {"blur", "local_illumination", "normal", "unclassified"}
# 기능: 보정 전 ROI·Resize 영상의 품질값으로 그룹을 자동 결정한다.
# 특징: 경계값 50·44는 포함하지 않는다. 필요하면 아래 숫자만 수정한다.
AUTO_GROUP_THRESHOLDS = {"blur_laplacian_lt": 50.0, "local_block_std_gt": 44.0}
INTERPOLATIONS = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC,
                  "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST,
                  "lanczos": cv2.INTER_LANCZOS4}
# 이전 P0/P1/P1+ 설정 — 실행하지 않는 참고 코드.
# DEFAULT_CONFIG = {
#     # road_focus: 사용자가 적용을 요청한 고정 좌표. ROI 수정은 이 부분에서만 한다.
#     "roi": {"type": "trapezoid", "top_y_ratio": 0.60,
#             "top_left_x_ratio": 0.075, "top_right_x_ratio": 0.725,
#             "bottom_left_x_ratio": 0.05, "bottom_right_x_ratio": 0.95,
#             "bottom_y_ratio": 1.00},
#     "roi_overrides": {},
#     "resize": {"long_side": 1024, "allow_upscale": True,
#                "down_interpolation": "area", "up_interpolation": "cubic"},
#     "gamma": {"value": 0.8, "channel": "BGR"},
#     "gaussian": {"kernel": 3, "sigma": 0.8},
#     "clahe": {"clip_limit": 2.0, "tile_grid": [8, 8], "channel": "LAB_L"},
#     "unsharp": {"amount": 0.5, "sigma": 1.0},
# }

# 활성 ROI 좌표와 크기 설정은 FINAL JSON 한 곳에서만 정의한다.
FINAL_CONFIG_PATH = Path(__file__).with_name("final_preprocessing_config.json")
DEFAULT_CONFIG = json.loads(FINAL_CONFIG_PATH.read_text(encoding="utf-8"))["geometry"]

GEOMETRY_COLUMNS = [
    "original_width", "original_height", "crop_x", "crop_y", "crop_width", "crop_height",
    "output_width", "output_height", "scale_x", "scale_y", "roi_json", "allow_upscale",
    "resize_interpolation",
    "roi_polygon_json", "roi_area_ratio",
]
PARAMETER_COLUMNS = [
    "gamma", "gamma_channel", "gaussian_kernel", "gaussian_sigma", "clahe_clip_limit",
    "clahe_tile_width", "clahe_tile_height", "clahe_channel", "unsharp_applied",
    "unsharp_amount", "unsharp_sigma",
]


# 기능: 이미지 처리에 필요한 양수·정수 설정만 간단히 확인한다.
# 특징: bool·NaN·무한대·범위 초과를 거부하며, 0을 허용할 항목은 하한을 별도로 지정한다.
def check_number(value, label, minimum=0, integer=False, strict=False):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and (value > minimum if strict else value >= minimum)
                 and (not integer or isinstance(value, int)))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"error:{label} 값 또는 범위 오류")


# 기능: ROI 형식과 좌표·크기를 검사한다.
# 특징: 이미지 경계 검사는 실제 해상도를 아는 geometry_preprocess에서 수행한다.
def check_roi(roi):
    if isinstance(roi, str) and roi in {"bottom_half", "full"}:
        return
    if isinstance(roi, dict):
        validate_trapezoid(roi)
        return
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("error:ROI는 사다리꼴 사전, bottom_half/full 또는 [x,y,width,height]")
    for index, value in enumerate(roi):
        check_number(value, "roi", 0 if index < 2 else 1, integer=True)


# 기능: 일부 사용자 설정을 기본값에 합치고 OpenCV 처리에 필요한 값만 검증한다.
# 특징: 입력 사전을 복사하며 설정 파일·플러그인·범용 검증 프레임워크를 사용하지 않는다.
def validate_config(updates=None):
    if updates is not None and not isinstance(updates, dict):
        raise ValueError("error:보정 설정은 사전이어야 함")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (updates or {}).items():
        if key not in cfg:
            raise ValueError(f"error:알 수 없는 보정 설정 {key}")
        if isinstance(cfg[key], dict) and key not in {"roi", "roi_overrides"}:
            if not isinstance(value, dict) or set(value) - set(cfg[key]):
                raise ValueError(f"error:{key} 설정 형식 또는 항목 오류")
            cfg[key].update(copy.deepcopy(value))
        else:
            cfg[key] = copy.deepcopy(value)
    check_roi(cfg["roi"])
    if not isinstance(cfg["roi_overrides"], dict):
        raise ValueError("error:roi_overrides는 image_id별 ROI 사전이어야 함")
    for image_id, roi in cfg["roi_overrides"].items():
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("error:ROI 설정의 image_id 누락")
        check_roi(roi)
    resize = cfg["resize"]
    check_number(resize["long_side"], "resize.long_side", 1, integer=True)
    if not isinstance(resize["allow_upscale"], bool):
        raise ValueError("error:allow_upscale는 True/False")
    for key in ("down_interpolation", "up_interpolation"):
        if not isinstance(resize[key], str) or resize[key] not in INTERPOLATIONS:
            raise ValueError(f"error:{key} 보간법 오류")
    for section, key, strict in [("gamma", "value", True), ("gaussian", "sigma", False),
                                 ("clahe", "clip_limit", True), ("unsharp", "amount", False),
                                 ("unsharp", "sigma", True)]:
        check_number(cfg[section][key], f"{section}.{key}", strict=strict)
    kernel = cfg["gaussian"]["kernel"]
    check_number(kernel, "gaussian.kernel", 1, integer=True)
    if kernel % 2 == 0:
        raise ValueError("error:Gaussian 커널은 양의 홀수")
    tiles = cfg["clahe"]["tile_grid"]
    if not isinstance(tiles, list) or len(tiles) != 2:
        raise ValueError("error:tile_grid는 [가로 타일 수,세로 타일 수]")
    for value in tiles:
        check_number(value, "clahe.tile_grid", 1, integer=True)
    if cfg["gamma"]["channel"] != "BGR" or cfg["clahe"]["channel"] != "LAB_L":
        raise ValueError("error:Gamma는 BGR, CLAHE는 LAB_L 채널")
    return cfg


# 기능: 품질 태그 문자열을 정렬된 튜플로 바꾼다.
# 특징: 흐림·국소조도 중첩은 허용한다. unclassified는 일부 지표를 측정할 수 없다는 뜻이다.
def parse_tags(text, allow_empty=False):
    if not isinstance(text, str):
        raise ValueError("error:tags는 세미콜론으로 구분한 문자열")
    tags = tuple(part.strip() for part in text.split(";") if part.strip())
    if ((not tags and not allow_empty) or len(set(tags)) != len(tags)
            or set(tags) - VALID_TAGS or ("normal" in tags and len(tags) != 1)):
        raise ValueError(f"error:품질 태그 오류 {text}")
    return tuple(sorted(tags))


# 기능: 사용자가 수정한 자동 분류 임계값을 실행 전에 검사하고 복사한다.
# 특징: 음수·NaN·무한대·잘못된 항목을 거부하여 설정 기록과 실제 분류 기준을 일치시킨다.
def validate_thresholds(thresholds=None):
    settings = AUTO_GROUP_THRESHOLDS if thresholds is None else thresholds
    if not isinstance(settings, dict) or set(settings) != {"blur_laplacian_lt", "local_block_std_gt"}:
        raise ValueError("error:자동 분류 임계값 항목 오류")
    for key, value in settings.items():
        check_number(value, key)
    return dict(settings)


# 기능: 보정 전 품질 지표에서 흐림·국소조도·정상을 자동 분류한다.
# 특징: 중첩 그룹을 허용하며 누락·비정상 지표를 정상으로 판정하지 않는다.
def classify_quality(values, thresholds=None):
    if not isinstance(values, dict):
        raise ValueError("error:품질 지표는 사전이어야 함")
    limits = validate_thresholds(thresholds)
    measurements = []
    for key in ("laplacian_variance", "block_mean_std_4x4"):
        value = values.get(key)
        valid = (not isinstance(value, (bool, np.bool_))
                 and isinstance(value, (int, float, np.integer, np.floating)))
        try:
            valid = valid and math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        measurements.append(value if valid else None)
    laplacian, block_std = measurements
    tags = []
    if laplacian is not None and laplacian < limits["blur_laplacian_lt"]:
        tags.append("blur")
    if block_std is not None and block_std > limits["local_block_std_gt"]:
        tags.append("local_illumination")
    if laplacian is None or block_std is None:
        tags.append("unclassified")
    return tuple(sorted(tags or ["normal"]))


# 기능: 전처리·측정 입력이 비어 있지 않은 uint8 BGR 배열인지 검사한다.
# 특징: 자료형을 자동 변환하지 않고 잘못된 입력은 해당 이미지의 실패로 기록하게 한다.
def validate_image(img):
    if (not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3
            or img.shape[2] != 3 or min(img.shape[:2]) == 0):
        raise ValueError("error:입력은 비어 있지 않은 uint8 BGR(H×W×3) 이미지")


# 기능: ROI의 외접 직사각형을 자르고 긴 변을 정규화하며 유효 ROI 밖은 0으로 채운다.
# 특징: 기존 Resize 규칙을 유지한다. 사다리꼴은 워핑하지 않고 실제 꼭짓점·면적을 기록한다.
def geometry_preprocess(img, cfg, image_id=""):
    validate_image(img)
    original_h, original_w = img.shape[:2]
    roi = cfg["roi_overrides"].get(image_id, cfg["roi"])
    polygon = None
    if roi == "bottom_half":
        x, y, width, height = 0, original_h // 2, original_w, original_h - original_h // 2
    elif roi == "full":
        x, y, width, height = 0, 0, original_w, original_h
    elif isinstance(roi, dict):
        original_mask, polygon = build_trapezoid_mask(original_h, original_w, roi)
        x, y, width, height = cv2.boundingRect(polygon)
    else:
        x, y, width, height = roi
    if x + width > original_w or y + height > original_h:
        raise ValueError("error:ROI가 원본 이미지 경계를 벗어남")
    cropped = img[y:y + height, x:x + width]
    scale = cfg["resize"]["long_side"] / max(width, height)
    if not cfg["resize"]["allow_upscale"]:
        scale = min(scale, 1.0)
    output_w, output_h = max(1, round(width * scale)), max(1, round(height * scale))
    interpolation = "none"
    if (output_w, output_h) != (width, height):
        key = "down_interpolation" if scale < 1 else "up_interpolation"
        interpolation = cfg["resize"][key]
        cropped = cv2.resize(cropped, (output_w, output_h), interpolation=INTERPOLATIONS[interpolation])
    else:
        cropped = cropped.copy()
    geometry = dict(original_width=original_w, original_height=original_h,
                    crop_x=x, crop_y=y, crop_width=width, crop_height=height,
                    output_width=output_w, output_height=output_h,
                    scale_x=output_w / width, scale_y=output_h / height,
                    roi_json=json.dumps(roi), allow_upscale=cfg["resize"]["allow_upscale"],
                    resize_interpolation=interpolation,
                    roi_polygon_json=json.dumps(polygon.tolist()) if polygon is not None else "",
                    roi_area_ratio=float(np.count_nonzero(original_mask) / (original_h * original_w))
                    if polygon is not None else width * height / (original_h * original_w))
    return apply_roi_mask(cropped, geometry_mask(geometry)), geometry


# 기능: Gamma별 256개 밝기 보정표를 만든다.
# 특징: A1의 float64 공식과 반올림을 유지하며 최대 32개 읽기 전용 표만 재사용한다.
@lru_cache(maxsize=32)
def gamma_lut(gamma):
    lut = np.rint(255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** gamma).astype(np.uint8)
    lut.setflags(write=False)
    return lut


# 기능: y=255*(x/255)^gamma 공식을 BGR 각 채널에 적용한다.
# 특징: gamma<1은 중간 밝기를 높인다. 자동 Gamma 선정이나 원본 수정은 하지 않는다.
def apply_gamma(img, gamma):
    check_number(gamma, "gamma", strict=True)
    return cv2.LUT(img, gamma_lut(float(gamma)))


# 기능: Lab L 채널에만 CLAHE를 적용한다.
# 특징: A1과 같은 색공간 변환·타일 개수·대비 제한값을 사용한다.
def apply_clahe(img, cfg):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=cfg["clip_limit"], tileGridSize=tuple(cfg["tile_grid"]))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# 기능: 입력과 Gaussian 흐림 영상의 차이를 더해 윤곽을 강조한다.
# 특징: float32 계산·0~255 제한·반올림을 유지하며 적용할 그룹은 조건 함수에서 결정한다.
def apply_unsharp(img, cfg):
    source = img.astype(np.float32)
    blurred = cv2.GaussianBlur(source, (0, 0), cfg["sigma"], borderType=cv2.BORDER_REFLECT_101)
    return np.rint(np.clip(source + cfg["amount"] * (source - blurred), 0, 255)).astype(np.uint8)


# 기능: 보정 경로 마지막에 Gaussian 평활화를 적용한다.
# 특징: Unsharp 다음에 실행하며 커널·sigma·REFLECT_101 경계 처리는 A1과 같다.
def apply_gaussian(img, cfg):
    kernel = cfg["kernel"]
    return cv2.GaussianBlur(img, (kernel, kernel), cfg["sigma"], borderType=cv2.BORDER_REFLECT_101)


# 독립 MSR 실험 설정. DEFAULT_CONFIG / P0 / P1 / P1+에는 연결하지 않는다.
# scales는 ROI+Resize 출력에서의 Gaussian sigma(픽셀)이다. 최적값이 아니다.
DEFAULT_MSR_CONFIG = {
    "enabled": True, "scales": [15.0, 80.0, 250.0],
    "weights": [1 / 3, 1 / 3, 1 / 3], "epsilon": 1.0,
    "channel": "LAB_L", "output_percentiles": [1.0, 99.0],
    "min_log_range": 1e-4,
    "blur_mode": "pyramid", "max_blur_sigma": 12.0,
}


def validate_msr_config(updates=None):
    """MSR 전용 설정을 복사·검증한다. 가중치는 합이 1이 되도록 정규화한다."""
    if updates is not None and (not isinstance(updates, dict)
                               or set(updates) - set(DEFAULT_MSR_CONFIG)):
        raise ValueError("error:MSR 설정 항목 오류")
    cfg = copy.deepcopy(DEFAULT_MSR_CONFIG)
    cfg.update(copy.deepcopy(updates or {}))
    if not isinstance(cfg["enabled"], bool) or cfg["channel"] != "LAB_L":
        raise ValueError("error:MSR enabled는 bool, channel은 LAB_L")
    scales, weights = cfg["scales"], cfg["weights"]
    if (not isinstance(scales, list) or not scales or not isinstance(weights, list)
            or len(scales) != len(weights)):
        raise ValueError("error:MSR scales/weights는 같은 길이의 비어 있지 않은 목록")
    for sigma in scales:
        check_number(sigma, "msr.scale", strict=True)
        if sigma > 4096:
            raise ValueError("error:MSR sigma는 4096 이하")
    for weight in weights:
        check_number(weight, "msr.weight")
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("error:MSR 가중치 합은 유한한 양수")
    cfg["weights"] = [float(weight / total) for weight in weights]
    for key in ("epsilon", "min_log_range", "max_blur_sigma"):
        check_number(cfg[key], "msr." + key, strict=True)
    if not 0.5 <= cfg["max_blur_sigma"] <= 64 or cfg["epsilon"] < 1e-6:
        raise ValueError("error:MSR max_blur_sigma는 0.5~64, epsilon은 1e-6 이상")
    percentiles = cfg["output_percentiles"]
    if not isinstance(percentiles, list) or len(percentiles) != 2:
        raise ValueError("error:MSR output_percentiles는 [하위, 상위]")
    for value in percentiles:
        check_number(value, "msr.output_percentile")
    if not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("error:MSR 출력 백분위 범위 오류")
    if cfg["blur_mode"] not in {"pyramid", "direct"}:
        raise ValueError("error:MSR blur_mode는 pyramid/direct")
    return cfg


def _msr_surround(light, valid, sigma, cfg):
    """G(L*M)/G(M): ROI 밖 검정 패딩을 조도 추정에서 제외한다.

    pyramid는 큰 sigma의 조도장만 축소 공간에서 근사한다. 영상·GT를
    워핑하지 않는다. direct는 원래 해상도의 Gaussian으로 검증할 수 있다.
    """
    height, width = light.shape
    factor = max(1, math.ceil(sigma / cfg["max_blur_sigma"])) if cfg["blur_mode"] == "pyramid" else 1
    target = (max(1, math.ceil(width / factor)), max(1, math.ceil(height / factor)))
    weighted = np.dstack((light * valid, valid)).astype(np.float32)
    if factor > 1:
        weighted = cv2.resize(weighted, target, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(weighted, (0, 0), sigma * target[0] / width,
                               sigmaY=sigma * target[1] / height,
                               borderType=cv2.BORDER_REFLECT_101)
    illumination = blurred[:, :, 0] / np.maximum(blurred[:, :, 1], 1e-8)
    if factor > 1:
        illumination = cv2.resize(illumination, (width, height), interpolation=cv2.INTER_LINEAR)
    return illumination


def apply_msr(img, cfg=None, roi_mask=None):
    """Lab L의 다중 스케일 로그 반사율을 uint8 BGR로 복원한다.

    R = sum(w * (log(L + epsilon) - log(G_masked(L) + epsilon))).
    L만 ROI 내부 p1~p99(설정 가능)로 0~255 매핑하고 a/b는 유지한다.
    평탄한 영상/퇴화 범위는 원본 ROI를 복사한다. 원본 배열은 수정하지 않는다.
    이는 luminance MSR 변형이며 MSRCR 색 복원이나 Gamma/CLAHE가 아니다.
    """
    validate_image(img)
    settings = validate_msr_config(cfg)
    if __package__:
        from .roi import validate_mask
    else:
        from roi import validate_mask
    mask = validate_mask(roi_mask, img.shape) if roi_mask is not None else np.full(img.shape[:2], 255, np.uint8)
    valid = (mask != 0).astype(np.float32)
    if not settings["enabled"]:
        return apply_roi_mask(img.copy(), mask)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)
    values = light[mask != 0]
    if float(np.ptp(values)) <= 1.0:
        return apply_roi_mask(img.copy(), mask)
    log_source = np.log(light + settings["epsilon"])
    retinex = np.zeros(light.shape, np.float32)
    for sigma, weight in zip(settings["scales"], settings["weights"]):
        if weight:
            surround = _msr_surround(light, valid, sigma, settings)
            retinex += weight * (log_source - np.log(np.maximum(surround, 0) + settings["epsilon"]))
    if not np.isfinite(retinex).all():
        raise ValueError("error:MSR에서 비유한 값 발생")
    low, high = np.percentile(retinex[mask != 0], settings["output_percentiles"])
    if high - low < settings["min_log_range"]:
        return apply_roi_mask(img.copy(), mask)
    lab[:, :, 0] = np.rint(np.clip((retinex - low) / (high - low) * 255, 0, 255)).astype(np.uint8)
    return apply_roi_mask(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), mask)


# LEGACY_PREPROCESS_BEGIN
# # 기능: 공통 기준 영상에서 P0_reference·P1·P1+ 결과와 실제 적용 파라미터를 만든다.
# # 특징: P1+는 모두 CLAHE를 적용하고 자동 분류된 blur에만 Unsharp를 적용한다.
# #       Gamma 캐시는 이미지마다 새로 만든다. 직접 전달한 태그는 같은 분류 함수의 결과여야 한다.
# def preprocess_condition(reference, cfg, condition, tags=None, gamma_cache=None, roi_mask=None):
#     if condition not in CONDITIONS:
#         raise ValueError(f"error:알 수 없는 전처리 조건 {condition}")
#     params = {name: None for name in PARAMETER_COLUMNS}
#     params.update(stages=["ROI", "Resize"], gamma_channel="", clahe_channel="", unsharp_applied=False)
#     if condition == "P0_reference":
#         return apply_roi_mask(reference.copy(), roi_mask), params
#     if condition == "P1+":
#         if tags is None:
#             # 기능: 단독 호출에서도 runner와 같은 기준 영상으로 자동 분류한다.
#             # 특징: metrics의 역방향 import를 피하기 위해 호출 시점에 불러온다.
#             if __package__:
#                 from .metrics import measure_quality
#             else:
#                 from metrics import measure_quality
#             tags = classify_quality(measure_quality(reference, roi_mask))
#         if not isinstance(tags, (str, tuple, list)) or (not isinstance(tags, str) and any(not isinstance(t, str) for t in tags)):
#             raise ValueError("error:품질 태그는 문자열 또는 문자열 목록")
#         tags = parse_tags(tags if isinstance(tags, str) else ";".join(tags))
#     if gamma_cache is None:
#         result = apply_gamma(reference, cfg["gamma"]["value"])
#     else:
#         if not isinstance(gamma_cache, dict):
#             raise ValueError("error:Gamma 캐시는 사전이어야 함")
#         # 기능: 캐시가 다른 기준 영상·Gamma에 재사용되면 이전 결과를 제거한다.
#         # 특징: 픽셀 전체를 해시하지 않고 배열 객체와 파라미터를 비교한다. 기준 영상은 수정하지 않는다.
#         if gamma_cache.get("reference") is not reference or gamma_cache.get("gamma") != cfg["gamma"]["value"]:
#             gamma_cache.clear()
#             gamma_cache.update(reference=reference, gamma=cfg["gamma"]["value"])
#         if "image" not in gamma_cache:
#             start = time.perf_counter()
#             gamma_cache["image"] = apply_gamma(reference, cfg["gamma"]["value"])
#             gamma_cache["ms"] = (time.perf_counter() - start) * 1000
#         result = gamma_cache["image"]
#     params["stages"].append("Gamma")
#     params.update(gamma=cfg["gamma"]["value"], gamma_channel="BGR")
#     if condition == "P1+":
#         result = apply_clahe(result, cfg["clahe"])
#         params["stages"].append("CLAHE")
#         params.update(clahe_clip_limit=cfg["clahe"]["clip_limit"],
#                       clahe_tile_width=cfg["clahe"]["tile_grid"][0],
#                       clahe_tile_height=cfg["clahe"]["tile_grid"][1], clahe_channel="LAB_L")
#         if "blur" in tags:
#             result = apply_unsharp(result, cfg["unsharp"])
#             params["stages"].append("Unsharp")
#             params.update(unsharp_applied=True, unsharp_amount=cfg["unsharp"]["amount"],
#                           unsharp_sigma=cfg["unsharp"]["sigma"])
#     result = apply_gaussian(result, cfg["gaussian"])
#     params["stages"].append("Gaussian")
#     params.update(gaussian_kernel=cfg["gaussian"]["kernel"], gaussian_sigma=cfg["gaussian"]["sigma"])
#     return apply_roi_mask(result, roi_mask), params
#
#
# # 기능: 다음 역할에서 이미지 한 장을 보정할 수 있도록 preprocess(img, cfg) 인터페이스를 제공한다.
# # 특징: condition 기본값은 P1. P1+도 수동 태그 없이 자동 분류하며 파일 입출력 없이 새 BGR 영상을 반환한다.
# def preprocess(img, cfg, return_mask=False):
#     if not isinstance(cfg, dict):
#         raise ValueError("error:cfg는 보정 설정 사전")
#     settings = copy.deepcopy(cfg)
#     condition, image_id = settings.pop("condition", "P1"), settings.pop("image_id", "")
#     normalized = validate_config(settings)
#     reference, geometry = geometry_preprocess(img, normalized, image_id)
#     mask = geometry_mask(geometry)
#     result = preprocess_condition(reference, normalized, condition, roi_mask=mask)[0]
#     return (result, mask) if return_mask else result
#
# LEGACY_PREPROCESS_END


def preprocess(img, cfg=None, return_mask=False):
    """FINAL 설정만 사용한다. 기본 반환은 BGR이며 return_mask=True이면 마스크도 반환한다."""
    settings = copy.deepcopy(cfg) if cfg is not None else None
    if settings is not None:
        if not isinstance(settings, dict):
            raise ValueError("error:cfg는 FINAL 설정 사전이어야 함")
        condition = settings.pop("condition", "FINAL")
        if condition != "FINAL":
            raise ValueError("P0/P1/P1+는 비활성화되었습니다. FINAL만 사용할 수 있습니다.")
        settings.pop("image_id", None)
        if not settings:
            settings = None
        elif "version" not in settings:
            raise ValueError("final_preprocessing_config.json 형식의 FINAL 설정이 필요합니다")
    result, metadata = adaptive_preprocess(img, settings)
    return (result, metadata["roi_mask"]) if return_mask else result


def adaptive_preprocess(image, config=None, stage="FINAL"):
    """동결 FINAL 전처리의 공개 인터페이스. 기존 preprocess/P0/P1/P1+는 유지한다."""
    if __package__:
        from .adaptive_preprocess import adaptive_preprocess as run_adaptive
    else:
        from adaptive_preprocess import adaptive_preprocess as run_adaptive
    return run_adaptive(image, config, stage)
