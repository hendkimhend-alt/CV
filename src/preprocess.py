"""공통 ROI · Resize와 품질값 자동 분류.

보정 단계(감마 · CLAHE · 언샤프 · 가우시안 · 조명 펴기)는 10/13에 정리했다 — 최종 구성의 보정은
CV-main/src/preprocessing/detector_corrections.py (노면 마스크 안에만, 연결 뒤 · 검출 앞). 실험 기록: results/experiments.md
여기 남은 것: ROI 자르기 · 긴 변 1024 맞추기(geometry_preprocess) · 품질 그룹 분류(classify_quality) — 분석 스크립트용
조건은 "none"(보정 없음)만 받는다.
"""

from __future__ import annotations

import copy
import json
import math

import cv2
import numpy as np

CONDITIONS = ("none",)                                     # 보정 없음만 (보정 단계는 10/13 정리)
VALID_TAGS = {"blur", "local_illumination", "normal", "unclassified"}
# 기능: 보정 전 ROI·Resize 영상의 품질값으로 그룹을 자동 결정한다.
# 특징: 경계값 50·44는 포함하지 않는다. 필요하면 아래 숫자만 수정한다.
AUTO_GROUP_THRESHOLDS = {"blur_laplacian_lt": 50.0, "local_block_std_gt": 44.0}
INTERPOLATIONS = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC,
                  "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST,
                  "lanczos": cv2.INTER_LANCZOS4}
DEFAULT_CONFIG = {
    "roi": "bottom_half", "roi_overrides": {},
    "resize": {"long_side": 1024, "allow_upscale": True,
               "down_interpolation": "area", "up_interpolation": "cubic"},
}
GEOMETRY_COLUMNS = [
    "original_width", "original_height", "crop_x", "crop_y", "crop_width", "crop_height",
    "output_width", "output_height", "scale_x", "scale_y", "roi_json", "allow_upscale",
    "resize_interpolation",
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


# 기능: "bottom_<N>" 형식이면 N(사진 아래쪽에서 남길 높이 %, 1~100)을 돌려준다. 아니면 None.
# 특징: bottom_half는 기존 결과를 그대로 재현하려고 따로 둔다 (bottom_50과 홀수 높이에서 1픽셀 다를 수 있음).
def roi_bottom_percent(roi):
    if not isinstance(roi, str) or not roi.startswith("bottom_") or roi == "bottom_half":
        return None
    text = roi[len("bottom_"):]
    if not text.isdigit() or not 1 <= int(text) <= 100:
        raise ValueError("error:ROI bottom_<N>의 N은 1~100 정수 (아래쪽에서 남길 높이 %)")
    return int(text)


# 기능: ROI 형식과 좌표·크기를 검사한다.
# 특징: 이미지 경계 검사는 실제 해상도를 아는 geometry_preprocess에서 수행한다.
def check_roi(roi):
    if isinstance(roi, str) and (roi in {"bottom_half", "full", "auto"} or roi_bottom_percent(roi)):
        return
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("error:ROI는 bottom_half/full/auto/bottom_<N> 또는 [x,y,width,height]")
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
        if isinstance(cfg[key], dict) and key != "roi_overrides":
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


# 기능: 공통 ROI를 자르고 긴 변을 정규화하며 좌표 변환용 실제 기하 정보를 반환한다.
# 특징: A1의 홀수 높이·종횡비 반올림 규칙을 유지하고 새 출력 배열로 원본을 보호한다.
def geometry_preprocess(img, cfg, image_id=""):
    validate_image(img)
    original_h, original_w = img.shape[:2]
    roi = cfg["roi_overrides"].get(image_id, cfg["roi"])
    if roi == "bottom_half":
        x, y, width, height = 0, original_h // 2, original_w, original_h - original_h // 2
    elif roi == "full":
        x, y, width, height = 0, 0, original_w, original_h
    elif roi == "auto":
        # 사진마다 도로가 시작하는 높이를 찾아 그 위만 자름 (src/roi.py)
        if __package__:
            from .roi import road_top
        else:
            from roi import road_top
        y = round(original_h * road_top(img))
        x, width, height = 0, original_w, original_h - y
    elif roi_bottom_percent(roi):
        height = max(1, round(original_h * roi_bottom_percent(roi) / 100))
        x, y, width = 0, original_h - height, original_w
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
                    resize_interpolation=interpolation)
    return cropped, geometry


# 기능: 조건 이름 확인 — 보정 단계는 정리되어 "none"만 받는다.
def parse_condition(condition):
    if condition != "none":
        raise ValueError(f"error:전처리 조건 {condition} — 보정 단계는 10/13 정리, \"none\"만 (최종 보정은 detector_corrections)")
    return ()


# 기능: 공통 기준 영상을 그대로 돌려준다 (예전 호출 모양 유지).
def preprocess_condition(reference, cfg, condition="none", tags=None, gamma_cache=None):
    parse_condition(condition)
    return reference.copy(), {"stages": ["ROI", "Resize"]}


# 기능: 이미지 한 장을 ROI · Resize 한다 (preprocess(img, cfg) 인터페이스).
def preprocess(img, cfg):
    if not isinstance(cfg, dict):
        raise ValueError("error:cfg는 설정 사전")
    settings = copy.deepcopy(cfg)
    condition, image_id = settings.pop("condition", "none"), settings.pop("image_id", "")
    normalized = validate_config(settings)
    reference, _ = geometry_preprocess(img, normalized, image_id)
    return preprocess_condition(reference, normalized, condition)[0]
