"""정규화 사다리꼴 ROI와 유효 픽셀 마스크. 후보 탐색이나 모델 추론은 하지 않는다."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np

COORDINATE_KEYS = (
    "top_y_ratio", "top_left_x_ratio", "top_right_x_ratio",
    "bottom_left_x_ratio", "bottom_right_x_ratio", "bottom_y_ratio",
)


def validate_trapezoid(roi):
    if not isinstance(roi, dict) or set(roi) != {"type", *COORDINATE_KEYS} or roi["type"] != "trapezoid":
        raise ValueError("error:사다리꼴 ROI는 type과 정규화 좌표 6개가 필요함")
    for key in COORDINATE_KEYS:
        value = roi[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"error:{key}는 0~1의 유한한 숫자여야 함")
    if not roi["top_y_ratio"] < roi["bottom_y_ratio"]:
        raise ValueError("error:사다리꼴 위 y는 아래 y보다 작아야 함")
    if not roi["bottom_left_x_ratio"] <= roi["top_left_x_ratio"] < roi["top_right_x_ratio"] <= roi["bottom_right_x_ratio"]:
        raise ValueError("error:사다리꼴 위쪽 좌우는 아래쪽 좌우 안에 있어야 함")


def build_trapezoid_mask(height, width, roi):
    """분석 도구와 동일한 픽셀 중심·반올림·LINE_8 규칙을 사용한다."""
    validate_trapezoid(roi)
    points = np.rint(np.array([
        [roi["top_left_x_ratio"] * (width - 1), roi["top_y_ratio"] * (height - 1)],
        [roi["top_right_x_ratio"] * (width - 1), roi["top_y_ratio"] * (height - 1)],
        [roi["bottom_right_x_ratio"] * (width - 1), roi["bottom_y_ratio"] * (height - 1)],
        [roi["bottom_left_x_ratio"] * (width - 1), roi["bottom_y_ratio"] * (height - 1)],
    ])).astype(np.int32)
    if height < 2 or width < 2 or cv2.contourArea(points) <= 0:
        raise ValueError("error:이미지가 너무 작아 사다리꼴 ROI를 만들 수 없음")
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, points, 255, lineType=cv2.LINE_8)
    return mask, points


def validate_mask(mask, shape):
    if not isinstance(mask, np.ndarray) or mask.shape != tuple(shape[:2]) or mask.dtype not in (np.uint8, np.bool_):
        raise ValueError("error:ROI 마스크는 영상과 같은 높이·너비의 uint8 또는 bool 배열이어야 함")
    mask = (mask != 0).astype(np.uint8) * 255
    if not mask.any():
        raise ValueError("error:ROI 마스크에 유효 픽셀이 없음")
    return mask


def geometry_mask(geometry):
    """원본 마스크를 같은 bbox로 자르고 NEAREST로 크기를 맞춘다. 직사각형 ROI는 None."""
    encoded = geometry.get("roi_polygon_json")
    if not encoded:
        return None
    points = np.array(json.loads(encoded), np.int32)
    mask = np.zeros((geometry["original_height"], geometry["original_width"]), np.uint8)
    cv2.fillConvexPoly(mask, points, 255, lineType=cv2.LINE_8)
    x, y, width, height = (geometry[k] for k in ("crop_x", "crop_y", "crop_width", "crop_height"))
    mask = mask[y:y + height, x:x + width]
    size = (geometry["output_width"], geometry["output_height"])
    if size != (width, height):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask


def apply_roi_mask(image, mask):
    return image if mask is None else cv2.bitwise_and(image, image, mask=validate_mask(mask, image.shape))


def interior_mask(mask, margin=1):
    """인공 경계에 걸치는 필터 응답을 제외한다. 실제 영상 가장자리는 유지한다."""
    return cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=margin,
                     borderType=cv2.BORDER_CONSTANT, borderValue=255)


def bbox_intersection_area(integral, bbox):
    """[x1,x2)×[y1,y2) 박스의 실제 마스크 교집합 면적. 소수 좌표도 보존한다."""
    height, width = np.array(integral.shape) - 1
    def at(x, y):
        x, y = float(np.clip(x, 0, width)), float(np.clip(y, 0, height))
        ix, iy = math.floor(x), math.floor(y)
        jx, jy = min(ix + 1, width), min(iy + 1, height)
        fx, fy = x - ix, y - iy
        return ((1 - fy) * ((1 - fx) * integral[iy, ix] + fx * integral[iy, jx])
                + fy * ((1 - fx) * integral[jy, ix] + fx * integral[jy, jx]))
    x1, y1, x2, y2 = bbox
    return max(0.0, float(at(x2, y2) - at(x1, y2) - at(x2, y1) + at(x1, y1)))
