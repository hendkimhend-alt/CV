"""전처리: 노면 마스크, 검증, Gamma."""
from dataclasses import dataclass, field

import numpy as np

from .gamma import correct_gamma
from .mask_validation import validate_road_mask
from .road_mask import extract_road_mask


@dataclass
class Preprocessed:
    road_mask: np.ndarray            # uint8 0/255, 원본 크기. 노면을 못 찾으면 None
    image: np.ndarray = None         # Gamma까지 적용한 영상 (노면 밖 화소는 원본 그대로)
    mask_passed: bool = False        # 자동 마스크가 검증을 통과했나
    fail_reasons: list = field(default_factory=list)


def preprocess(image, cfg):
    road_mask, seed_points = extract_road_mask(image, cfg["road_mask"])
    passed, reasons = validate_road_mask(road_mask, seed_points, cfg["validation"])
    if not np.any(road_mask):
        # 시드는 있었는데 영역이 하나도 안 자란 경우
        return Preprocessed(None, mask_passed=False, fail_reasons=reasons)
    # FAIL이어도 이 마스크로 계속 간다. 검증 결과는 화면 표시에만 쓴다
    image = correct_gamma(image, road_mask, cfg["gamma"])
    return Preprocessed(road_mask, image, passed, reasons)
