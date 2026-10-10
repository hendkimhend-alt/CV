"""영상 · 마스크 읽기/쓰기와 형식 검사. 마스크는 uint8, 도로 255 / 나머지 0."""
from pathlib import Path

import cv2
import numpy as np

from paths import imread, imwrite

ROAD_VALUE = 255
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageLoadError(ValueError):
    pass


def validate_image(img):
    ok = (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3
          and img.shape[2] == 3 and min(img.shape[:2]) > 0)
    if not ok:
        raise ImageLoadError("입력은 비어 있지 않은 uint8 BGR(H×W×3) 이미지여야 함")
    return img


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"이미지 파일 없음: {path}")
    try:
        img = imread(path, cv2.IMREAD_COLOR)
    except ValueError as exc:
        raise ImageLoadError(str(exc)) from None
    return validate_image(img)


def to_binary_mask(mask):
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("마스크는 2차원 배열이어야 함")
    return np.where(mask > 0, ROAD_VALUE, 0).astype(np.uint8)


def validate_mask(mask, shape=None):
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("마스크는 uint8 2차원 배열이어야 함")
    if shape is not None and mask.shape != tuple(shape[:2]):
        raise ValueError(f"마스크 크기 {mask.shape}가 영상 크기 {tuple(shape[:2])}와 다름")
    if not set(np.unique(mask).tolist()) <= {0, ROAD_VALUE}:
        raise ValueError("마스크 값은 0 또는 255만 가능")
    return mask


def read_mask(path, shape, threshold=128):
    """회색조로 읽어 threshold 이상을 도로로 본다."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"마스크 파일 없음: {path}")
    try:
        raw = imread(path, cv2.IMREAD_GRAYSCALE)
    except ValueError as exc:
        raise ImageLoadError(str(exc)) from None
    if raw.shape != tuple(shape[:2]):
        raise ValueError(f"마스크 크기 {raw.shape}가 영상 크기 {tuple(shape[:2])}와 다름 ({path.name})")
    return np.where(raw >= threshold, ROAD_VALUE, 0).astype(np.uint8)


def write_image(path, img):
    imwrite(Path(path), img)
    return Path(path)
