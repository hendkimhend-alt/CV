"""road_mask를 침식해 품질 측정용 analysis_mask를 만든다."""
import cv2
import numpy as np

from .image_io import to_binary_mask

SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}


def make_analysis_mask(road_mask, cfg):
    size = cfg["kernel_size"]
    kernel = cv2.getStructuringElement(SHAPES[cfg["kernel_shape"]], (size, size))
    analysis = to_binary_mask(cv2.erode(road_mask.copy(), kernel, iterations=cfg["iterations"]))

    road_pixels = int(np.count_nonzero(road_mask))
    pixels = int(np.count_nonzero(analysis))
    ratio = pixels / road_pixels if road_pixels else 0.0
    sufficient = pixels >= cfg["min_pixels"] and ratio >= cfg["min_ratio_of_road"]
    info = {"kernel_shape": cfg["kernel_shape"], "kernel_size": size, "iterations": cfg["iterations"],
            "pixels": pixels, "road_pixels": road_pixels, "ratio_of_road": round(ratio, 6),
            "sufficient": sufficient, "min_pixels": cfg["min_pixels"],
            "min_ratio_of_road": cfg["min_ratio_of_road"]}
    return analysis, info
