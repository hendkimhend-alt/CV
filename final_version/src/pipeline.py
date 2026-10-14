"""사진 한 장을 처리하는 전체 흐름."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import ConfigError, check_keys, load_json
from detect import detect_cracks
from paths import ROOT
from pothole import detect_potholes
from preprocessing.detector_adapter import to_detector_input, to_original_bbox
from preprocessing.detector_corrections import apply_corrections, is_blurry
from preprocessing.preprocess import preprocess

PRE_CFG = ROOT / "configs" / "preprocessing.json"
DET_CFG = ROOT / "configs" / "detection.json"

CRACK_KEYS = ("line_sigmas", "tensor_propagate", "tensor_ma", "tensor_mi", "tensor_bins", "tensor_beta",
              "line_hi_abs", "line_lo_ratio", "crack_min_length", "crack_min_elong",
              "drop_border", "border_margin",
              "valley_min_ratio", "valley_gap", "valley_samples", "valley_radius", "cue_filter")
POTHOLE_KEYS = ("tex_k", "ring_ratio", "min_frac", "max_frac", "chroma_max", "darkp10_min", "aspect_max",
                "min_rel_y", "road_color_max")
GAMMA_KEYS = ("candidates", "dark_mean_below", "dark_ratio_above", "dark_pixel_value",
              "max_saturation_after", "measure_long_side",
              "erode_kernel", "min_pixels", "min_ratio_of_road")
VALIDATION_KEYS = ("mask_area_ratio", "largest_component_ratio", "hole_ratio", "seed_consistency")
CORRECTION_KEYS = ("gaussian", "line_close", "unsharp", "clahe", "order")
CUE_KEYS = ("energy", "coherence", "color", "depth_min")


@dataclass
class Result:
    ok: bool                         # 노면을 못 찾으면 False
    road_mask: np.ndarray = None
    boxes: list = field(default_factory=list)   # 원본 좌표 [{"type", "bbox": [x, y, w, h]}]
    unverified_mask: bool = False    # 검증 FAIL 마스크를 그대로 썼나
    blurry: bool = False
    detector_image: np.ndarray = None


def load_settings():
    pre, det = load_json(PRE_CFG), load_json(DET_CFG)
    check_keys(pre, ("road_mask", "validation", "gamma"), PRE_CFG.name)
    check_keys(pre["validation"], VALIDATION_KEYS, f"{PRE_CFG.name} validation")
    check_keys(pre["gamma"], GAMMA_KEYS, f"{PRE_CFG.name} gamma")
    check_keys(det, ("crop", "corrections", "crack", "pothole"), DET_CFG.name)
    check_keys(det["crop"], ("margin", "long_side", "min_road_overlap"), f"{DET_CFG.name} crop")
    check_keys(det["corrections"], CORRECTION_KEYS, f"{DET_CFG.name} corrections")
    check_keys(det["crack"], CRACK_KEYS, f"{DET_CFG.name} crack")
    check_keys(det["crack"]["cue_filter"], CUE_KEYS, f"{DET_CFG.name} crack.cue_filter")
    check_keys(det["pothole"], POTHOLE_KEYS, f"{DET_CFG.name} pothole")
    return {"preprocessing": pre, "detection": det}


def check_opencv():
    # 세선화(cv2.ximgproc.thinning)는 contrib 패키지에만 있다
    if not hasattr(cv2, "ximgproc"):
        raise ConfigError("cv2.ximgproc가 없습니다. pip install -r requirements.txt 로 "
                          "opencv-contrib-python을 설치하세요 "
                          "(opencv-python이 이미 있으면 먼저 pip uninstall opencv-python).")


def road_overlap(mask, bbox):
    # 박스 안에서 노면이 차지하는 비율 (검출기 좌표)
    x, y, w, h = (int(round(v)) for v in bbox)
    H, W = mask.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + max(w, 1)), min(H, y + max(h, 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.count_nonzero(mask[y0:y1, x0:x1])) / float((y1 - y0) * (x1 - x0))


def process(image, settings):
    pre = preprocess(image, settings["preprocessing"])
    if pre.road_mask is None:
        return Result(ok=False)

    det = settings["detection"]
    crop = det["crop"]
    img, mask, geo = to_detector_input(pre.image, pre.road_mask, crop["long_side"], crop["margin"])
    blurry = is_blurry(img, mask, geo, det["corrections"])
    img = apply_corrections(img, mask, det["corrections"], blurry)

    # 박스 면적의 절반 이상이 노면인 것만 남긴다
    boxes = []
    for d in detect_cracks(img, det["crack"]) + detect_potholes(img, mask, det["pothole"]):
        if road_overlap(mask, d["bbox"]) >= crop["min_road_overlap"]:
            x, y, w, h = to_original_bbox(d["bbox"], geo)
            bbox = [round(x, 1), round(y, 1), round(w, 1), round(h, 1)]
            boxes.append({"type": d["type"], "bbox": bbox})
    return Result(True, pre.road_mask, boxes, not pre.mask_passed, blurry, img)
