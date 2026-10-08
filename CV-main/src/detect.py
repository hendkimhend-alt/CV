"""
개발 B - 검출기.

detect(img, cfg) -> list[{bbox, type, area, length, width, elong, contrast, branch}]

입력 img는 A의 preprocess() 출력(노면 ROI + 긴 변 1024 정규화)을 가정한다.
혼자 돌릴 때는 run_detect.py의 prepare()가 같은 처리를 해 준다.

- D0 (비교 기준): Canny → Closing
- D1: 균열 branch   Black-hat(작은 커널) → [선 열림] → 이중 임계값 → Closing
      포트홀 branch Adaptive Threshold(큰 블록) → Opening
- 공통: 연결 요소 → [가까운 조각 묶기] → 형태 특징 → 규칙으로 crack / pothole / noise → [가장자리 제외]

[ ]는 cfg로 켜고 끄는 단계. 기본값: 가장자리 제외만 켜짐. 선 열림·조각 묶기는 RDD 평가에서 효과가 없어 꺼 둠
(재현용으로 남김 — analysis/eval_detector.py). 계획서 원안(adaptive threshold)은 crack_thresh="adaptive"로 재현 가능.
bbox는 (x, y, w, h), 입력 img 좌표계 기준.
"""
import cv2
import numpy as np

if __package__:
    from .roi import interior_mask, validate_mask
else:
    from roi import interior_mask, validate_mask

DEFAULT_CFG = {
    "detector": "D1",            # "D0" | "D1"
    "keep_noise": False,         # True면 noise 판정 후보도 반환

    # D0
    "canny_low": 50,
    "canny_high": 150,
    "d0_close_ksize": 5,

    # D1 균열 branch
    "blackhat_ksize": 7,         # 균열 폭(2~3px)보다 크고 그림자 띠보다 작게 [L4]. 15는 그림자 띠까지 잡음
    "line_len": 0,               # [실험, 효과 없음] >0이면 Black-hat 결과를 막대로 여러 방향 열림 → 선 모양만 남김
    "line_angles": 12,           # 막대 방향 개수 (180° / 12 = 15° 간격)
    "crack_thresh": "hysteresis",  # "hysteresis" | "adaptive" (계획서 원안, 비교용)
    # hysteresis: 강한 픽셀(씨앗)과 이어진 약한 픽셀만 살림 — Canny 이중 임계값 [3-1]을 Black-hat에 적용
    "crack_hi_pct": 97.0,        # 강한 기준 = 사진 안 Black-hat 상위 3% (RDD 비교: 99는 재현율 하락)
    "crack_hi_min": 20,          # 단, 최소 20 (흐린 사진에서 잡음이 씨앗이 되는 것 방지)
    "crack_lo_ratio": 0.5,       # 약한 기준 = 강한 기준 × 0.5
    "crack_block": 31,           # adaptive 모드 전용 (홀수)
    "crack_C": -5,               # adaptive 모드 전용
    "crack_close_ksize": 3,      # 5는 그림자 조각까지 이어 붙임

    # D1 포트홀 branch
    "pothole_block": 101,        # 큰 블록 → 덩어리 단위로 어두운 곳 (홀수)
    "pothole_C": 15,             # 국소 평균보다 15 이상 어두움
    "pothole_open_ksize": 9,

    # 덩어리
    "group_ksize": 0,            # [실험, 효과 없음] >0이면 이 거리 안의 조각을 한 후보로 묶음 (모양은 원래 픽셀로 잼)

    # 규칙
    "min_area_ratio": 0.0002,    # 이보다 작으면 noise
    "crack_min_elong": 3.0,      # 세장비 ≥ 이 값 → crack
    "crack_min_contrast": 0.0,   # 덩어리 평균 Black-hat / 강한 기준 ≥ 이 값 → crack (0 = 끔)
    "pothole_min_area_ratio": 0.002,
    "pothole_max_elong": 3.0,    # 세장비 < 이 값 + 면적 충분 → pothole
    "drop_border": "t",          # 이 변에 닿는 후보 버림: "t"(위) "b"(아래) "l"(왼) "r"(오른) 조합.
                                 # 위(ROI 경계 = 원경 차량·인도)만 — 아래까지 버리면 화면 밖으로 뻗은 균열을 놓침
    "border_margin": 2,
}


def detect(img, cfg=None, roi_mask=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    gray = _to_gray(img)
    H, W = gray.shape
    valid = None
    img_area = H * W
    if roi_mask is not None:
        roi_mask = validate_mask(roi_mask, gray.shape)
        if not np.all(roi_mask):
            img_area = int(np.count_nonzero(roi_mask))
            # 기존 border_margin을 사용해 검정 패딩의 인공 경계만 후보에서 제외한다.
            valid = interior_mask(roi_mask, max(1, cfg["border_margin"]))

    if cfg["detector"] == "D0":
        branches = [("any", _d0_mask(gray, cfg), None, None)]
    elif cfg["detector"] == "D1":
        mask, bh, hi = _crack_mask(gray, cfg, valid)
        branches = [("crack", mask, bh, hi), ("pothole", _pothole_mask(gray, cfg), None, None)]
    else:
        raise ValueError(f"unknown detector: {cfg['detector']}")

    detections = []
    for branch, mask, bh, hi in branches:
        if valid is not None:
            mask = cv2.bitwise_and(mask, valid)
        for pts in _components(mask, cfg["group_ksize"]):
            det = shape_features(pts, bh, hi)
            det["type"] = classify(det, img_area, branch, cfg)
            det["branch"] = branch
            if det["type"] != "noise" and _touches_border(det["bbox"], W, H, cfg):
                det["type"] = "noise"
            if det["type"] != "noise" or cfg["keep_noise"]:
                detections.append(det)
    return detections


# ---------- 마스크 ----------

def _d0_mask(gray, cfg):
    edges = cv2.Canny(gray, cfg["canny_low"], cfg["canny_high"])
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _kernel(cfg["d0_close_ksize"]))


def _crack_mask(gray, cfg, roi_mask=None):
    # 노면보다 어둡고 가는 구조만 밝게 남긴다
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _kernel(cfg["blackhat_ksize"]))
    if cfg["line_len"] > 0:
        bh = _line_open(bh, cfg["line_len"], cfg["line_angles"])
    if cfg["crack_thresh"] == "hysteresis":
        binary, hi = _hysteresis(bh, cfg, roi_mask)
    else:
        binary = cv2.adaptiveThreshold(bh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       cfg["crack_block"], cfg["crack_C"])
        hi = max(float(bh.max()), 1.0)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel(cfg["crack_close_ksize"])), bh, hi


def _line_open(bh, length, n_angles):
    # 막대 모양 구조 요소로 열림: 그 방향으로 length 이상 이어진 밝은 구조만 살아남음.
    # 여러 방향의 최댓값 → 어느 방향이든 "선"이면 남고, 점·짧은 조각은 사라짐
    out = np.zeros_like(bh)
    for i in range(n_angles):
        out = np.maximum(out, cv2.morphologyEx(bh, cv2.MORPH_OPEN, _line_kernel(length, 180.0 * i / n_angles)))
    return out


def _hysteresis(bh, cfg, roi_mask=None):
    # 기준을 사진마다 정함: Black-hat 값 범위가 사진마다 수십 배 차이 (골재 사진 중앙값 56, 흐린 사진 0)
    samples = bh if roi_mask is None else bh[roi_mask != 0]
    hi = max(cfg["crack_hi_min"], float(np.percentile(samples, cfg["crack_hi_pct"]))) if samples.size else cfg["crack_hi_min"]
    lo = hi * cfg["crack_lo_ratio"]
    weak = (bh >= lo).astype(np.uint8)
    if roi_mask is not None:
        weak[roi_mask == 0] = 0
    n, labels = cv2.connectedComponents(weak, connectivity=8)
    seeded = np.zeros(n, bool)
    seeded[np.unique(labels[bh >= hi])] = True
    seeded[0] = False                       # 배경
    return (seeded[labels] * 255).astype(np.uint8), hi


def _pothole_mask(gray, cfg):
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                   cfg["pothole_block"], cfg["pothole_C"])
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, _kernel(cfg["pothole_open_ksize"]))


# ---------- 덩어리 ----------

def _components(mask, group_ksize=0):
    """흰 픽셀을 연결 요소 [2-1]로 묶어 덩어리별 (x, y) 좌표 배열 목록으로."""
    grouping = cv2.dilate(mask, _kernel(group_ksize)) if group_ksize > 0 else mask
    n, labels = cv2.connectedComponents((grouping > 0).astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(mask)               # 묶음은 넓힌 마스크로, 모양은 원래 픽셀로
    lab = labels[ys, xs]
    order = np.argsort(lab, kind="stable")
    lab, pts = lab[order], np.stack([xs[order], ys[order]], axis=1).astype(np.int32)
    cuts = np.flatnonzero(np.diff(lab)) + 1
    return [p for p in np.split(pts, cuts) if len(p)]


# ---------- 형태 특징 · 규칙 ----------

def shape_features(pts, bh=None, hi=None):
    x, y, w, h = cv2.boundingRect(pts)
    area = float(len(pts))                       # 픽셀 수
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)   # 회전 박스 → 대각선 균열도 세장비가 나옴
    rw, rh = rw + 1, rh + 1                      # 픽셀 중심 기준이라 1px 보정
    length = max(rw, rh)
    width = area / length                        # 평균 폭 ≈ 면적 / 길이
    elong = length / min(rw, rh)
    contrast = float(bh[pts[:, 1], pts[:, 0]].mean() / hi) if bh is not None else 0.0
    return {"bbox": (x, y, w, h), "area": area, "length": float(length),
            "width": float(width), "elong": float(elong), "contrast": contrast}


def classify(det, img_area, branch, cfg):
    area_ratio = det["area"] / img_area
    if area_ratio < cfg["min_area_ratio"]:
        return "noise"
    # 규칙 1: 세장비 (+ 진하기)
    if (branch in ("crack", "any") and det["elong"] >= cfg["crack_min_elong"]
            and det["contrast"] >= cfg["crack_min_contrast"]):
        return "crack"
    # 규칙 2: 면적
    if (branch in ("pothole", "any") and area_ratio >= cfg["pothole_min_area_ratio"]
            and det["elong"] < cfg["pothole_max_elong"]):
        return "pothole"
    return "noise"


def _touches_border(bbox, W, H, cfg):
    x, y, w, h = bbox
    m = cfg["border_margin"]
    sides = cfg["drop_border"]
    return (("t" in sides and y <= m) or ("b" in sides and y + h >= H - m)
            or ("l" in sides and x <= m) or ("r" in sides and x + w >= W - m))


# ---------- 유틸 ----------

def _to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _line_kernel(length, angle_deg):
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2
    dx, dy = np.cos(np.radians(angle_deg)) * c, -np.sin(np.radians(angle_deg)) * c
    cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return k
