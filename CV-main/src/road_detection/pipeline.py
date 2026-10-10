"""도로 손상 검출 시스템 — 전처리(자동 Road Mask · 보정)와 검출(detect.py)을 한 장 단위로 연결한다.

  입력 영상 → [전처리] 자동 Road Mask → PASS/FAIL → (수동 마스크) → analysis_mask → 조건부 Gamma · Gaussian
            → [연결] road_mask 외접 영역 자르기 · 긴 변 1024 (detector_adapter)
            → [검출] detect() — 알고리즘 변경 없음
            → [연결] 후보 박스가 road_mask 위에 있는지 판정 (min_road_overlap) · 원본 좌표로 되돌림
            → 결과 (전처리 결과 + 후보 목록)

자동 마스크가 FAIL이고 수동 마스크가 없으면 기본은 검출을 건너뛴다 (on_fail="skip").
on_fail="use_candidate"면 검증되지 않은 후보 마스크로 계속하며, 결과에 그 사실을 표시한다.
"""
from __future__ import annotations

import time

import numpy as np

from detect import detect
from preprocessing.detector_adapter import to_detector_input, to_original_bbox
from preprocessing.pipeline import MANUAL_REQUIRED, SUCCESS, process_image

KINDS = ("crack", "pothole")


def road_overlap(det_mask, bbox):
    """검출기 좌표 박스 (x, y, w, h) 안에서 road_mask가 차지하는 비율."""
    x, y, w, h = (int(round(v)) for v in bbox)
    H, W = det_mask.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + max(w, 1)), min(H, y + max(h, 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.count_nonzero(det_mask[y0:y1, x0:x1])) / float((y1 - y0) * (x1 - x0))


def detect_on_road(processed_image, road_mask, det):
    """전처리 결과 한 장에 검출을 돌리고 후보마다 도로 위 여부와 원본 좌표를 붙인다."""
    integ = det["integration"]
    t0 = time.perf_counter()
    det_img, det_mask, geometry = to_detector_input(processed_image, road_mask, integ["long_side"],
                                                    crop_to_road=integ["input_mode"] == "crop_to_road",
                                                    crop_margin=integ["crop_margin"])
    raw = detect(det_img, det["detect_cfg"])
    elapsed = (time.perf_counter() - t0) * 1000
    candidates = []
    for k, d in enumerate(raw):
        overlap = road_overlap(det_mask, d["bbox"])
        x, y, w, h = to_original_bbox(d["bbox"], geometry)
        candidates.append({"id": k, "type": d["type"], "branch": d.get("branch"),
                           "bbox_original": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                           "bbox_detector": [int(v) for v in d["bbox"]],
                           "road_overlap": round(overlap, 4), "on_road": overlap >= integ["min_road_overlap"],
                           **{key: (round(float(d[key]), 4) if d.get(key) is not None else None)
                              for key in ("area", "length", "width", "elong", "contrast")}})
    return candidates, geometry, det_img, det_mask, elapsed


def run_image(image, pre_cfg, det, *, image_id="image", input_path=None, manual_mask=None, manual_source=None, editor=None):
    """→ dict(status, preprocess(PreprocessResult), candidates, geometry, unverified_mask, timing)."""
    t_all = time.perf_counter()
    pre = process_image(image, pre_cfg, image_id=image_id, input_path=input_path, manual_mask=manual_mask,
                        manual_source=manual_source, editor=editor)
    unverified = False
    if pre.status == MANUAL_REQUIRED and det["integration"]["on_fail"] == "use_candidate" and pre.auto_mask is not None \
            and np.count_nonzero(pre.auto_mask):
        # 검증되지 않은 자동 후보 마스크로 전처리를 끝까지 진행 (메타데이터에 표시)
        pre = process_image(image, pre_cfg, image_id=image_id, input_path=input_path, manual_mask=pre.auto_mask,
                            manual_source="auto_candidate_unverified")
        unverified = True
    out = {"status": pre.status, "preprocess": pre, "candidates": [], "geometry": None, "unverified_mask": unverified,
           "timing_ms": {"preprocess": pre.metadata["timing_ms"].get("total")}, "error": None}
    if pre.status != SUCCESS:
        out["status"] = "skipped_" + pre.status
    else:
        try:
            cands, geometry, _, _, ms = detect_on_road(pre.processed_image, pre.road_mask, det)
            out.update(candidates=cands, geometry=geometry)
            out["timing_ms"]["detection"] = ms
            out["status"] = "detected"
        except Exception as exc:          # 한 장의 실패를 기록하고 배치는 계속
            out.update(status="error_detection", error=f"{type(exc).__name__}: {exc}")
    out["timing_ms"]["total"] = (time.perf_counter() - t_all) * 1000
    pre.metadata["detection"] = {
        "status": out["status"], "detector": det["detector_name"], "detection_config": det["name"],
        "integration": det["integration"], "unverified_mask": unverified, "geometry": out["geometry"],
        "counts_on_road": {k: sum(c["on_road"] and c["type"] == k for c in out["candidates"]) for k in KINDS},
        "n_off_road": sum(not c["on_road"] for c in out["candidates"]),
        "timing_ms": out["timing_ms"], "error": out["error"]}
    return out
