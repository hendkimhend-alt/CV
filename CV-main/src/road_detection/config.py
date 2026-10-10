"""검출 설정 — configs/detection_*.json 읽기 · 검증, 검출기 이름 → detect() 설정 변환.

설정 파일 형식
  {
    "description": "...",
    "detector": "D1v",                  # D0 / D1 + 뒤에 v(양쪽 확인) · l(조각 잇기) · h(Hessian 선 찾기)
    "valley_ratio": 0.35,               # (선택) 아래 값들은 null이면 detect.py 기본값
    "crack_lo_ratio": null, "crack_hi_pct": null, "link_dist": null, "line_hi_abs": null, "canny": null,
    "set": { "detect.py DEFAULT_CFG 키": 값, ... },      # 그 밖의 검출 설정 덮어쓰기
    "integration": {
      "input_mode": "crop_to_road" | "full",   # 검출기 입력: road_mask 외접 영역만 자름 / 영상 전체
      "crop_margin": 16,                       # crop_to_road일 때 외접 사각형 여유 (원본 화소)
      "long_side": 1024,                       # 검출기 입력 긴 변 (검출 파라미터가 이 배율 기준)
      "min_road_overlap": 0.5,                 # 후보 박스 면적 중 road_mask 비율이 이 값 이상이면 "도로 위"
      "on_fail": "skip" | "use_candidate"      # 자동 마스크 FAIL + 수동 마스크 없음일 때: 검출 건너뜀 / 검증 안 된 후보 마스크 사용
    }
  }
검출기 이름 해석(detector_cfg)은 팀의 이전 실행기 규칙을 그대로 옮긴 것이다 (검출 알고리즘 불변).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from detect import DEFAULT_CFG

INTEGRATION_DEFAULTS = {"input_mode": "crop_to_road", "crop_margin": 16, "long_side": 1024, "min_road_overlap": 0.5, "on_fail": "skip"}
TOP_KEYS = {"description", "detector", "valley_ratio", "crack_lo_ratio", "crack_hi_pct", "link_dist", "line_hi_abs", "canny", "set", "integration"}


class DetectionConfigError(ValueError):
    pass


def detector_cfg(name, valley_ratio=None, crack_lo_ratio=None, link_dist=None, crack_hi_pct=None, canny=None,
                 line_hi_abs=None, overrides=None):
    """검출기 이름 → detect() 설정. 뒤에 붙는 글자: v = 양쪽 확인, l = 조각 잇기, h = 찾기를 Hessian 선 점수로."""
    base, flags = name[:2], name[2:]
    if base not in ("D0", "D1") or set(flags) - {"v", "l", "h"} or len(set(flags)) != len(flags):
        raise DetectionConfigError(f"검출기 {name} — D0 / D1 + 뒤에 v · l · h (예: D1v, D1vl, D1hv)")
    cfg = {"detector": base, "valley_check": "v" in flags}
    if "h" in flags:
        cfg["crack_find"] = "line"
        cfg["line_hi_abs"] = line_hi_abs or 32.5
    if "l" in flags:
        cfg["link_dist"] = link_dist or 15
    if valley_ratio is not None:
        cfg["valley_min_ratio"] = valley_ratio
    if crack_lo_ratio is not None:
        cfg["crack_lo_ratio"] = crack_lo_ratio
    if crack_hi_pct is not None:
        cfg["crack_hi_pct"] = crack_hi_pct
    if canny is not None:
        cfg["canny_low"], cfg["canny_high"] = canny
    cfg.update(overrides or {})
    return cfg


def needs_ximgproc(cfg):
    """이 설정이 opencv-contrib(cv2.ximgproc)를 쓰는가 — Hessian 선 찾기(세선화) · 조각 잇기 · 가이드 필터."""
    merged = {**DEFAULT_CFG, **cfg}
    return merged["crack_find"] == "line" or merged["link_dist"] > 0 or bool(merged.get("guided_filter"))


def load_detection_config(path):
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DetectionConfigError(f"검출 설정 파일 없음: {path}") from None
    unknown = set(raw) - TOP_KEYS
    if unknown:
        raise DetectionConfigError(f"{path.name}: 모르는 키 {sorted(unknown)}")
    if "detector" not in raw:
        raise DetectionConfigError(f"{path.name}: detector가 필요함")
    bad = set(raw.get("set") or {}) - set(DEFAULT_CFG)
    if bad:
        raise DetectionConfigError(f"{path.name}: set의 {sorted(bad)}는 detect.py 설정에 없는 키")
    integration = {**INTEGRATION_DEFAULTS, **(raw.get("integration") or {})}
    unknown = set(integration) - set(INTEGRATION_DEFAULTS)
    if unknown:
        raise DetectionConfigError(f"{path.name}: integration의 모르는 키 {sorted(unknown)}")
    if integration["input_mode"] not in ("crop_to_road", "full"):
        raise DetectionConfigError("integration.input_mode는 crop_to_road 또는 full")
    if integration["on_fail"] not in ("skip", "use_candidate"):
        raise DetectionConfigError("integration.on_fail은 skip 또는 use_candidate")
    if not 0 <= integration["min_road_overlap"] <= 1:
        raise DetectionConfigError("integration.min_road_overlap은 0~1")
    cfg = detector_cfg(raw["detector"], raw.get("valley_ratio"), raw.get("crack_lo_ratio"), raw.get("link_dist"),
                       raw.get("crack_hi_pct"), tuple(raw["canny"]) if raw.get("canny") else None, raw.get("line_hi_abs"), raw.get("set"))
    return {"name": path.stem, "description": raw.get("description", ""), "detector_name": raw["detector"],
            "detect_cfg": cfg, "integration": integration, "needs_ximgproc": needs_ximgproc(cfg)}


def check_runtime(det):
    """설정이 필요로 하는 OpenCV 모듈이 있는지 실행 전에 확인한다."""
    if det["needs_ximgproc"] and not hasattr(cv2, "ximgproc"):
        raise DetectionConfigError(
            f"검출 설정 '{det['name']}'은 cv2.ximgproc(opencv-contrib-python)가 필요한데 이 환경에 없습니다. "
            "configs/detection_default.json을 쓰거나 opencv-contrib-python을 설치하세요 (README '환경').")
