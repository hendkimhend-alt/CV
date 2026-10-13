"""조건부 Gamma 보정 (도로 영역 안에서만). 공식: out = 255 * (in / 255) ^ gamma"""
from functools import lru_cache

import cv2
import numpy as np

CONDITIONS = (("gray_mean_below", "gray_mean", "<"), ("dark_ratio_above", "dark_ratio", ">"))


@lru_cache(maxsize=32)
def gamma_lut(gamma):
    lut = np.rint(255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** gamma).astype(np.uint8)
    lut.setflags(write=False)
    return lut


def apply_gamma(img, gamma, mask=None):
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or gamma <= 0:
        raise ValueError("gamma는 0보다 큰 수")
    corrected = cv2.LUT(img, gamma_lut(float(gamma)))
    if mask is None:
        return corrected
    out = img.copy()
    road = mask > 0
    out[road] = corrected[road]
    return out


def evaluate_conditions(metrics, conds):
    out = []
    for name, metric, op in CONDITIONS:
        rule = conds[name]
        if not rule["enabled"]:
            continue
        value = metrics.get(metric)
        if value is None:
            result = None
        elif op == "<":
            result = bool(value < rule["threshold"])
        else:
            result = bool(value > rule["threshold"])
        out.append({"name": name, "metric": metric, "op": op,
                    "threshold": rule["threshold"], "value": value, "result": result})
    return out


def decide_gamma(metrics, cfg_gamma, analysis_sufficient):
    mode = cfg_gamma["mode"]
    decision = {"apply": False, "mode": mode, "value": cfg_gamma["value"], "logic": None,
                "conditions": [], "reason": ""}
    if mode == "off":
        decision["reason"] = "mode_off"
        return decision
    if mode == "fixed":
        decision["apply"] = cfg_gamma["value"] != 1.0
        decision["reason"] = "fixed_mode" if decision["apply"] else "gamma_is_1"
        return decision

    conds = cfg_gamma["conditions"]
    decision["logic"] = conds["logic"]
    if not analysis_sufficient:
        decision["reason"] = "insufficient_analysis_pixels"
        return decision
    decision["conditions"] = evaluate_conditions(metrics, conds)
    results = [c["result"] for c in decision["conditions"]]
    if None in results:
        decision["reason"] = "metric_unavailable"
        return decision

    triggered = all(results) if conds["logic"] == "AND" else any(results)
    if cfg_gamma["selection"] == "adaptive":
        has_gamma = any(c < 1.0 for c in cfg_gamma["candidates"])
    else:
        has_gamma = cfg_gamma["value"] != 1.0
    decision["apply"] = triggered and has_gamma
    if decision["apply"]:
        decision["reason"] = "conditions_met"
    else:
        decision["reason"] = "gamma_is_1" if triggered else "conditions_not_met"
    return decision


def choose_gamma(image, mask, before, cfg_gamma, measure):
    """후보 중 약한 gamma부터 적용해 보고 어두움 조건이 풀리는 첫 값을 고른다.

    끝까지 안 풀리면 포화 방지를 통과한 가장 강한 값. 전부 포화되면 (None, None, None, tried).
    """
    tried, best = [], None
    for g in sorted({float(c) for c in cfg_gamma["candidates"] if c < 1.0}, reverse=True):
        img = apply_gamma(image, g, mask)
        after = measure(img)
        guard = bool(guard_triggered(before, after, cfg_gamma))
        conds = evaluate_conditions(after, cfg_gamma["conditions"])
        resolved = all(c["result"] is False for c in conds)
        tried.append({"value": g, "gray_mean": after.get("gray_mean"), "dark_ratio": after.get("dark_ratio"),
                      "saturation_ratio": after.get("saturation_ratio"),
                      "guard_triggered": guard, "resolved": resolved})
        if guard:          # 더 작은 gamma는 더 밝아지므로 볼 필요 없음
            break
        best = (g, img, after)
        if resolved:
            break
    if best is None:
        return None, None, None, tried
    return (*best, tried)


def guard_triggered(before, after, cfg_gamma):
    """보정 뒤 포화 화소가 한도를 넘고 늘었으면 True."""
    limit = cfg_gamma["guard"]["max_saturation_ratio_after"]
    sat_before, sat_after = before.get("saturation_ratio"), after.get("saturation_ratio")
    if limit is None or sat_after is None:
        return False
    return sat_after > limit and (sat_before is None or sat_after > sat_before)
