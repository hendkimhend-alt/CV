"""조건부 Gaussian (도로 영역 안에서만). 현재 설정에서는 꺼져 있다."""
import cv2
import numpy as np


def gaussian_blur(img, kernel, sigma):
    return cv2.GaussianBlur(img, (kernel, kernel), sigma, borderType=cv2.BORDER_REFLECT_101)


def masked_gaussian(img, mask, kernel, sigma, mask_aware=True):
    if sigma <= 0:
        raise ValueError("sigma는 0보다 커야 함 (끄려면 mode=off)")
    road = mask > 0
    out = img.copy()
    if not road.any():
        return out
    if mask_aware:
        # G(I*M) / G(M): 경계에서 도로 밖 화소가 섞이지 않게 한다
        weight = road.astype(np.float32)
        numerator = gaussian_blur(img.astype(np.float32) * weight[..., None], kernel, sigma)
        denominator = np.maximum(gaussian_blur(weight, kernel, sigma), 1e-6)
        filtered = np.clip(np.rint(numerator / denominator[..., None]), 0, 255).astype(np.uint8)
    else:
        filtered = gaussian_blur(img, kernel, sigma)
    np.copyto(out, filtered, where=road[..., None])
    return out


def decide_gaussian(metrics, cfg_gaussian, analysis_sufficient):
    mode = cfg_gaussian["mode"]
    noise = metrics.get("noise_sigma")
    decision = {"apply": False, "mode": mode, "sigma": cfg_gaussian["sigma"], "kernel": cfg_gaussian["kernel"],
                "mask_aware": cfg_gaussian["mask_aware"], "noise_sigma": noise,
                "threshold": cfg_gaussian["noise_threshold"] if mode == "conditional" else None, "reason": ""}
    if mode == "off":
        decision["reason"] = "mode_off"
    elif mode == "fixed":
        decision.update(apply=True, reason="fixed_mode")
    elif not analysis_sufficient:
        decision["reason"] = "insufficient_analysis_pixels"
    elif noise is None:
        decision["reason"] = "metric_unavailable"
    elif noise >= cfg_gaussian["noise_threshold"]:
        decision.update(apply=True, reason="noise_sigma_at_or_above_threshold")
    else:
        decision["reason"] = "noise_sigma_below_threshold"
    return decision
