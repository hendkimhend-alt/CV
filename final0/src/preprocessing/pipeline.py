"""영상 한 장 전처리와 결과 저장.

status: success / manual_required (자동 마스크 FAIL + 수동 마스크 없음) / error
processed_image는 road_mask 밖 화소가 입력과 같아야 한다.
"""
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .analysis_mask import make_analysis_mask
from .config import config_hash
from .gamma import apply_gamma, choose_gamma, decide_gamma, guard_triggered
from .gaussian import decide_gaussian, masked_gaussian
from .image_io import to_binary_mask, validate_image, validate_mask, write_image
from .mask_validation import PASS, compute_mask_metrics, validate_road_mask
from .quality import METRIC_NAMES, measure_in_masks
from .road_mask import METHOD, extract_road_mask

SUCCESS, MANUAL_REQUIRED, ERROR = "success", "manual_required", "error"


@dataclass
class PreprocessResult:
    image_id: str
    status: str
    metadata: dict
    processed_image: np.ndarray = None
    road_mask: np.ndarray = None
    analysis_mask: np.ndarray = None
    auto_mask: np.ndarray = None
    seeds: list = field(default_factory=list)


def _round(value, digits=6):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


def _changes(initial, final):
    out = {}
    for name in METRIC_NAMES:
        a, b = initial.get(name), final.get(name)
        out[name] = None if a is None or b is None else b - a
    return out


def base_metadata(image_id, image, cfg, input_path=None):
    return {
        "image_id": image_id,
        "input_path": str(input_path) if input_path else None,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "config_version": cfg["config_version"],
        "config_sha256": config_hash(cfg),
        "status": None,
        "roi": {}, "validation": {}, "manual_correction": {}, "final_mask": {},
        "analysis_mask": {}, "quality": {}, "gamma": {}, "noise_remeasurement": {}, "gaussian": {},
        "timing_ms": {}, "outputs": {}, "warnings": [], "errors": [],
        "versions": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
    }


def _elapsed(t0):
    return (time.perf_counter() - t0) * 1000


def _choose_final_mask(image, auto, passed, cfg, meta, manual_mask, manual_source, editor, review, title):
    """수동 마스크 파일 → 편집기 → (PASS면) 자동 마스크 순서. 해당 없으면 None."""
    manual = {"required": not passed, "applied": False, "source": None, "path": None}
    final_mask = None
    if manual_mask is not None:
        if cfg["manual_correction"]["allow_external_mask"]:
            validate_mask(manual_mask, image.shape)
            final_mask = manual_mask.copy()
            manual.update(applied=True, source="external_file",
                          path=str(manual_source) if manual_source else None)
        else:
            meta["warnings"].append("external_mask_ignored:allow_external_mask=false")
    if final_mask is None and editor is not None and (not passed or review):
        edited = editor(image, auto.mask, title)
        if edited is not None:
            final_mask = to_binary_mask(edited)
            manual.update(applied=True, source="interactive_editor")
    if final_mask is None and passed:
        final_mask = auto.mask
    meta["manual_correction"] = manual
    return final_mask


def _apply_gamma_step(image, final_mask, initial, cfg_gamma, sufficient, measure, meta):
    """→ (Gamma 후 영상, Gamma 후 품질)."""
    decision = decide_gamma(initial, cfg_gamma, sufficient)
    current, after, guard, tried = image, None, False, None
    if decision["apply"] and cfg_gamma["selection"] == "adaptive":
        value, img, after, tried = choose_gamma(image, final_mask, initial, cfg_gamma, measure)
        if value is None:
            guard = True
        else:
            decision["value"], current = value, img
    elif decision["apply"]:
        current = apply_gamma(image, decision["value"], final_mask)

    if after is None:
        after = initial if current is image else measure(current)
    if decision["apply"] and not guard and guard_triggered(initial, after, cfg_gamma):
        guard = True
    if guard:
        current, after = image, initial

    applied = decision["apply"] and not guard
    meta["gamma"] = {**decision, "selection": cfg_gamma["selection"], "applied": applied,
                     "value_used": decision["value"] if applied else 1.0, "guard_triggered": guard,
                     "channel": "BGR", "region": "road_mask"}
    if tried is not None:
        meta["gamma"]["tried"] = tried
    if guard:
        meta["gamma"]["reason"] = "reverted_by_saturation_guard"
    return current, after


def process_image(image, cfg, *, image_id="image", input_path=None, manual_mask=None, manual_source=None,
                  editor=None, review=False):
    """manual_mask가 있으면 자동 검증 결과와 관계없이 그것을 쓴다.
    editor(image, mask, title)는 FAIL일 때(review면 PASS도) 호출된다.
    """
    t_total = time.perf_counter()
    meta = base_metadata(image_id, image, cfg, input_path)
    result = PreprocessResult(image_id=image_id, status=ERROR, metadata=meta)
    stage = "validate_input"
    try:
        validate_image(image)

        # 1. 자동 Road Mask
        stage = "road_mask"
        t0 = time.perf_counter()
        auto = extract_road_mask(image, cfg["roi"])
        meta["timing_ms"]["road_mask"] = _elapsed(t0)
        result.auto_mask, result.seeds = auto.mask, auto.seeds
        meta["roi"] = {"method": METHOD, "work_scale": auto.work_scale, "work_size": list(auto.work_size),
                       "grid_shape": list(auto.grid_shape), "n_seeds": len(auto.seeds),
                       "seeds": auto.seeds, "stage_timing_ms": auto.timings_ms}
        meta["warnings"].extend(auto.warnings)

        # 2. 검증
        stage = "validation"
        t0 = time.perf_counter()
        metrics = compute_mask_metrics(auto.mask, [(s["x"], s["y"]) for s in auto.seeds])
        validation = validate_road_mask(metrics, cfg["validation"])
        meta["timing_ms"]["validation"] = _elapsed(t0)
        meta["validation"] = {"status": validation["status"], "metrics": metrics,
                              "checks": validation["checks"], "fail_reasons": validation["fail_reasons"]}
        meta["warnings"].extend(validation["warnings"])

        # 3. 최종 road_mask
        stage = "manual_correction"
        passed = validation["status"] == PASS
        title = f"{image_id} — 자동 검증 {validation['status']}"
        final_mask = _choose_final_mask(image, auto, passed, cfg, meta, manual_mask, manual_source,
                                        editor, review, title)
        if final_mask is None:
            meta["status"] = result.status = MANUAL_REQUIRED
            meta["final_mask"] = {"source": None}
            return _finish(result, t_total)
        road_pixels = int(np.count_nonzero(final_mask))
        if not road_pixels:
            raise ValueError("최종 road_mask가 비어 있음")
        result.road_mask = final_mask
        meta["final_mask"] = {"source": "manual" if meta["manual_correction"]["applied"] else "auto",
                              "road_pixels": road_pixels, "area_ratio": road_pixels / final_mask.size}

        # 4. analysis_mask
        stage = "analysis_mask"
        t0 = time.perf_counter()
        analysis, analysis_info = make_analysis_mask(final_mask, cfg["analysis_mask"])
        meta["timing_ms"]["analysis_mask"] = _elapsed(t0)
        result.analysis_mask = analysis
        meta["analysis_mask"] = analysis_info
        sufficient = analysis_info["sufficient"]
        if not sufficient:
            meta["warnings"].append("analysis_mask_too_small:conditional_corrections_skipped")

        def measure(img):
            if not sufficient:
                return {name: None for name in METRIC_NAMES}
            return measure_in_masks(img, analysis, final_mask, cfg["quality"])[0]

        # 5. 초기 품질
        stage = "quality_initial"
        t0 = time.perf_counter()
        if sufficient:
            initial, measure_info = measure_in_masks(image, analysis, final_mask, cfg["quality"])
        else:
            initial, measure_info = measure(image), None
        meta["timing_ms"]["quality_initial"] = _elapsed(t0)

        # 6. 조건부 Gamma (+ 노이즈 재측정)
        stage = "gamma"
        t0 = time.perf_counter()
        current, after_gamma = _apply_gamma_step(image, final_mask, initial, cfg["gamma"], sufficient,
                                                 measure, meta)
        meta["timing_ms"]["gamma"] = _elapsed(t0)
        meta["noise_remeasurement"] = {"noise_sigma": after_gamma.get("noise_sigma"),
                                       "laplacian_variance": after_gamma.get("laplacian_variance"),
                                       "image": "after_gamma" if meta["gamma"]["applied"] else "unchanged_input"}

        # 7. 조건부 Gaussian
        stage = "gaussian"
        t0 = time.perf_counter()
        gauss = decide_gaussian(after_gamma, cfg["gaussian"], sufficient)
        if gauss["apply"]:
            current = masked_gaussian(current, final_mask, gauss["kernel"], gauss["sigma"], gauss["mask_aware"])
        meta["timing_ms"]["gaussian"] = _elapsed(t0)
        meta["gaussian"] = {**gauss, "applied": gauss["apply"], "region": "road_mask",
                            "sigma_used": gauss["sigma"] if gauss["apply"] else 0,
                            "kernel_used": gauss["kernel"] if gauss["apply"] else None}

        # 8. 최종 품질
        stage = "quality_final"
        t0 = time.perf_counter()
        final = measure(current)
        meta["timing_ms"]["quality_final"] = _elapsed(t0)
        meta["quality"] = {"measurement": measure_info, "region": "analysis_mask",
                           "initial": initial, "after_gamma": after_gamma, "final": final,
                           "change_final_minus_initial": _changes(initial, final)}

        # 9. 출력 확인: 크기 유지, 도로 밖 화소 불변
        stage = "output_check"
        processed = current.copy() if current is image else current
        if processed.shape != image.shape or final_mask.shape != image.shape[:2]:
            raise RuntimeError("출력 크기가 입력과 다름")
        outside = final_mask == 0
        if not np.array_equal(processed[outside], image[outside]):
            raise RuntimeError("road_mask 밖 화소가 바뀜")
        result.processed_image = processed
        meta["status"] = result.status = SUCCESS
    except Exception as exc:  # 한 장이 실패해도 배치는 계속
        meta["status"] = result.status = ERROR
        meta["errors"].append({"stage": stage, "type": type(exc).__name__, "message": str(exc)})
    return _finish(result, t_total)


def _finish(result, t_total):
    result.metadata["timing_ms"]["total"] = _elapsed(t_total)
    result.metadata = _round(result.metadata)
    return result


def make_overlay(image, mask, analysis_mask=None, seeds=(), text="", max_side=1024):
    """검토용 그림: 도로(초록) · analysis_mask 외곽(파랑) · 시드(빨강, 못 자란 시드는 자홍)."""
    scale = min(1.0, max_side / max(image.shape[:2]))
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    vis = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    if mask is not None:
        road = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0
        tint = vis.copy()
        tint[road] = (0, 200, 0)
        vis = cv2.addWeighted(vis, 0.6, tint, 0.4, 0)
        contours, _ = cv2.findContours(road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)
    if analysis_mask is not None:
        inner = cv2.resize(analysis_mask, size, interpolation=cv2.INTER_NEAREST) > 0
        contours, _ = cv2.findContours(inner.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, (255, 128, 0), 1)
    for seed in seeds:
        center = (int(round(seed["x"] * scale)), int(round(seed["y"] * scale)))
        color = (0, 0, 255) if seed.get("grown", True) else (255, 0, 255)
        cv2.circle(vis, center, 6, color, -1)
        cv2.circle(vis, center, 6, (255, 255, 255), 1)
    for i, line in enumerate(text.split("\n") if text else []):
        y = 20 + 20 * i
        cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def overlay_text(result):
    meta = result.metadata
    lines = [f"{result.image_id}  {result.status}  auto={meta.get('validation', {}).get('status', '-')}"]
    if result.status == SUCCESS:
        lines.append(f"mask={meta['final_mask']['source']}  gamma={meta['gamma']['value_used']}  "
                     f"gauss={meta['gaussian']['sigma_used']}")
    reasons = meta.get("validation", {}).get("fail_reasons") or []
    if reasons:
        lines.append("FAIL: " + "; ".join(reasons)[:90])
    return "\n".join(lines)


def save_result(result, image_dir, cfg_output, original=None):
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if result.status == SUCCESS:
        write_image(image_dir / "processed_image.png", result.processed_image)
        write_image(image_dir / "road_mask.png", result.road_mask)
        outputs.update(processed_image="processed_image.png", road_mask="road_mask.png")
        if cfg_output["save_analysis_mask"]:
            write_image(image_dir / "analysis_mask.png", result.analysis_mask)
            outputs["analysis_mask"] = "analysis_mask.png"
    elif result.auto_mask is not None:
        write_image(image_dir / "road_mask_candidate.png", result.auto_mask)
        outputs["road_mask_candidate"] = "road_mask_candidate.png"

    if cfg_output["save_debug"] and original is not None:
        mask = result.road_mask if result.road_mask is not None else result.auto_mask
        vis = make_overlay(original, mask, result.analysis_mask, result.seeds, overlay_text(result))
        write_image(image_dir / "overlay.jpg", vis)
        outputs["overlay"] = "overlay.jpg"

    if cfg_output["save_metadata"]:
        outputs["metadata"] = "metadata.json"
        result.metadata["outputs"] = outputs
        text = json.dumps(result.metadata, ensure_ascii=False, indent=2, allow_nan=False)
        (image_dir / "metadata.json").write_text(text, encoding="utf-8")
    else:
        result.metadata["outputs"] = outputs
    return outputs
