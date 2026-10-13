"""검출기 입력 보정 — 연결(노면 영역 자르기 · 긴 변 1024) 뒤, 검출기 바로 앞에서 노면 마스크 안에만 거는 보정.

균열 최종 구성 (10/11 ~ 12, 실험 기록: 저장소 results/experiments.md · EVALUATION.md)
  선명한 사진:   가우시안 σ0.8 → 막대 닫힘 7px (12방향 최솟값) → CLAHE 1.0
  아주 흐린 사진: (질감 처리 건너뜀) → 언샤프 3.0 · σ3 → CLAHE 1.0
  "아주 흐림" = 노면 마스크 안 선명도(Laplacian 분산, 사진 긴 변 1024 척도) < 2.76

설정 (검출 설정 파일의 "corrections", 또는 analysis/run_road_pipeline.py의 --extra · configs/final*.json의 "extra")
  {"gaussian": {"ksize": 3, "sigma": 0.8}, "line_close": {"length": 7, "angles": 12},
   "unsharp": {"amount": 3.0, "sigma": 3.0}, "clahe": {"clip": 1.0, "tile": 8},
   "order": [단계 순서]}
  단계마다 "apply_to": "all"(기본) · "blur_rt"(아주 흐린 사진만) · "not_blur_rt"(그 밖) · 그룹 이름
  흐림 판단 기준은 단계 중 하나의 "blur_thr" (기본 50)
"""
import cv2
import numpy as np

from detect import _line_kernel

STEPS = ("clahe", "gaussian", "line_close", "unsharp")     # 사진마다 감마 · Harris 거친 곳(실험 탈락)은 10/13 정리
DEFAULT_ORDER = ("clahe", "gaussian", "line_close", "unsharp")


def validate(corr):
    """보정 설정 확인 — 모르는 단계 · order에 없는 단계는 실행 전에 오류."""
    unknown = set(corr) - set(STEPS) - {"order"}
    if unknown:
        raise ValueError(f"corrections: 모르는 단계 {sorted(unknown)} (쓸 수 있는 것: {list(STEPS)})")
    for step in corr.get("order", ()):
        if step not in STEPS:
            raise ValueError(f"corrections.order: 모르는 단계 {step}")
    return corr


def road_sharpness(img, mask, geo):
    """노면 마스크(5×5로 깎음) 안 회색의 Laplacian 분산 — 사진 전체 긴 변을 1024로 맞춘 척도 (품질 측정 · 1차 흐림 분류와 같은 척도).
    img · mask = 검출기 입력 (to_detector_input 결과), geo = 그 geometry."""
    f = (1024.0 / max(geo["original_width"], geo["original_height"])) / geo["scale_x"]
    size = (max(1, round(img.shape[1] * f)), max(1, round(img.shape[0] * f)))
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    m = cv2.erode((cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=1)
    return float(lap[m].var()) if m.sum() > 100 else float(lap.var())


def judge_tags(img, mask, geo, corr, base_tags=()):
    """사진 태그 + 파이프라인이 직접 판단한 흐림 태그 (blur_rt: 노면 선명도 < blur_thr / 아니면 not_blur_rt)."""
    tags = set(base_tags)
    steps = [c for c in corr.values() if isinstance(c, dict) and c.get("apply_to") in ("blur_rt", "not_blur_rt")]
    if steps:
        thr = next((c["blur_thr"] for c in steps if "blur_thr" in c), 50.0)
        tags.add("blur_rt" if road_sharpness(img, mask, geo) < thr else "not_blur_rt")
    return tags


def apply_corrections(img, mask, corr, tags):
    """보정을 corr["order"] 순서로, 노면 마스크 안에만. → (영상, 적용했나)"""
    out, applied = img, False
    for step in corr.get("order", DEFAULT_ORDER):
        if step not in corr:
            continue
        c = corr[step]
        if c.get("apply_to", "all") != "all" and c["apply_to"] not in tags:
            continue
        if step == "clahe":
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=c["clip"], tileGridSize=(c.get("tile", 8),) * 2).apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif step == "gaussian":
            out = cv2.GaussianBlur(out, (c["ksize"], c["ksize"]), c["sigma"], borderType=cv2.BORDER_REFLECT_101)
        elif step == "line_close":
            closed = [cv2.morphologyEx(out, cv2.MORPH_CLOSE, _line_kernel(c["length"], 180.0 * i / c["angles"]),
                                       borderType=cv2.BORDER_REPLICATE) for i in range(c["angles"])]
            out = np.min(np.stack(closed), axis=0)
        elif step == "unsharp":
            src = out.astype(np.float32)
            blur = cv2.GaussianBlur(src, (0, 0), c["sigma"], borderType=cv2.BORDER_REFLECT_101)
            out = np.rint(np.clip(src + c["amount"] * (src - blur), 0, 255)).astype(np.uint8)
        else:
            raise ValueError(f"corrections: 모르는 단계 {step}")
        applied = True
    if not applied:
        return img, False
    return np.where((mask > 0)[..., None], out, img).astype(np.uint8), True
