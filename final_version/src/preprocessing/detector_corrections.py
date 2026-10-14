"""검출 직전 보정. 잘라서 크기를 맞춘 검출기 입력에, 노면 마스크 안에만 건다.

선명한 사진은 가우시안과 막대 닫힘으로 잔 질감을 줄인 뒤 CLAHE로 대비를 올리고,
아주 흐린 사진(노면 선명도 < blur_thr)은 질감 처리를 건너뛰고 언샤프 후 CLAHE만 한다.
어느 단계를 어떤 사진에 걸지는 configs/detection.json "corrections"의 apply_to로 정한다
(없으면 모든 사진, blur_rt면 흐린 사진만, not_blur_rt면 나머지만).
"""
import cv2
import numpy as np


def line_kernel(length, angle_deg):
    """angle_deg 방향, 길이 length의 1px 막대 커널."""
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2
    dx, dy = np.cos(np.radians(angle_deg)) * c, -np.sin(np.radians(angle_deg)) * c
    cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return k


def road_sharpness(img, mask, geo):
    """노면 안 Laplacian 분산. 자른 영상이 아니라 원본 사진 긴 변을 1024로 맞춘 척도로 잰다."""
    f = (1024.0 / max(geo["original_width"], geo["original_height"])) / geo["scale_x"]
    size = (max(1, round(img.shape[1] * f)), max(1, round(img.shape[0] * f)))
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), size, interpolation=interp)
    road = (cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    m = cv2.erode(road, np.ones((5, 5), np.uint8)) > 0
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=1)
    return float(lap[m].var()) if m.sum() > 100 else float(lap.var())


def is_blurry(img, mask, geo, corr):
    """노면 선명도가 blur_thr보다 낮으면 아주 흐린 사진으로 본다."""
    thr = next(c["blur_thr"] for c in corr.values() if isinstance(c, dict) and "blur_thr" in c)
    return road_sharpness(img, mask, geo) < thr


def apply_corrections(img, mask, corr, blurry):
    """흐림 여부에 맞는 보정을 order 순서로 걸고, 노면 안 화소만 바꾼다."""
    tag = "blur_rt" if blurry else "not_blur_rt"
    out = img
    for step in corr["order"]:
        c = corr[step]
        if "apply_to" in c and c["apply_to"] != tag:
            continue
        if step == "clahe":
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=c["clip"], tileGridSize=(c["tile"],) * 2)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif step == "gaussian":
            k = (c["ksize"], c["ksize"])
            out = cv2.GaussianBlur(out, k, c["sigma"], borderType=cv2.BORDER_REFLECT_101)
        elif step == "line_close":
            # 여러 방향 막대로 닫힘 후 최솟값: 가는 어두운 선은 어느 방향이든 남고 잔 질감만 메워진다
            closed = []
            for i in range(c["angles"]):
                kernel = line_kernel(c["length"], 180.0 * i / c["angles"])
                closed.append(cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel,
                                               borderType=cv2.BORDER_REPLICATE))
            out = np.min(np.stack(closed), axis=0)
        elif step == "unsharp":
            src = out.astype(np.float32)
            blur = cv2.GaussianBlur(src, (0, 0), c["sigma"], borderType=cv2.BORDER_REFLECT_101)
            out = np.rint(np.clip(src + c["amount"] * (src - blur), 0, 255)).astype(np.uint8)
        else:
            raise ValueError(f"corrections: 모르는 단계 {step}")
    return np.where((mask > 0)[..., None], out, img).astype(np.uint8)
