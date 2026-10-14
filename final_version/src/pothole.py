"""포트홀 검출.

detect_potholes(img, mask, cfg)에 검출기 입력(BGR, 긴 변 1024)과 노면 마스크를 넣으면 포트홀 박스 목록을 돌려준다.

개발 세트 포트홀을 보면 밝기는 주변 노면과 거의 같고, 대신 안쪽이 거칠고(깨진 골재) 일부만 어둡다(패인 곳 그림자).
그래서 어두운 곳이 아니라 '주변 노면보다 거친 곳'을 후보로 찾는다. 거칠기는 기울기 크기를 k x k로 평균한 값이고,
주변 고리(노면 픽셀만)의 평균 거칠기보다 ring_ratio배 넘게 거친 영역이 후보가 된다.
같은 줄 전체와 비교해 봤더니 노면 밖이나 먼 곳 질감이 섞여서 바로 주변하고만 비교한다.

후보는 다섯 가지를 모두 통과해야 포트홀로 남긴다.
  - 색이 주변과 비슷할 것 (차, 간판 제외)
  - 안쪽 어두운 부분이 주변보다 어둡거나 같을 것 (패인 곳 그림자)
  - 가늘고 길지 않을 것 (균열, 차선 제외)
  - 화면 위쪽이 아닐 것 (먼 곳, 노면 밖 제외)
  - 주변 색이 그 사진의 도로 표본과 비슷할 것 (roadcue.py)
"""
import cv2
import numpy as np

from roadcue import cue_values, maps, ref_stats

RINGS = ((41, 121), (121, 361))     # (안쪽 창, 바깥 창). 고리 = 바깥 - 안쪽
MIN_RING_COVERAGE = 0.3             # 고리 안에 노면이 이것보다 적으면 비교 안 함


def _window_sum(a, n):
    return cv2.boxFilter(a, -1, (n, n), normalize=False, borderType=cv2.BORDER_CONSTANT)


def ring_ratio(gray, mask, k):
    """픽셀마다 주변 노면 고리보다 몇 배 거친지. 노면 밖은 0."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    E = cv2.blur(grad, (k, k)).astype(np.float64)
    M = (mask > 0).astype(np.float64)
    EM, out = E * M, np.zeros_like(E)
    for a, b in RINGS:
        S = _window_sum(EM, b) - _window_sum(EM, a)      # 고리 안 노면의 거칠기 합
        N = _window_sum(M, b) - _window_sum(M, a)        # 고리 안 노면 픽셀 수
        r = np.where(N >= MIN_RING_COVERAGE * (b * b - a * a), E / (S / np.maximum(N, 1) + 1e-3), 0)
        out = np.maximum(out, r)
    return out * M


def _box_cues(gray, chrom, box):
    """상자 안과 주변 고리(상자를 사방으로 절반씩 넓힌 창에서 상자를 뺀 부분)를 비교한 값.

    chroma: 색도 중앙값 차 * 100
    dark_p10: (고리 하위 10% 밝기 - 안쪽 하위 10% 밝기) / 고리 사분위 범위. 양수면 안쪽이 더 어둡다
    상자나 고리가 너무 작으면 None을 돌려주고, 그 후보는 버린다.
    """
    H, W = gray.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    rx1, ry1 = max(0, x1 - bw // 2), max(0, y1 - bh // 2)
    rx2, ry2 = min(W, x2 + bw // 2), min(H, y2 + bh // 2)
    ring = np.ones((ry2 - ry1, rx2 - rx1), bool)
    ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
    if bw < 4 or bh < 4 or ring.sum() < 20:
        return None
    gi, gr = gray[y1:y2, x1:x2].ravel(), gray[ry1:ry2, rx1:rx2][ring]
    q1, q3 = np.percentile(gr, [25, 75])
    dark_p10 = (np.percentile(gr, 10) - np.percentile(gi, 10)) / max(q3 - q1, 1.0)
    ci = np.median(chrom[y1:y2, x1:x2].reshape(-1, 2), 0)
    cr = np.median(chrom[ry1:ry2, rx1:rx2][ring], 0)
    return {"chroma": float(np.linalg.norm(ci - cr) * 100), "dark_p10": float(dark_p10)}


def detect_potholes(img, mask, cfg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    s = img.astype(np.float32).sum(2) + 1
    chrom = np.stack([img[..., 2] / s, img[..., 1] / s], 2)    # 정규화 RGB. 그림자에 덜 흔들린다
    cue_maps = maps(img)
    ref = ref_stats(cue_maps, H, W)

    m = (ring_ratio(gray, mask, cfg["tex_k"]) > cfg["ring_ratio"]).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if not (cfg["min_frac"] * H * W <= a <= cfg["max_frac"] * H * W):
            continue
        cues = _box_cues(gray, chrom, (x, y, x + w, y + h))
        if cues is None:
            continue

        ok = (cues["chroma"] <= cfg["chroma_max"]
              and cues["dark_p10"] >= cfg["darkp10_min"]
              and max(w, h) / max(1, min(w, h)) <= cfg["aspect_max"]
              and (y + h / 2) / H >= cfg["min_rel_y"])
        if ok:
            cv = cue_values(cue_maps, ref, (x, y, x + w, y + h), W, H)
            # 고리가 작아서 못 재면 버리지 않는다
            ok = cv is None or cv["color"] <= cfg["road_color_max"]
        if ok:
            out.append({"type": "pothole", "bbox": (x, y, w, h)})
    return out
