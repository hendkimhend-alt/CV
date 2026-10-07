"""
노면 단서 — 후보 주변이 그 사진의 도로 표본과 얼마나 다른가 (거르기: 노면 밖 후보 버리기).
사진을 자르지 않고, 이미 나온 후보마다 "주변 고리"(상자를 2배로 넓힌 창 − 후보 상자)를 도로 표본과 비교한다.
도로 표본 = ROI auto와 같은 자리 (높이 60~80% · 너비 30~70%). 값이 클수록 도로답지 않음.
  color      HSV 색상 · 채도 평면 좌표의 중앙값 거리 ÷ 표본 퍼짐                 (수업 2-1 컬러 모델)
  bright     |명도 V 중앙값 차| ÷ 표본 사분위 범위                               (2-1 히스토그램)
  energy     |log(구조 텐서 λ1+λ2 중앙값 비)| — 너무 매끈 · 너무 강함 둘 다        (3-1 그레이디언트 · 3-2 2차 모멘트 행렬)
  coherence  정렬도 ((λ1−λ2)/(λ1+λ2))² 고리 평균 − 표본 평균                      (3-2)
  corner     log(λ2 중앙값 비) — 모서리                                           (3-2 Shi-Tomasi · 해리스)
근거: 거르기 R1 (개발 세트) — 세기 · 정렬 · 색 세 단서로 진짜 14% 손실에 가짜 70% 버림
"""
import cv2
import numpy as np

SIGMA = 4.0


def maps(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ang = hsv[..., 0] * (np.pi / 90.0)                       # OpenCV 색상 0~180 → 라디안
    sat = hsv[..., 1] / 255.0
    cx, cy, v = sat * np.cos(ang), sat * np.sin(ang), hsv[..., 2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ix, iy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    a = cv2.GaussianBlur(ix * ix, (0, 0), SIGMA)
    b = cv2.GaussianBlur(ix * iy, (0, 0), SIGMA)
    c = cv2.GaussianBlur(iy * iy, (0, 0), SIGMA)
    root = np.sqrt((a - c) ** 2 + 4 * b * b)
    l1, l2 = (a + c + root) / 2, np.maximum((a + c - root) / 2, 0)
    energy = l1 + l2
    coh = np.where(energy > 1e-6, ((l1 - l2) / np.maximum(energy, 1e-6)) ** 2, 0)
    return cx, cy, v, energy, coh, l2


def ref_stats(m, H, W):
    r = (slice(int(.60 * H), int(.80 * H)), slice(int(.30 * W), int(.70 * W)))
    cx, cy, v, e, coh, l2 = (x[r] for x in m)
    pts = np.stack([cx.ravel(), cy.ravel()], 1)
    med = np.median(pts, 0)
    spread = max(float(np.median(np.linalg.norm(pts - med, axis=1))), 0.02)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(cmed=med, cspread=spread, vmed=float(np.median(v)), viqr=max(float(q3 - q1), 5.0),
                emed=max(float(np.median(e)), 1e-3), cohm=float(coh.mean()), l2med=max(float(np.median(l2)), 1e-3))


def ring(b, W, H):
    x1, y1, x2, y2 = b
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) + 6, (y2 - y1) + 6
    X1, Y1, X2, Y2 = max(0, int(cx - hw)), max(0, int(cy - hh)), min(W, int(cx + hw)), min(H, int(cy + hh))
    m = np.ones((Y2 - Y1, X2 - X1), bool)
    m[max(0, y1 - 2 - Y1):max(0, y2 + 2 - Y1), max(0, x1 - 2 - X1):max(0, x2 + 2 - X1)] = False
    return (slice(Y1, Y2), slice(X1, X2)), m


def cue_values(m, st, b, W, H):
    """b = (x1, y1, x2, y2) → 단서 값 사전 (고리가 너무 작으면 None — 버리지 않음)."""
    sl, keep = ring(b, W, H)
    if keep.sum() < 20:
        return None
    cx, cy, v, e, coh, l2 = (x[sl][keep] for x in m)
    pts = np.stack([cx, cy], 1)
    return {"color": float(np.linalg.norm(np.median(pts, 0) - st["cmed"]) / st["cspread"]),
            "bright": float(abs(np.median(v) - st["vmed"]) / st["viqr"]),
            "energy": float(abs(np.log(max(float(np.median(e)), 1e-3) / st["emed"]))),
            "coherence": float(coh.mean() - st["cohm"]),
            "corner": float(np.log(max(float(np.median(l2)), 1e-3) / st["l2med"]))}


def line_resid(pts):
    """후보 픽셀이 직선에서 벗어난 정도 (직선 맞춤 잔차 RMS, px) — 작을수록 곧음 (연석 · 차선 테두리)."""
    c = pts.astype(np.float64) - pts.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    return float(sv[-1] / np.sqrt(len(pts)))
