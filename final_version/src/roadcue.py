"""노면 단서. 후보 주변이 그 사진의 도로와 얼마나 다른지 잰다.

후보 상자를 약 2배로 넓힌 창에서 상자를 뺀 고리를, 사진의 도로 표본(높이 60~80%, 너비 30~70%)과 비교한다.
값이 클수록 도로답지 않다.

  color: HSV 색상-채도 좌표의 중앙값 거리를 표본 퍼짐으로 나눈 것
  energy: 구조 텐서 l1 + l2 중앙값이 표본과 몇 배 다른지 (log 절댓값). 너무 매끈하거나 너무 센 곳
  coherence: 결 정렬도의 고리 평균 - 표본 평균
  depth: 고리 밝기 중앙값과 후보 밝기 평균의 차를 고리 사분위 범위로 나눈 것. 작으면 얕은 골이라 질감일 가능성이 크다
"""
import cv2
import numpy as np

SIGMA = 4.0


def maps(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ang = hsv[..., 0] * (np.pi / 90.0)                       # OpenCV 색상은 0~180
    sat = hsv[..., 1] / 255.0
    cx, cy = sat * np.cos(ang), sat * np.sin(ang)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    a = cv2.GaussianBlur(ix * ix, (0, 0), SIGMA)
    b = cv2.GaussianBlur(ix * iy, (0, 0), SIGMA)
    c = cv2.GaussianBlur(iy * iy, (0, 0), SIGMA)
    root = np.sqrt((a - c) ** 2 + 4 * b * b)
    l1, l2 = (a + c + root) / 2, np.maximum((a + c - root) / 2, 0)
    energy = l1 + l2
    coh = np.where(energy > 1e-6, ((l1 - l2) / np.maximum(energy, 1e-6)) ** 2, 0)
    return cx, cy, energy, coh


def ref_stats(m, H, W):
    r = (slice(int(.60 * H), int(.80 * H)), slice(int(.30 * W), int(.70 * W)))
    cx, cy, e, coh = (x[r] for x in m)
    pts = np.stack([cx.ravel(), cy.ravel()], 1)
    med = np.median(pts, 0)
    spread = max(float(np.median(np.linalg.norm(pts - med, axis=1))), 0.02)
    return dict(cmed=med, cspread=spread, emed=max(float(np.median(e)), 1e-3), cohm=float(coh.mean()))


def ring(b, W, H):
    x1, y1, x2, y2 = b
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) + 6, (y2 - y1) + 6
    X1, Y1 = max(0, int(cx - hw)), max(0, int(cy - hh))
    X2, Y2 = min(W, int(cx + hw)), min(H, int(cy + hh))
    m = np.ones((Y2 - Y1, X2 - X1), bool)
    m[max(0, y1 - 2 - Y1):max(0, y2 + 2 - Y1), max(0, x1 - 2 - X1):max(0, x2 + 2 - X1)] = False
    return (slice(Y1, Y2), slice(X1, X2)), m


def cue_values(m, st, b, W, H):
    sl, keep = ring(b, W, H)
    if keep.sum() < 20:
        return None
    cx, cy, e, coh = (x[sl][keep] for x in m)
    pts = np.stack([cx, cy], 1)
    return {"color": float(np.linalg.norm(np.median(pts, 0) - st["cmed"]) / st["cspread"]),
            "energy": float(abs(np.log(max(float(np.median(e)), 1e-3) / st["emed"]))),
            "coherence": float(coh.mean() - st["cohm"])}


def onroad_depth(pts, smooth, b, W, H):
    sl, keep = ring(b, W, H)
    rg = smooth[sl][keep]
    q1, q3 = np.percentile(rg, [25, 75])
    return float((np.median(rg) - smooth[pts[:, 1], pts[:, 0]].mean()) / max(q3 - q1, 2.0))
