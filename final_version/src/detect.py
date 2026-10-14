"""균열 검출.

detect_cracks(img, cfg)에 검출기 입력(BGR, 긴 변 1024)을 넣으면 균열 박스 목록을 돌려준다.
Hessian 선 점수로 어두운 선을 찾고 (아주 흐린 사진은 pipeline.py가 더 큰 sigma를 넘긴다),
텐서 전파로 끊긴 조각을 이은 뒤 이중 임계값으로 흔적을 남긴다. 덩어리마다 길이와 모양을 본 다음
양옆이 밝은 골짜기인지, 주변이 도로처럼 보이는지(roadcue.py)로 한 번 더 거른다.
"""
import cv2
import numpy as np

from roadcue import cue_values, maps, onroad_depth, ref_stats


def detect_cracks(img, cfg):
    """균열 박스 목록을 돌려준다. 흔적을 찾은 뒤 덩어리마다 모양, 위 테두리, 양쪽 확인, 노면 단서 · 깊이로 거른다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    trace = line_trace(gray, cfg)

    smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    cue_maps = maps(img)
    ref = ref_stats(cue_maps, H, W)

    detections = []
    for pts in _components(trace):
        det = shape_features(pts)
        if not is_crack_shape(det, cfg) or _touches_border(det["bbox"], W, H, cfg):
            continue
        if valley_score(smooth, pts, det["width"], cfg) < cfg["valley_min_ratio"]:
            continue
        x, y, w, h = det["bbox"]
        box = (x, y, x + w, y + h)
        # 고리가 너무 작아서 못 재면 그 단서들은 건너뛴다
        cues = cue_values(cue_maps, ref, box, W, H) or {}
        cues["depth"] = onroad_depth(pts, smooth, box, W, H)
        if _rejected_by_cues(cues, cfg["cue_filter"]):
            continue
        detections.append({"type": "crack", "bbox": det["bbox"]})
    return detections


def _rejected_by_cues(cues, limits):
    """노면 단서 중 하나라도 기준을 넘으면 True."""
    # 키가 _min으로 끝나면 하한, 나머지는 상한
    for k, t in limits.items():
        if k.endswith("_min"):
            if cues[k[:-4]] < t:
                return True
        elif k in cues and cues[k] > t:
            return True
    return False


# ---------- 찾기 (Hessian) ----------

def line_score(gray, sigmas):
    """Hessian 선 점수. 밝기를 지형처럼 보고, 한 방향으로만 오목하게 파인 정도를 잰다.

    Hessian의 고윳값을 l1 >= l2라고 하면 어두운 선은 가로지르는 방향으로 크게 오목하고(l1이 큼)
    따라가는 방향으로는 평평하다(|l2|가 작음). 그래서 점수는 l1 - |l2|.
    골재 틈새 같은 점은 사방이 오목해서 l1, l2가 비슷하므로 점수가 낮게 나온다.

    균열 폭이 제각각이라 sigma 여러 개로 구해서 평균한다 (sigma^2를 곱해 크기를 맞춤).
    방향은 평균에 가장 가까운 sigma의 것을 쓰고, 선을 가로지르는 방향(라디안)이다.
    """
    g = gray.astype(np.float32)
    scores, angles = [], []
    for s in sigmas:
        b = cv2.GaussianBlur(g, (0, 0), s)
        xx = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3)
        yy = cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3)
        xy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3)
        root = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2)
        l1, l2 = (xx + yy + root) / 2, (xx + yy - root) / 2
        scores.append(np.clip(l1 - np.abs(l2), 0, None) * s * s)
        angles.append(0.5 * np.arctan2(2 * xy, xx - yy))
    S, A = np.stack(scores), np.stack(angles)
    best = S.mean(0)
    pick = np.abs(S - best).argmin(0)
    angle = np.take_along_axis(A, pick[None], 0)[0]
    return best.astype(np.float32), angle.astype(np.float32)


def line_centerline(score, angle):
    """선 점수에서 선을 가로지르는 방향으로 가장 큰 점만 남긴다 (중심선)."""
    # Canny의 non-maximum suppression과 같은 방식 (4방향)
    # 선을 가로지르는 방향을 4개로 나눔 (그 방향의 양옆 이웃과 비교): 0 가로, 1 \ 대각선, 2 세로, 3 / 대각선
    q = (np.round(angle / (np.pi / 4)) % 4).astype(np.int8)
    p = np.pad(score, 1)
    c = p[1:-1, 1:-1]
    sides = {0: (p[1:-1, :-2], p[1:-1, 2:]), 1: (p[:-2, :-2], p[2:, 2:]),
             2: (p[:-2, 1:-1], p[2:, 1:-1]), 3: (p[:-2, 2:], p[2:, :-2])}
    keep = np.zeros(c.shape, bool)
    for k, (a, b) in sides.items():
        keep |= (q == k) & (c >= a) & (c >= b)
    return np.where(keep & (c > 0), c, 0).astype(np.float32)


def seeded(weak, strong):
    """약한 기준을 넘은 덩어리 중 강한 점을 포함한 것만 남긴다 (이중 임계값)."""
    n, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[strong & weak])] = True
    keep[0] = False
    return (keep[labels] * 255).astype(np.uint8)


_PROP_KERNELS = {}


def _prop_kernels(n, length, width):
    """텐서 전파에 쓰는 방향별 커널 목록 [(각도, 커널)]."""
    # 방향 칸마다 그 방향으로 길쭉한 가우시안 커널. 매번 만들면 느려서 캐시해 둔다
    key = (n, length, width)
    if key not in _PROP_KERNELS:
        size = int(6 * length) | 1
        g = cv2.getGaussianKernel(size, length) @ cv2.getGaussianKernel(size, width).T
        ks = []
        for k in range(n):
            phi = np.pi * k / n
            center = ((size - 1) / 2, (size - 1) / 2)
            M = cv2.getRotationMatrix2D(center, 90 - np.degrees(phi), 1.0)
            r = cv2.warpAffine(g.astype(np.float32), M, (size, size))
            ks.append((phi, r / r.sum()))
        _PROP_KERNELS[key] = ks
    return _PROP_KERNELS[key]


def tensor_propagate(score, angle, cfg):
    """텐서 전파 (Chen et al. 2021). 끊긴 균열 조각을 방향을 따라 이어 준다.

    각 점이 자기 선 방향으로 길쭉한 텐서를 이웃에 뿌리고, 점마다 받은 걸 더해서 고윳값을 구한다.
    한 방향으로 모인 정도(l1 - l2)가 새 세기, e1이 새 방향이 되고, 이걸 tensor_propagate번 반복한다.
    방향이 엇갈린 표는 서로 상쇄된다.

    처음 세기는 1 - exp(-score^2 / (2 * beta^2)), 그다음부터는 이전 결과를 0~1로 자른 값.
    결과는 곧은 선 위 점이 받는 합(대략 tensor_ma * sqrt(pi))으로 나눈다. 사진마다 따로 맞추지 않는다.
    """
    n = cfg["tensor_bins"]
    ks = _prop_kernels(n, cfg["tensor_ma"], cfg["tensor_mi"])
    m = (1 - np.exp(-score.astype(np.float32) ** 2 / (2 * cfg["tensor_beta"] ** 2))).astype(np.float32)
    theta = angle + np.pi / 2                                  # 가로지르는 방향 -> 선이 뻗는 방향
    for _ in range(cfg["tensor_propagate"]):
        k_of = (np.round(np.mod(theta, np.pi) / (np.pi / n)).astype(np.int64)) % n
        A = np.zeros_like(m)
        B = np.zeros_like(m)
        C = np.zeros_like(m)
        for k, (phi, ker) in enumerate(ks):
            sel = np.where(k_of == k, m, 0).astype(np.float32)
            if not sel.any():
                continue
            t = cv2.filter2D(sel, -1, ker / ker.max(), borderType=cv2.BORDER_CONSTANT)
            c, s_ = np.cos(phi), np.sin(phi)
            A += t * c * c
            B += t * c * s_
            C += t * s_ * s_
        strength = np.sqrt((A - C) ** 2 + 4 * B * B) / (cfg["tensor_ma"] * np.sqrt(np.pi))
        strength = strength.astype(np.float32)
        theta = 0.5 * np.arctan2(2 * B, A - C)
        m = np.clip(strength, 0, 1)
    return strength


def line_trace(gray, cfg):
    """찾기와 잇기: 선 점수 → 텐서 전파 → 중심선 → 이중 임계값 → 1px 세선화. 균열 흔적(0/255)을 돌려준다."""
    score, angle = line_score(gray, cfg["line_sigmas"])
    field = tensor_propagate(score, angle, cfg)
    # 전파 결과는 띠로만 쓰고 중심선은 원래 점수 방향으로 찾는다
    center = line_centerline(score * field, angle)
    hi = float(cfg["line_hi_abs"])
    trace = seeded(center >= hi * cfg["line_lo_ratio"], center >= hi)
    return cv2.ximgproc.thinning(trace)    # 2px 폭 선이 두 줄로 남는 걸 막으려고


# ---------- 덩어리 ----------

def _components(mask):
    """흔적을 8-연결 덩어리로 나눠 덩어리마다 점 좌표 (x, y) 배열을 돌려준다."""
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(mask)
    lab = labels[ys, xs]
    order = np.argsort(lab, kind="stable")
    lab = lab[order]
    pts = np.stack([xs[order], ys[order]], axis=1).astype(np.int32)
    cuts = np.flatnonzero(np.diff(lab)) + 1
    return [p for p in np.split(pts, cuts) if len(p)]


def shape_features(pts):
    """덩어리의 박스, 길이, 평균 폭, 세장비를 잰다."""
    x, y, w, h = cv2.boundingRect(pts)
    area = float(len(pts))
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)   # 대각선 균열 때문에 회전 박스로 잼
    rw, rh = rw + 1, rh + 1
    length = max(rw, rh)
    return {"bbox": (x, y, w, h), "length": float(length),
            "width": float(area / length),
            "elong": float(length / min(rw, rh))}


def is_crack_shape(det, cfg):
    """길이와 세장비가 기준 이상인가."""
    return det["length"] >= cfg["crack_min_length"] and det["elong"] >= cfg["crack_min_elong"]


def valley_score(gray, pts, width, cfg):
    """양옆이 둘 다 가운데보다 밝은 정도 (0~1).

    후보 위 몇 군데에서 선 방향을 구하고, 직각으로 양옆 밝기와 가운데 밝기 차를 잰다.
    두 차이 중 작은 쪽 / 큰 쪽을 쓰므로 진짜 골짜기(균열)는 1 근처, 한쪽만 밝은
    계단(그림자 경계, 차선 옆)은 0 근처가 나온다. 위치별 값의 중앙값을 돌려준다.
    """
    H, W = gray.shape
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=min(cfg["valley_samples"], len(pts)), replace=False)
    d = width / 2 + cfg["valley_gap"]
    r2 = cfg["valley_radius"] ** 2
    scores = []
    for x, y in pts[idx].astype(np.float32):
        near = pts[((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2) <= r2].astype(np.float32)
        if len(near) < 3:
            continue
        cov = np.cov((near - near.mean(0)).T)
        vals, vecs = np.linalg.eigh(cov)
        nx, ny = -vecs[1, 1], vecs[0, 1]                     # 선 방향에 직각
        sides = []
        for sgn in (1, -1):
            sx, sy = int(round(x + sgn * d * nx)), int(round(y + sgn * d * ny))
            if not (0 <= sx < W and 0 <= sy < H):
                break
            sides.append(gray[sy, sx] - gray[int(y), int(x)])
        if len(sides) < 2:
            continue
        lo, hi = min(sides), max(sides)
        scores.append(0.0 if lo <= 0 or hi <= 0 else lo / hi)
    return float(np.median(scores)) if scores else 1.0     # 못 재면 버리지 않음


def _touches_border(bbox, W, H, cfg):
    """박스가 drop_border로 지정한 테두리에 닿았는가."""
    # 설정은 위쪽만("t"). 위 경계에는 먼 차나 인도가 걸리고, 아래까지 버리면 화면 밖으로 이어진 균열을 놓친다
    x, y, w, h = bbox
    m = cfg["border_margin"]
    sides = cfg["drop_border"]
    return (("t" in sides and y <= m) or ("b" in sides and y + h >= H - m)
            or ("l" in sides and x <= m) or ("r" in sides and x + w >= W - m))
