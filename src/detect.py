"""
개발 B - 검출기.

detect(img, cfg) -> list[{bbox, type, area, length, width, elong, contrast, branch}]

입력 img는 A의 preprocess() 출력(노면 ROI + 긴 변 1024 정규화)을 가정한다.
혼자 돌릴 때는 run_detect.py의 prepare()가 같은 처리를 해 준다.

- D0 (비교 기준): Canny → Closing
- D1: 균열 branch   Black-hat(작은 커널) → [선 열림] → 이중 임계값 → Closing → [조각 잇기]
      포트홀 branch Adaptive Threshold(큰 블록) → Opening
- 공통: 연결 요소 → [가까운 조각 묶기] → 형태 특징 → 규칙으로 crack / pothole / noise → [가장자리 제외]
        → [양쪽 확인: 균열 후보가 "양쪽 주변보다 모두" 어두운가 — 한쪽만 밝은 계단(그림자 경계 · 차선 옆)은 noise]

[ ]는 cfg로 켜고 끄는 단계. 기본값: 가장자리 제외만 켜짐. 선 열림·조각 묶기는 RDD 평가에서 효과가 없어 꺼 둠
(재현용으로 남김 — analysis/eval_detector.py). 계획서 원안(adaptive threshold)은 crack_thresh="adaptive"로 재현 가능.
bbox는 (x, y, w, h), 입력 img 좌표계 기준.
"""
import cv2
import numpy as np

DEFAULT_CFG = {
    "detector": "D1",            # "D0" | "D1"
    "keep_noise": False,         # True면 noise 판정 후보도 반환

    # D0
    "canny_low": 50,
    "canny_high": 150,
    "d0_close_ksize": 5,

    # D1 균열 branch
    "blackhat_ksize": 7,         # 균열 폭(2~3px)보다 크고 그림자 띠보다 작게 [L4]. 15는 그림자 띠까지 잡음
    "line_len": 0,               # [실험, 효과 없음] >0이면 Black-hat 결과를 막대로 여러 방향 열림 → 선 모양만 남김
    "line_angles": 12,           # 막대 방향 개수 (180° / 12 = 15° 간격)
    "crack_thresh": "hysteresis",  # "hysteresis" | "adaptive" (계획서 원안, 비교용)
    # hysteresis: 강한 픽셀(씨앗)과 이어진 약한 픽셀만 살림 — Canny 이중 임계값 [3-1]을 Black-hat에 적용
    "crack_hi_pct": 97.0,        # 강한 기준 = 사진 안 Black-hat 상위 3% (RDD 비교: 99는 재현율 하락)
    "crack_hi_min": 20,          # 단, 최소 20 (흐린 사진에서 잡음이 씨앗이 되는 것 방지)
    "crack_lo_ratio": 0.5,       # 약한 기준 = 강한 기준 × 0.5
    "crack_block": 31,           # adaptive 모드 전용 (홀수)
    "crack_C": -5,               # adaptive 모드 전용
    "crack_close_ksize": 3,      # 5는 그림자 조각까지 이어 붙임

    # D1 포트홀 branch
    "pothole_block": 101,        # 큰 블록 → 덩어리 단위로 어두운 곳 (홀수)
    "pothole_C": 15,             # 국소 평균보다 15 이상 어두움
    "pothole_open_ksize": 9,

    # 덩어리
    "group_ksize": 0,            # [실험, 효과 없음] >0이면 이 거리 안의 조각을 한 후보로 묶음 (모양은 원래 픽셀로 잼)

    # 규칙
    "min_area_ratio": 0.0002,    # 이보다 작으면 noise
    "crack_min_elong": 3.0,      # 세장비 ≥ 이 값 → crack
    "crack_min_contrast": 0.0,   # 덩어리 평균 Black-hat / 강한 기준 ≥ 이 값 → crack (0 = 끔)
    "pothole_min_area_ratio": 0.002,
    "pothole_max_elong": 3.0,    # 세장비 < 이 값 + 면적 충분 → pothole
    # 포트홀 2단계 (src/pothole.py) — "texture"로 켤 때만. 기본 "adaptive" = 1차 방식(적응형 임계값) 그대로
    "pothole_method": "adaptive",
    "pothole_tex_k": 21,         # [후보] 거칠기(기울기 세기)를 평균할 창 (px, 긴 변 1024 기준)
    "pothole_tex_z": 2.5,        # [후보] 같은 높이 노면 대비 거칠기 z가 이보다 크면 후보
    "pothole_min_frac": 0.0006,  # [후보] 면적 하한 (사진 면적 대비)
    "pothole_max_frac": 0.08,    # [후보] 면적 상한
    "pothole_chroma_max": 1.0,   # [검증 ①] 상자 안 · 주변 색도 차 ≤ 이 값 (색이 다른 물체 제거)
    "pothole_darkp10_min": 0.0,  # [검증 ②] 상자 안 어두운 10%가 주변보다 어둡거나 같음
    "pothole_aspect_max": 3.0,   # [검증 ③] 상자 세장비 ≤ 이 값 (긴 균열 · 차선 제거)
    "pothole_min_rel_y": 0.0,    # [검증 ④] 상자 중심 높이(사진 높이 대비) ≥ 이 값 (0 = 끔, 최종 0.4)
    "pothole_road_color_max": None,  # [검증 ⑤] 주변이 도로 표본과 다른 색 정도(roadcue) ≤ 이 값 (None = 끔, 최종 2.0)
    "drop_border": "t",          # 이 변에 닿는 후보 버림: "t"(위) "b"(아래) "l"(왼) "r"(오른) 조합.
                                 # 위(ROI 경계 = 원경 차량·인도)만 — 아래까지 버리면 화면 밖으로 뻗은 균열을 놓침
    "border_margin": 2,

    # 양쪽 확인 — 균열 정의 "주변보다 어둡고 가는 선"의 "주변보다"를 "양쪽 주변보다 모두"로
    # 균열 = 양쪽이 밝은 골짜기, 그림자 경계 · 차선 옆 = 한쪽만 밝은 계단 → 계단 모양 균열 후보는 noise
    "valley_check": False,       # 기본 끔 (실험: 검출기 이름 뒤 v — D1v, D0v)
    "valley_min_ratio": 0.35,    # 양옆 "가운데보다 밝은 정도"의 작은 쪽 ÷ 큰 쪽 (1 = 완전 대칭) — 이보다 작으면 계단
    "valley_gap": 2,             # 선 폭의 절반 + 이만큼 바깥을 "옆"으로 봄 (px)
    "valley_samples": 40,        # 후보마다 재는 위치 수
    "valley_radius": 7,          # 위치마다 선 방향을 구할 주변 반경 (px)

    # 조각 잇기 — 균열이 조각으로 끊겨 "너무 작음"으로 버려지는 것을 막음 (판정 전에 잇는다)
    # 조각 뼈대의 끝점끼리, 가깝고(link_dist 이내) 서로를 향하는(link_angle 이내) 것만 선으로 이음
    # → 방향이 제각각인 노면 질감 조각은 안 이어짐 (밝기만 보고 잇는 이중 임계값 완화와 다른 점)
    "link_dist": 0,              # 0 = 끔 (실험: 검출기 이름 뒤 l — D1vl). 긴 변 1024 기준 px
    "link_angle": 30,            # 끝점의 바깥 방향과 상대 끝점 쪽 방향의 최대 각도 (도)
    "link_radius": 6,            # 끝점의 바깥 방향을 구할 주변 뼈대 반경 (px)

    # 찾기 방식 — "blackhat": 주변보다 얼마나 어두운가 (지금) / "line": 선 모양으로 파였는가 (Hessian)
    # line: Hessian 선 점수 → 선 중심만 남기기(1px) → 중심선 위 이중 임계값 (Canny 방식을 선에 적용)
    "crack_find": "blackhat",
    "line_sigmas": (1.0, 1.5, 2.0),  # 가우시안 스케일 — 폭 1~4px 균열 (긴 변 1024 기준)
    "line_hi_pct": 80.0,         # 강한 기준 = 사진 안 중심선 점수의 상위 (100 − 이 값)% (line_hi_abs가 없을 때)
    "line_hi_abs": None,         # 고정 강한 기준 (모든 사진 같은 값) — E8-b: 32.5 ≈ 흔적 양 5.9%. 깨끗한 사진엔 흔적이 적게
    "line_lo_ratio": 0.4,        # 약한 기준 = 강한 기준 × 이 값
    "line_thicken": 3,           # 거르기 전에 1px 흔적을 이 폭으로 굵게 — 거르기 규칙(면적 · 세장비 · 양쪽 확인)을 그대로 쓰려고 (최소 조정)
                                 # 1 = 굵게 안 함. 굵게 하면 2~3px 옆의 차선 테두리 · 경계선과 붙는 문제 (E9-b)
    "crack_min_length": 0,       # >0이면 균열 크기 규칙을 면적 대신 길이(회전 사각형 긴 변, px)로 — 1px 흔적용

    # 노면 단서로 거르기 (src/roadcue.py) — {단서: 최댓값} 넘으면 버림. "…_min": 하한 — resid_min(너무 곧음) · depth_min(얕은 골)
    # 노면 위 단서: "streak"(노면 결을 따라감, 최댓값) · "depth_min"
    # 거르기 R1 (개발 세트): {"energy": 1.608, "coherence": 0.294, "color": 2.996}
    "cue_filter": None,

    # 구조장 전파 (논문 반영 2단계 — Chen et al. 2021 아이디어를 참고해 직접 설계): 0 = 끔, n = 반복 횟수
    # 방향을 prop_bins칸으로 나눠 각 칸을 가리키는 점의 점수를 그 방향으로 길쭉한 가우시안(prop_len × prop_width)으로 퍼뜨려 더함
    "line_propagate": 0, "prop_len": 7.0, "prop_width": 1.0, "prop_bins": 12,
    # 여러 크기 합치기: "max"(지금) · "mean"(논문 §3.2)
    "line_scale_combine": "max",
    # 논문식 텐서 전파 (§4 식 12~16): 0 = 끔, n = 반복 횟수 · 타원 σ_ma(긴 축) × σ_mi(짧은 축) · 방향 칸 수
    "tensor_propagate": 0, "tensor_ma": 10.0, "tensor_mi": 2.0, "tensor_bins": 16,
    # 식 9의 β: 논문은 "모든 사진 세기 최댓값의 절반" — 근접 노면 사진에선 균열이 가장 센 구조라 맞지만, 차량 시점에선 최댓값이
    # 차선 · 경계(426)라 균열 표가 0에 가까워짐 → 우리 강한 기준 32.5가 M = 0.5가 되게: β = 32.5 / √(2 ln 2) = 27.6
    "tensor_beta": 27.6,
    "tensor_gate": False,
    # 가이드 필터 (Chen et al. 2021 §2): 찾기 입력에만. None = 끔 · {"r": 반지름, "eps": "std" | "var" | 숫자, "eps_scale": 배수}
    "guided_filter": None,
    # 중심선 NMS: False = 4방향으로 묶어서 (지금) · True = 연속 방향 보간 (논문 §5.1)
    "nms_interp": False,   # True: 찾기 점수 = 선 점수 × 전파 세기, 중심선은 원래 방향 (논문 §5.1 "곡선 띠 안의 중심")
}


def detect(img, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    gray = _to_gray(img)
    H, W = gray.shape

    if cfg["detector"] == "D0":
        branches = [("any", _d0_mask(gray, cfg), None, None)]
    elif cfg["detector"] == "D1":
        mask, bh, hi = _crack_mask(gray, cfg)
        branches = [("crack", mask, bh, hi)]
        if cfg["pothole_method"] == "adaptive":
            branches.append(("pothole", _pothole_mask(gray, cfg), None, None))
    else:
        raise ValueError(f"unknown detector: {cfg['detector']}")

    detections = []
    smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32) if cfg["valley_check"] else None
    cue = None
    if cfg["cue_filter"]:
        if __package__:
            from .roadcue import cue_values, line_resid, maps, onroad_values, ref_stats, tensor
        else:
            from roadcue import cue_values, line_resid, maps, onroad_values, ref_stats, tensor
        cue_maps = maps(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cue = (cue_maps, ref_stats(cue_maps, H, W))
        need_on = any(k in cfg["cue_filter"] for k in ("depth_min", "streak"))
        J = tensor(gray) if need_on else None
        if need_on and smooth is None:
            smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    for branch, mask, bh, hi in branches:
        for pts in _components(mask, cfg["group_ksize"]):
            det = shape_features(pts, bh, hi)
            det["type"] = classify(det, H * W, branch, cfg)
            det["branch"] = branch
            if det["type"] != "noise" and _touches_border(det["bbox"], W, H, cfg):
                det["type"] = "noise"
            if det["type"] == "crack" and cfg["valley_check"]:
                det["valley"] = valley_score(smooth, pts, det["width"], cfg)
                if det["valley"] < cfg["valley_min_ratio"]:
                    det["type"] = "noise"
            if det["type"] == "crack" and cue is not None:
                x, y, w, h = det["bbox"]
                cv = cue_values(cue[0], cue[1], (x, y, x + w, y + h), W, H) or {}
                cv["resid"] = line_resid(pts)
                if need_on:
                    cv.update(onroad_values(pts, smooth, J, (x, y, x + w, y + h), W, H))
                det["cues"] = cv
                for k, t in cfg["cue_filter"].items():
                    if k.endswith("_min"):                      # 하한 (resid_min · depth_min): 이보다 작으면 버림
                        bad = cv[k[:-4]] < t
                    else:
                        bad = k in cv and cv[k] > t
                    if bad:
                        det["type"] = "noise"
                        break
            if det["type"] != "noise" or cfg["keep_noise"]:
                detections.append(det)
    if cfg["detector"] == "D1" and cfg["pothole_method"] == "texture":
        if __package__:
            from .pothole import texture_potholes
        else:
            from pothole import texture_potholes
        detections.extend(texture_potholes(img, cfg, cfg["keep_noise"]))
    return detections


# ---------- 마스크 ----------

def _d0_mask(gray, cfg):
    edges = cv2.Canny(gray, cfg["canny_low"], cfg["canny_high"])
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _kernel(cfg["d0_close_ksize"]))


def _crack_mask(gray, cfg, with_angle=False):
    """→ (흑백 흔적, 점수 지도, 강한 기준) · with_angle=True면 끝에 방향 지도(선을 가로지르는 방향, 라디안)도 — 선 모양 찾기에서만."""
    if cfg["crack_find"] == "line":
        binary, score, hi, angle = line_trace(gray, cfg, with_angle=True)
        if cfg["link_dist"] > 0:
            binary = link_fragments(binary, cfg)
        if cfg["line_thicken"] > 1:
            binary = cv2.dilate(binary, _kernel(cfg["line_thicken"]))
        return (binary, score, hi, angle) if with_angle else (binary, score, hi)
    # 노면보다 어둡고 가는 구조만 밝게 남긴다
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _kernel(cfg["blackhat_ksize"]))
    if cfg["line_len"] > 0:
        bh = _line_open(bh, cfg["line_len"], cfg["line_angles"])
    if cfg["crack_thresh"] == "hysteresis":
        binary, hi = _hysteresis(bh, cfg)
    else:
        binary = cv2.adaptiveThreshold(bh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       cfg["crack_block"], cfg["crack_C"])
        hi = max(float(bh.max()), 1.0)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel(cfg["crack_close_ksize"]))
    if cfg["link_dist"] > 0:
        binary = link_fragments(binary, cfg)
    return (binary, bh, hi, None) if with_angle else (binary, bh, hi)


def link_fragments(binary, cfg):
    """끊긴 균열 조각 잇기 — ① 세선화 ② 끝점(이웃 1개) ③ 끝점의 바깥 방향(끝점 − 주변 뼈대 평균)
    ④ 다른 조각의 끝점 중 거리 ≤ link_dist 이고 두 끝점이 서로를 향하는(각도 ≤ link_angle) 쌍을
    가까운 순서로 하나씩 고름 ⑤ 선(두께 2)으로 이음. 균열처럼 일직선으로 줄지은 조각만 이어진다."""
    skel = cv2.ximgproc.thinning(binary)
    on = (skel > 0).astype(np.uint8)
    nbr = cv2.filter2D(on, cv2.CV_16S, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int16))
    ys, xs = np.nonzero((on == 1) & (nbr == 1))
    if len(xs) < 2:
        return binary
    _, labels = cv2.connectedComponents(on, connectivity=8)
    r = cfg["link_radius"]
    H, W = on.shape
    pts, dirs, labs = [], [], []
    for x, y in zip(xs, ys):
        y1, y2, x1, x2 = max(0, y - r), min(H, y + r + 1), max(0, x - r), min(W, x + r + 1)
        win = labels[y1:y2, x1:x2] == labels[y, x]
        wy, wx = np.nonzero(win)
        if len(wx) < 3:
            continue
        v = np.array([x - (wx.mean() + x1), y - (wy.mean() + y1)], np.float32)
        n = float(np.hypot(*v))
        if n < 1e-3:
            continue
        pts.append((x, y))
        dirs.append(v / n)
        labs.append(labels[y, x])
    if len(pts) < 2:
        return binary
    P, U, Lb = np.array(pts, np.float32), np.array(dirs, np.float32), np.array(labs)
    ii, jj = _near_pairs(P, cfg["link_dist"])             # 거리 ≤ link_dist 인 쌍만 (끝점이 수만 개여도 메모리 일정)
    D = P[jj] - P[ii]                                       # i → j
    dist = np.hypot(D[:, 0], D[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        V = D / dist[:, None]
    cos = np.cos(np.radians(cfg["link_angle"]))
    ok = ((dist > 0) & (dist <= cfg["link_dist"]) & (Lb[ii] != Lb[jj])
          & ((U[ii] * V).sum(-1) >= cos)                    # i의 바깥 방향이 j 쪽을 향함
          & ((U[jj] * -V).sum(-1) >= cos))                  # j의 바깥 방향이 i 쪽을 향함
    ii, jj, dist = ii[ok], jj[ok], dist[ok]
    out = binary.copy()
    used = set()
    for k in np.lexsort((jj, ii, dist)):                    # 가까운 순서 (같으면 번호 순)
        i, j = int(ii[k]), int(jj[k])
        if i in used or j in used:
            continue
        used.update((i, j))
        cv2.line(out, tuple(int(v) for v in P[i]), tuple(int(v) for v in P[j]), 255, 2)
    return out


def _near_pairs(P, d):
    """점들 중 x · y 차이가 모두 d 이하인 쌍 (i < j) — x로 정렬해 창 안만 본다."""
    order = np.argsort(P[:, 0], kind="stable")
    xs, ys = P[order, 0], P[order, 1]
    ends = np.searchsorted(xs, xs + d, side="right")
    I, J = [], []
    for a in range(len(order)):
        if ends[a] > a + 1:
            j = np.arange(a + 1, ends[a])
            j = j[np.abs(ys[j] - ys[a]) <= d]
            I.append(np.full(len(j), a)); J.append(j)
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    I, J = order[np.concatenate(I)], order[np.concatenate(J)]
    return np.minimum(I, J), np.maximum(I, J)


def _line_open(bh, length, n_angles):
    # 막대 모양 구조 요소로 열림: 그 방향으로 length 이상 이어진 밝은 구조만 살아남음.
    # 여러 방향의 최댓값 → 어느 방향이든 "선"이면 남고, 점·짧은 조각은 사라짐
    out = np.zeros_like(bh)
    for i in range(n_angles):
        out = np.maximum(out, cv2.morphologyEx(bh, cv2.MORPH_OPEN, _line_kernel(length, 180.0 * i / n_angles)))
    return out


def _hysteresis(bh, cfg):
    # 기준을 사진마다 정함: Black-hat 값 범위가 사진마다 수십 배 차이 (골재 사진 중앙값 56, 흐린 사진 0)
    hi = max(cfg["crack_hi_min"], float(np.percentile(bh, cfg["crack_hi_pct"])))
    lo = hi * cfg["crack_lo_ratio"]
    n, labels = cv2.connectedComponents((bh >= lo).astype(np.uint8), connectivity=8)
    seeded = np.zeros(n, bool)
    seeded[np.unique(labels[bh >= hi])] = True
    seeded[0] = False                       # 배경
    return (seeded[labels] * 255).astype(np.uint8), hi


# ---------- 찾기: 선 모양 (Hessian) ----------

def line_score(gray, sigmas=(1.0, 1.5, 2.0), combine="max"):
    """Hessian 선 점수 — 밝기를 지형으로 보고, 한 방향으로만 오목한(어두운 골짜기) 정도.
    2차 미분 표 [[Ixx, Ixy], [Ixy, Iyy]]의 고윳값 λ1 ≥ λ2 = 가장 많이 / 가장 적게 휜 방향의 휘어짐.
    어두운 선 = 가로지르는 방향으로 크게 오목(λ1 큰 양수) + 따라가는 방향은 평평(|λ2| 작음) → λ1 − |λ2|.
    점(골재 틈새)은 사방이 오목(λ1 ≈ λ2)해서 점수가 낮다. σ마다 σ²를 곱해 맞춘 뒤 합침 (폭이 다른 균열):
      combine="max"  가장 센 크기 · 방향도 그 크기의 것
      combine="mean" 크기들의 평균 · 방향은 평균에 가장 가까운 크기의 것 (Chen et al. 2021 §3.2 식 11 —
                     최댓값은 한 크기에서만 튀는 잡음이 섞인다)
    반환: 점수, 선을 가로지르는 방향(라디안, λ1의 고유벡터)"""
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
    if combine == "mean":
        best = S.mean(0)
        pick = np.abs(S - best).argmin(0)
    else:
        best = np.zeros_like(g)
        pick = np.zeros(g.shape, np.int64)
        for i in range(len(sigmas)):                      # 같은 값이면 앞 크기 (예전 코드와 같은 결과)
            upd = S[i] > best
            best[upd] = S[i][upd]
            pick[upd] = i
    angle = np.take_along_axis(A, pick[None], 0)[0]
    return best.astype(np.float32), angle.astype(np.float32)


def line_centerline(score, angle):
    """선 중심만 남기기 — 선을 가로지르는 방향으로 양옆 이웃보다 작지 않은 점만 (Canny의 최댓값만 남기기, 4방향)."""
    q = (np.round(angle / (np.pi / 4)) % 4).astype(np.int8)    # 0: 가로 · 1: ↘ · 2: 세로 · 3: ↗
    p = np.pad(score, 1)
    c = p[1:-1, 1:-1]
    sides = {0: (p[1:-1, :-2], p[1:-1, 2:]), 1: (p[:-2, :-2], p[2:, 2:]),
             2: (p[:-2, 1:-1], p[2:, 1:-1]), 3: (p[:-2, 2:], p[2:, :-2])}
    keep = np.zeros(c.shape, bool)
    for k, (a, b) in sides.items():
        keep |= (q == k) & (c >= a) & (c >= b)
    return np.where(keep & (c > 0), c, 0).astype(np.float32)


def line_centerline_interp(score, angle):
    """선 중심만 남기기 — 연속 방향 (Chen et al. 2021 §5.1 "균열에 수직인 방향"): 선을 가로지르는 방향으로 ±1px 떨어진
    두 점의 점수를 쌍선형 보간으로 구해, 둘보다 작지 않은 점만 (4방향으로 묶지 않음)."""
    H, W = score.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    c, s = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    a = cv2.remap(score, xx + c, yy + s, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    b = cv2.remap(score, xx - c, yy - s, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    keep = (score >= a) & (score >= b) & (score > 0)
    return np.where(keep, score, 0).astype(np.float32)


def seeded(weak, strong):
    """이중 임계값: 약한 픽셀 중 강한 픽셀과 이어진 덩어리만 살림."""
    n, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[strong & weak])] = True
    keep[0] = False
    return (keep[labels] * 255).astype(np.uint8)


_PROP_KERNELS = {}


def _prop_kernels(n, length, width):
    """방향 칸마다 그 방향(선이 뻗는 방향)으로 길쭉한 가우시안 커널 (합 1)."""
    key = (n, length, width)
    if key not in _PROP_KERNELS:
        size = int(6 * length) | 1
        g = cv2.getGaussianKernel(size, length) @ cv2.getGaussianKernel(size, width).T     # 세로로 긴 커널
        ks = []
        for k in range(n):
            phi = np.pi * k / n                                                           # 선이 뻗는 방향
            M = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), 90 - np.degrees(phi), 1.0)
            r = cv2.warpAffine(g.astype(np.float32), M, (size, size))
            ks.append((phi, r / r.sum()))
        _PROP_KERNELS[key] = ks
    return _PROP_KERNELS[key]


def propagate(score, angle, cfg):
    """구조장 전파: 각 점이 자기 방향 앞뒤로 점수를 나눠 줌 → 같은 방향으로 늘어선 점끼리 강해지고 끊긴 곳이 채워짐.
    angle = 선을 가로지르는 방향(line_score) → 선이 뻗는 방향 θ = angle + 90°. cos⁸(θ − φ)로 각 칸에 나눠 담음."""
    theta = angle + np.pi / 2
    s = score.astype(np.float32)
    ks = _prop_kernels(cfg["prop_bins"], cfg["prop_len"], cfg["prop_width"])
    weights = [np.cos(theta - phi) ** 8 for phi, _ in ks]
    for _ in range(cfg["line_propagate"]):
        out = np.zeros_like(s)
        for w, (_, k) in zip(weights, ks):
            out += cv2.filter2D(s * w, -1, k, borderType=cv2.BORDER_REFLECT)
        s = out
    return s


def tensor_propagate(score, angle, cfg):
    """논문식 구조 전파 (Chen et al. 2021 §4): 각 점 O가 자기 선 방향 t_O로 막대 텐서 M_O · G(타원) · t_O t_Oᵀ 를
    이웃에 보냄 (G = 선 방향으로 σ_ma, 수직으로 σ_mi인 가우시안 — 식 12~14) → 점마다 받은 텐서를 더함 (식 15)
    → 고윳값 분해: 새 세기 = λ1 − λ2 (한 방향으로 모인 정도 — 엇갈린 표는 상쇄), 새 방향 = e1 (식 16) → 반복.
    M_O = 1 − exp(−점수² / 2β²) (식 9의 세기 항 — 곡선성 항은 선 점수 λ1 − |λ2|에 이미 들어 있음).
    다음 반복의 M_O = 세기 (0~1로 자름 · 세기는 곧은 선이 M이 되게 고정 상수로 나눈 값 — 다시 꺾으면 반복마다 사라짐).
    방향은 tensor_bins칸으로 나눠 칸마다 회전한 타원 커널로 모음.
    반환: 세기, 선을 가로지르는 방향 (line_centerline용)"""
    n = cfg["tensor_bins"]
    ks = _prop_kernels(n, cfg["tensor_ma"], cfg["tensor_mi"])
    m = (1 - np.exp(-score.astype(np.float32) ** 2 / (2 * cfg["tensor_beta"] ** 2))).astype(np.float32)
    theta = angle + np.pi / 2                                  # 선이 뻗는 방향
    for _ in range(cfg["tensor_propagate"]):
        k_of = (np.round(np.mod(theta, np.pi) / (np.pi / n)).astype(np.int64)) % n
        A = np.zeros_like(m); B = np.zeros_like(m); C = np.zeros_like(m)
        for k, (phi, ker) in enumerate(ks):
            sel = np.where(k_of == k, m, 0).astype(np.float32)
            if not sel.any():
                continue
            t = cv2.filter2D(sel, -1, ker / ker.max(), borderType=cv2.BORDER_CONSTANT)  # 꼭짓점 1인 타원 (식 12)
            c, s_ = np.cos(phi), np.sin(phi)
            A += t * c * c; B += t * c * s_; C += t * s_ * s_
        # λ1 − λ2 · 고정 상수로 나눔: M=1인 곧은 선 위 점이 받는 합 ≈ σ_ma·√π → 1 (사진마다 맞추지 않음 — E8 교훈)
        strength = (np.sqrt((A - C) ** 2 + 4 * B * B) / (cfg["tensor_ma"] * np.sqrt(np.pi))).astype(np.float32)
        theta = 0.5 * np.arctan2(2 * B, A - C)                 # e1
        m = np.clip(strength, 0, 1)                            # 다음 반복의 M_O (0~1)
    return strength, (theta - np.pi / 2).astype(np.float32)


def find_input(gray, cfg):
    """찾기 입력 전처리. guided_filter가 있으면 자기 자신을 안내 영상으로 한 가이드 필터 —
    밝기 변화가 큰 곳(균열 경계)은 그대로, 평평한 노면은 평균 (a = σ²/(σ² + ε)). ε = 사진 표준편차("std") · 분산("var") · 숫자."""
    gf = cfg.get("guided_filter")
    if not gf:
        return gray
    g = gray.astype(np.float32)
    eps = gf.get("eps", "std")
    eps = float(g.std()) if eps == "std" else float(g.var()) if eps == "var" else float(eps)
    eps *= float(gf.get("eps_scale", 1.0))
    return cv2.ximgproc.guidedFilter(g, g, int(gf.get("r", 4)), eps)


def line_trace(gray, cfg, hi_pct=None, with_angle=False):
    """찾기(선 모양): Hessian 선 점수 → 중심선 → 중심선 위 이중 임계값 → 1px 흔적 지도.
    with_angle=True면 방향 지도도 돌려줌 (가장 센 σ에서 λ1 고유벡터 = 선을 가로지르는 방향, 연속 라디안 —
    선이 뻗는 방향은 여기에 +90°). NMS 뒤에도 방향을 버리지 않기 위함 (논문 반영 0단계)."""
    score, angle = line_score(find_input(gray, cfg), cfg["line_sigmas"], cfg.get("line_scale_combine", "max"))
    nms = line_centerline_interp if cfg.get("nms_interp", False) else line_centerline
    if cfg.get("tensor_propagate", 0) > 0:
        field, nangle = tensor_propagate(score, angle, cfg)
        if cfg.get("tensor_gate", False):                       # P2-b: 전파는 띠(지지)로만, 중심선은 원래 점수 · 방향으로
            center = nms(score * field, angle)
        else:
            center = nms(field, nangle)
    else:
        field = propagate(score, angle, cfg) if cfg.get("line_propagate", 0) > 0 else score
        center = nms(field, angle)
    if cfg["line_hi_abs"] is not None and hi_pct is None:
        hi = float(cfg["line_hi_abs"])
    else:
        vals = center[center > 0]
        hi = float(np.percentile(vals, cfg["line_hi_pct"] if hi_pct is None else hi_pct)) if vals.size else 1.0
    hi = max(hi, 1e-6)
    trace = seeded(center >= hi * cfg["line_lo_ratio"], center >= hi)
    trace = cv2.ximgproc.thinning(trace)                       # 폭 2px 선의 중심이 두 줄로 남는 경우를 1px로
    return (trace, score, hi, angle) if with_angle else (trace, score, hi)


def _pothole_mask(gray, cfg):
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                   cfg["pothole_block"], cfg["pothole_C"])
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, _kernel(cfg["pothole_open_ksize"]))


# ---------- 덩어리 ----------

def _components(mask, group_ksize=0):
    """흰 픽셀을 연결 요소 [2-1]로 묶어 덩어리별 (x, y) 좌표 배열 목록으로."""
    grouping = cv2.dilate(mask, _kernel(group_ksize)) if group_ksize > 0 else mask
    n, labels = cv2.connectedComponents((grouping > 0).astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(mask)               # 묶음은 넓힌 마스크로, 모양은 원래 픽셀로
    lab = labels[ys, xs]
    order = np.argsort(lab, kind="stable")
    lab, pts = lab[order], np.stack([xs[order], ys[order]], axis=1).astype(np.int32)
    cuts = np.flatnonzero(np.diff(lab)) + 1
    return [p for p in np.split(pts, cuts) if len(p)]


# ---------- 형태 특징 · 규칙 ----------

def shape_features(pts, bh=None, hi=None):
    x, y, w, h = cv2.boundingRect(pts)
    area = float(len(pts))                       # 픽셀 수
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)   # 회전 박스 → 대각선 균열도 세장비가 나옴
    rw, rh = rw + 1, rh + 1                      # 픽셀 중심 기준이라 1px 보정
    length = max(rw, rh)
    width = area / length                        # 평균 폭 ≈ 면적 / 길이
    elong = length / min(rw, rh)
    contrast = float(bh[pts[:, 1], pts[:, 0]].mean() / hi) if bh is not None else 0.0
    return {"bbox": (x, y, w, h), "area": area, "length": float(length),
            "width": float(width), "elong": float(elong), "contrast": contrast}


def classify(det, img_area, branch, cfg):
    area_ratio = det["area"] / img_area
    if branch == "crack" and cfg["crack_min_length"] > 0:
        if det["length"] < cfg["crack_min_length"]:
            return "noise"
    elif area_ratio < cfg["min_area_ratio"]:
        return "noise"
    # 규칙 1: 세장비 (+ 진하기)
    if (branch in ("crack", "any") and det["elong"] >= cfg["crack_min_elong"]
            and det["contrast"] >= cfg["crack_min_contrast"]):
        return "crack"
    # 규칙 2: 면적
    if (branch in ("pothole", "any") and area_ratio >= cfg["pothole_min_area_ratio"]
            and det["elong"] < cfg["pothole_max_elong"]):
        return "pothole"
    return "noise"


def valley_score(gray, pts, width, cfg):
    """균열 후보가 '양쪽 주변보다 모두' 어두운 정도 (0 ~ 1).
    후보 픽셀 중 몇 곳에서: 주변 후보 픽셀로 선 방향을 구하고(주성분) → 직각 양옆(폭/2 + gap)의 밝기 − 가운데 밝기
    → 작은 쪽 ÷ 큰 쪽 (한쪽이라도 가운데보다 어두우면 0). 위치별 값의 중앙값.
    골짜기(균열) ≈ 1, 계단(그림자 경계 · 차선 옆) ≈ 0"""
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
        nx, ny = -vecs[1, 1], vecs[0, 1]                     # 선 방향(최대 고유벡터)에 직각
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
    return float(np.median(scores)) if scores else 1.0     # 잴 수 없으면 버리지 않음


def _touches_border(bbox, W, H, cfg):
    x, y, w, h = bbox
    m = cfg["border_margin"]
    sides = cfg["drop_border"]
    return (("t" in sides and y <= m) or ("b" in sides and y + h >= H - m)
            or ("l" in sides and x <= m) or ("r" in sides and x + w >= W - m))


# ---------- 유틸 ----------

def _to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _line_kernel(length, angle_deg):
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2
    dx, dy = np.cos(np.radians(angle_deg)) * c, -np.sin(np.radians(angle_deg)) * c
    cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return k
