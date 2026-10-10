"""
포트홀 2단계 검출 — 후보 생성(단계 B) → 후보 검증(단계 C).

detect()에서 pothole_method="texture"일 때만 쓰인다. 기본값 "adaptive"는 1차 방식(적응형 임계값) 그대로라
이 파일을 추가해도 기존 설정(p2bd 등)의 결과는 바뀌지 않는다. 최종 설정: configs/p2bd_pothole.json

포트홀을 보는 관점
  1차 방식은 포트홀을 "주변보다 어두운 덩어리"로 정의했다. 개발 세트 정답 포트홀을 재 보면 밝기 중앙값은 주변 노면과
  거의 같고, 대신 안쪽이 거칠고(깨진 골재 · 경계) 일부만 어둡다(패인 곳의 그림자). 그래서 이 파일은
  "같은 거리의 노면보다 거칠고, 어두운 부분을 포함하며, 노면 위에 있는 둥근 영역"을 포트홀 후보로 본다.
  (근거 수치와 설정을 고른 과정은 보고서 · results/experiments.md · analysis/pothole_diagnosis.py)

처리 순서 (긴 변 1024로 맞춘 전처리 결과 BGR 영상이 입력)
  B. 후보 생성
     1) 기울기 세기: σ1 가우시안 평활 → 소벨 x · y → 크기                            (수업 3-1 그레이디언트)
     2) 거칠기 지도: 기울기 세기를 k×k 창으로 평균                                    (국소 평균)
     3) 행별 표준화: 같은 높이(= 카메라에서 같은 거리)끼리 중앙값 · MAD로 z 점수
        → 먼 곳일수록 질감이 촘촘해지는 원근 차이를 행 단위로 보정 (조감도 변환 없이)
     4) z > 기준인 픽셀 → 열림 5×5로 잡음 제거 → 연결 요소 → 면적 범위 안의 것만 후보 (수업 2-1 형태학 · 연결 요소)
  C. 후보 검증 (모두 통과해야 포트홀)
     ① 색도 차 ≤ pothole_chroma_max    : 상자 안과 주변 고리의 정규화 RGB 색도 차 — 차량 · 간판 등 색이 다른 물체 제거
     ② 어두운 10% ≥ pothole_darkp10_min : 상자 안 하위 10% 밝기가 주변보다 어둡거나 같음 — 패인 부분의 그림자
     ③ 세장비 ≤ pothole_aspect_max      : 상자가 가늘고 길면 균열 · 차선으로 보고 제거
     ④ 높이 ≥ pothole_min_rel_y         : 상자 중심이 화면 위쪽(원경 · 노면 밖)이면 제거        (0 = 끔)
     ⑤ 노면 색 ≤ pothole_road_color_max : 주변 고리가 그 사진의 도로 표본과 색이 비슷한가
                                          — 균열 거르기와 같은 도구(src/roadcue.py)            (None = 끔)

출력은 detect()의 다른 후보와 같은 사전 형식이다 (bbox = (x, y, w, h), 입력 영상 좌표).
"""
import cv2
import numpy as np


def texture_z(gray, k, row_band=15):
    """거칠기 z 지도: 픽셀마다 '같은 높이의 노면 대비 얼마나 거친가'.

    gray     : 회색 영상 (float32 권장)
    k        : 거칠기를 평균할 창 크기 (px)
    row_band : 행 통계(중앙값 · MAD)를 위아래로 이만큼 평활 — 한 행만 쓰면 값이 들쭉날쭉해서
    반환     : z = (거칠기 − 그 행의 중앙값) ÷ (그 행의 MAD × 1.4826)
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)            # 미세 잡음이 기울기를 키우지 않게
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    E = cv2.blur(grad, (k, k))                                             # k×k 평균 = 그 주변의 거칠기
    med = np.median(E, axis=1)                                             # 행마다 '보통 노면'의 거칠기
    mad = np.median(np.abs(E - med[:, None]), axis=1) * 1.4826 + 1e-3     # 행마다 퍼짐 (이상값에 강한 표준편차)
    med = cv2.blur(med[:, None], (1, row_band)).ravel()
    mad = cv2.blur(mad[:, None], (1, row_band)).ravel()
    return (E - med[:, None]) / mad[:, None]


def _box_cues(gray, chrom, box):
    """검증 ① · ②에 쓰는 값. 상자 안 vs 주변 고리(상자를 사방으로 절반씩 넓힌 창 − 상자).

    gray  : 회색 영상, chrom : 정규화 RGB 색도 (r/(R+G+B), g/(R+G+B)) 2채널, box : (x1, y1, x2, y2)
    반환  : {"chroma": 색도 중앙값 차 × 100, "dark_p10": (고리 하위 10% − 안 하위 10%) ÷ 고리 사분위 범위}
            상자나 고리가 너무 작으면 None (그 후보는 버림)
    """
    H, W = gray.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    rx1, ry1, rx2, ry2 = max(0, x1 - bw // 2), max(0, y1 - bh // 2), min(W, x2 + bw // 2), min(H, y2 + bh // 2)
    ring = np.ones((ry2 - ry1, rx2 - rx1), bool)
    ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False                   # 넓힌 창에서 상자 부분을 뺀 고리
    if bw < 4 or bh < 4 or ring.sum() < 20:
        return None
    gi, gr = gray[y1:y2, x1:x2].ravel(), gray[ry1:ry2, rx1:rx2][ring]
    q1, q3 = np.percentile(gr, [25, 75])
    dark_p10 = (np.percentile(gr, 10) - np.percentile(gi, 10)) / max(q3 - q1, 1.0)   # + = 안쪽 어두운 부분이 더 어두움
    ci = np.median(chrom[y1:y2, x1:x2].reshape(-1, 2), 0)
    cr = np.median(chrom[ry1:ry2, rx1:rx2][ring], 0)
    return {"chroma": float(np.linalg.norm(ci - cr) * 100), "dark_p10": float(dark_p10)}


def texture_potholes(img, cfg, keep_noise=False):
    """포트홀 후보 생성 → 검증. img: BGR(또는 회색) 영상, cfg: detect.DEFAULT_CFG에 덮어쓴 설정.

    keep_noise=True면 검증에서 떨어진 후보도 type="noise"로 함께 돌려준다 (분석용).
    """
    road_color_max = cfg.get("pothole_road_color_max")                    # None = 검증 ⑤ 끔
    min_rel_y = cfg.get("pothole_min_rel_y", 0.0)                          # 0 = 검증 ④ 끔
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    s = bgr.astype(np.float32).sum(2) + 1
    chrom = np.stack([bgr[..., 2] / s, bgr[..., 1] / s], 2)               # 밝기와 무관한 색 (그림자에 덜 흔들림)

    cue = None                                                             # 검증 ⑤용 노면 단서 지도 (켤 때만 계산)
    if road_color_max is not None:
        if __package__:
            from .roadcue import cue_values, maps, ref_stats
        else:
            from roadcue import cue_values, maps, ref_stats
        cue_maps = maps(bgr)
        cue = (cue_maps, ref_stats(cue_maps, H, W), cue_values)

    # ---- 단계 B. 후보 생성 ----
    Z = texture_z(gray, cfg["pothole_tex_k"])
    m = (Z > cfg["pothole_tex_z"]).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # 점 같은 잡음 제거
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = []
    for i in range(1, n):                                                  # 0 = 배경
        x, y, w, h, a = (int(v) for v in st[i])
        if not (cfg["pothole_min_frac"] * H * W <= a <= cfg["pothole_max_frac"] * H * W):
            continue                                                       # 너무 작거나(질감 점) 너무 큰(넓은 영역) 것
        aspect = max(w, h) / max(1, min(w, h))
        cues = _box_cues(gray, chrom, (x, y, x + w, y + h))
        if cues is None:
            continue
        cues["rel_y"] = (y + h / 2) / H

        # ---- 단계 C. 검증 ----
        ok = (cues["chroma"] <= cfg["pothole_chroma_max"]                 # ① 색이 주변과 비슷
              and cues["dark_p10"] >= cfg["pothole_darkp10_min"]          # ② 어두운 부분을 포함
              and aspect <= cfg["pothole_aspect_max"]                      # ③ 가늘고 길지 않음
              and cues["rel_y"] >= min_rel_y)                              # ④ 화면 위쪽이 아님
        if ok and cue is not None:                                         # ⑤ 주변이 도로처럼 보임
            cv = cue[2](cue[0], cue[1], (x, y, x + w, y + h), W, H)
            cues["road_color"] = None if cv is None else cv["color"]
            ok = cv is None or cv["color"] <= road_color_max               # 고리가 너무 작아 못 재면 버리지 않음 (균열과 같음)

        if ok or keep_noise:
            out.append({"bbox": (x, y, w, h), "area": float(a), "length": float(max(w, h)), "width": float(a / max(w, h)),
                        "elong": float(aspect), "contrast": 0.0, "branch": "pothole",
                        "type": "pothole" if ok else "noise", "cues": {**cues, "aspect": aspect}})
    return out
