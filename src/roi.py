"""
ROI 자동 — 사진마다 도로가 시작하는 높이를 찾아 그 위만 잘라 낸다 (preprocess의 roi="auto").

왜: 고정 비율(하단 50% 등)은 노면이 화면 가득한 사진(드론 · 오토바이 · 근접)에서 손상을 잘라 내고,
    원경 사진도 도로가 시작하는 높이가 사진마다 다르다. 손상을 잘라 내는 실수는 되돌릴 수 없으므로 넉넉하게(덜) 자른다.

방법 (수업: 색 모델 [2-1], 히스토그램)
① 도로 견본: 높이 60~80% · 가로 30~70% (맨 아래는 와이퍼 · 보닛일 수 있어 피함)
   → 색(LAB의 a · b), 질감(라플라시안 크기), 밝기의 기준값
② 픽셀별 도로다움 = 색이 견본과 비슷 + 너무 매끈하지 않음(하늘) + 너무 밝지 않음(하늘 · 흰 벽)
   → 가로 띠 24개마다 도로다운 픽셀 비율
③ 견본 띠에서 위로 올라가며 도로 띠(비율 ≥ 0.5)가 끊기는 곳(한 띠 끊김은 허용) = 도로 시작
   → 사진 높이의 5%를 여유로 두고 그 위만 자름

알려진 약점: 견본이 그림자 속이면 햇빛 받은 도로를 "도로 아님"으로 볼 수 있다 /
            저조도에선 도로와 주변을 못 가려 안 자른다 (안전한 쪽으로 실패)
"""
import cv2
import numpy as np

N_BANDS = 24
MARGIN = 0.05          # 찾은 높이보다 사진 높이의 5% 위에서 자른다 (손상 보호용 여유)
ROAD_FRAC = 0.5        # 띠 안 도로다운 픽셀이 이 비율 이상이면 도로 띠
WORK_WIDTH = 512       # 계산용 축소 폭


def road_top(bgr):
    """BGR 원본 → 잘라 낼 위쪽 비율 (0 = 안 자름 ~ 1)."""
    h0, w0 = bgr.shape[:2]
    img = cv2.resize(bgr, (WORK_WIDTH, max(1, round(WORK_WIDTH * h0 / w0))), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(L, cv2.CV_32F, ksize=3)), (0, 0), 3)

    ref = (slice(int(.60 * H), int(.80 * H)), slice(int(.30 * W), int(.70 * W)))
    ma, mb = np.median(A[ref]), np.median(B[ref])
    sa, sb = max(float(A[ref].std()), 3.0), max(float(B[ref].std()), 3.0)
    mL, sL = np.median(L[ref]), float(L[ref].std())
    mt = np.median(tex[ref])

    road = ((np.sqrt(((A - ma) / sa) ** 2 + ((B - mb) / sb) ** 2) < 3)
            & (tex > 0.35 * mt)
            & (L < mL + max(3 * sL, 45)))
    bands = [road[int(i * H / N_BANDS):int((i + 1) * H / N_BANDS)].mean() for i in range(N_BANDS)]

    i = top = int(.70 * N_BANDS)
    gap = 0
    while i > 0:
        i -= 1
        if bands[i] >= ROAD_FRAC:
            top, gap = i, 0
        else:
            gap += 1
            if gap > 1:
                break
    return max(0.0, top / N_BANDS - MARGIN)
