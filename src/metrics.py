"""개발 A 1차: A1.py와 같은 공식으로 영상 품질 지표 9개를 측정한다."""

import math

import cv2
import numpy as np

if __package__:
    from .preprocess import validate_image
else:
    from preprocess import validate_image

METRIC_NAMES = (
    "gray_mean", "gray_std", "block_mean_std_4x4", "saturation_ratio",
    "dark_ratio", "bright_ratio", "hsv_v_mean", "laplacian_variance", "noise_sigma",
)
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
NOISE_KERNEL.setflags(write=False)
NOISE_FACTOR = math.sqrt(math.pi / 2)


# 기능: 최종 영상의 밝기·구역 간 조도·포화·경계·노이즈 지표를 측정한다.
# 특징: 모집단 표준편차, 0~1 비율, 유효 내부 노이즈 응답 등 A1의 계산식을 그대로 유지한다.
#       4×4 블록은 남는 픽셀까지 나누고 각 블록 평균에 동일 가중치를 준다.
#       작은 영상의 측정 불가 값은 None이다. 이 함수는 측정만 담당하고 자동 분류는 preprocess가 담당한다.
def measure_quality(img):
    validate_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    block_std = None
    if height >= 4 and width >= 4:
        blocks = [float(block.mean()) for rows in np.array_split(gray, 4, axis=0)
                  for block in np.array_split(rows, 4, axis=1)]
        block_std = float(np.std(blocks, ddof=0))
    noise = None
    if height >= 3 and width >= 3:
        # 기능: 정수 커널의 응답을 int16으로 계산해 임시 배열 메모리와 연산량을 줄인다.
        # 특징: uint8 입력에서 응답 범위는 -2040~2040으로 int16 안에 들어간다.
        #       절댓값 합은 int64로 계산해 누적 오버플로를 피하며 기존 float64 결과와 같다.
        response = cv2.filter2D(gray, cv2.CV_16S, NOISE_KERNEL)
        noise = float(NOISE_FACTOR * np.abs(response[1:-1, 1:-1]).sum(dtype=np.int64)
                      / (6 * (width - 2) * (height - 2)))
    return {
        "gray_mean": float(gray.mean()), "gray_std": float(gray.std(ddof=0)),
        "block_mean_std_4x4": block_std,
        "saturation_ratio": float(np.mean((gray == 0) | (gray == 255))),
        "dark_ratio": float(np.mean(gray < 40)), "bright_ratio": float(np.mean(gray > 215)),
        "hsv_v_mean": float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].mean()),
        # ksize=1의 응답은 -1020~1020이다. int16 응답의 분산은 NumPy가 float64로 계산한다.
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_16S, ksize=1,
                                                  borderType=cv2.BORDER_REFLECT_101).var()),
        "noise_sigma": noise,
    }
