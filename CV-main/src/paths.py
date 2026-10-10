"""
공통 경로와 이미지 입출력. 모든 스크립트는 경로를 여기서만 가져온다.

- 경로는 저장소 기준 상대 위치 → Windows·macOS 어디서 clone해도 그대로 동작
- 데이터·출력 위치가 다르면 환경변수로 덮어쓴다 (코드 수정 불필요)
    CV_DATA_DIR   기본 <저장소>/data
- 이미지 읽기·쓰기는 한글 경로를 위해 np.fromfile / imencode 사용 (Windows의 cv2.imread는 한글 경로 실패)
"""
import os
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CV_DATA_DIR", REPO_ROOT / "data"))
LABELS_DIR = REPO_ROOT / "labels"            # RDD 분할 CSV (rdd_test 보호에 사용)

PROVIDED_DIR = DATA_DIR / "provided"          # 제공 13장
CAPTURED_DIR = DATA_DIR / "captured"          # 직접 촬영분
RDD_DIR = DATA_DIR / "RDD2020_train" / "train"  # img/, ann/


def imread(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags) if data.size else None
    if img is None:
        raise ValueError(f"error:이미지 읽기 실패 {path}")
    return img


def imwrite(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise OSError(f"error:이미지 인코딩 실패 {path}")
    buf.tofile(str(path))
