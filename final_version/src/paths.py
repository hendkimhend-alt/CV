import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _find_data_dir(root):
    # 이 폴더의 data/가 비어 있으면 위 폴더들(세 단계까지)의 data/를 쓴다
    for base in [root, *list(root.parents)[:3]]:
        d = base / "data"
        if d.is_dir() and any(p.is_dir() for p in d.iterdir()):
            return d
    return root / "data"


DATA_DIR = Path(os.environ.get("CV_DATA_DIR") or _find_data_dir(ROOT))


# cv2.imread / imwrite는 Windows에서 한글 경로를 못 읽어서 바이트로 읽고 쓴다
def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없음: {path}")
    return img


def imwrite(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise OSError(f"이미지를 저장할 수 없음: {path}")
    buf.tofile(str(path))
