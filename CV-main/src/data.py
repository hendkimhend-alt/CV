"""
데이터셋 목록과 정답 박스 로더. (계획서상 개발 A 담당 — 전체 실행 파일을 위해 B가 초안 작성)

정답 박스는 원본 이미지 좌표 (type, x1, y1, x2, y2), type은 "crack" / "pothole".
- RDD2020: <RDD_DIR>/ann/<이미지명>.json (Supervisely). 균열 3종 → crack, pothole → pothole, 그 외(other corruption)는 제외
- RDD 분할: rdd_dev(70%) / rdd_test(30%) — labels/split_rdd.csv (analysis/make_split.py). 값 고르기는 dev에서만, test는 마지막에 한 번
- 제공 13장·직접 촬영분: labels/<데이터 이름>.csv (열: image, x, y, w, h, type) — 문서 담당이 만드는 정답 CSV
  labels/는 git에 올라가 팀이 공유한다 (data/는 안 올라감). 파일이 없으면 정답 없음 → 평가 칸은 비고 나머지 지표만 기록
"""
import csv
import json
from pathlib import Path

from paths import CAPTURED_DIR, LABELS_DIR, PROVIDED_DIR, RDD_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RDD_TYPES = {"longitudinal crack": "crack", "transverse crack": "crack", "alligator crack": "crack",
             "pothole": "pothole"}
DATASETS = {"provided": PROVIDED_DIR, "captured": CAPTURED_DIR, "rdd": RDD_DIR / "img"}


def list_images(dataset):
    """dataset 이름(provided / captured / rdd) 또는 폴더 경로 → (이름, 이미지 경로 목록)."""
    if dataset in ("rdd_dev", "rdd_test"):
        _, folder, images = list_images("rdd")
        with (LABELS_DIR / "split_rdd.csv").open(encoding="utf-8", newline="") as f:
            keep = {r["image"] for r in csv.DictReader(f) if r["split"] == dataset[4:]}
        return dataset, folder, [p for p in images if p.name in keep]
    folder = Path(DATASETS.get(dataset, dataset))
    if not folder.is_dir():
        raise ValueError(f"error:이미지 폴더 없음 {folder}")
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
                    and not any(part.startswith(".") for part in p.relative_to(folder).parts))
    name = dataset if dataset in DATASETS else folder.name
    return name, folder, images


def load_gt(dataset, name):
    """정답 로더를 돌려준다: loader(image_path) -> list[(type, x1, y1, x2, y2)] 또는 None(정답 없음)."""
    if dataset.startswith("rdd"):
        def rdd(path):
            ann = RDD_DIR / "ann" / f"{path.name}.json"
            if not ann.exists():
                return None
            boxes = []
            for o in json.loads(ann.read_text(encoding="utf-8"))["objects"]:
                kind = RDD_TYPES.get(o["classTitle"])
                if kind:
                    (x1, y1), (x2, y2) = o["points"]["exterior"]
                    boxes.append((kind, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            return boxes
        return rdd

    gt_csv = LABELS_DIR / f"{name}.csv"
    if not gt_csv.exists():
        return lambda path: None
    table = {}
    with gt_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
            kind = "pothole" if "pothole" in r["type"].lower() else "crack"
            table.setdefault(r["image"], []).append((kind, x, y, x + w, y + h))
    # 정답 CSV에 없는 사진 = 손상 없음으로 본다 (그 데이터 전체에 정답을 만들었으므로)
    return lambda path: table.get(path.name, table.get(path.stem, []))
