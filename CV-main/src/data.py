"""입력 이미지 목록 — 데이터셋 이름 또는 폴더 → 이미지 경로 목록.

- RDD 분할: rdd_dev(70%) / rdd_test(30%) — labels/split_rdd.csv. rdd_tune / rdd_val은 rdd_dev를 다시 나눈 것 (labels/split_rdd_dev.csv)
  rdd_test는 최종 확인용이라 실행기(preprocessing/run_preprocess.py · run_road_detection.py)에서 막는다.
- 제공 13장 · 직접 촬영분: data/provided, data/captured
"""
import csv
from pathlib import Path

from paths import CAPTURED_DIR, LABELS_DIR, PROVIDED_DIR, RDD_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DATASETS = {"provided": PROVIDED_DIR, "captured": CAPTURED_DIR, "rdd": RDD_DIR / "img"}


def list_images(dataset):
    """dataset 이름(provided / captured / rdd / rdd_dev / rdd_test / rdd_tune / rdd_val) 또는 폴더 경로 → (이름, 폴더, 이미지 경로 목록)."""
    if dataset in ("rdd_dev", "rdd_test"):
        _, folder, images = list_images("rdd")
        with (LABELS_DIR / "split_rdd.csv").open(encoding="utf-8", newline="") as f:
            keep = {r["image"] for r in csv.DictReader(f) if r["split"] == dataset[4:]}
        return dataset, folder, [p for p in images if p.name in keep]
    if dataset in ("rdd_tune", "rdd_val"):
        _, folder, images = list_images("rdd")
        with (LABELS_DIR / "split_rdd_dev.csv").open(encoding="utf-8", newline="") as f:
            keep = {r["image"] for r in csv.DictReader(f) if r["split"] == dataset[4:]}
        return dataset, folder, [p for p in images if p.name in keep]
    folder = Path(DATASETS.get(dataset, dataset))
    if not folder.is_dir():
        raise ValueError(f"error:이미지 폴더 없음 {folder}")
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
                    and not any(part.startswith(".") for part in p.relative_to(folder).parts))
    name = dataset if dataset in DATASETS else folder.name
    return name, folder, images
