"""
균열 종류별 Recall (세로 · 가로 · 거북등) — 실행들의 detections.json을 RDD 정답 종류로 나눠 본다 (개발 세트 · 진단용).
맞힘 = 균열 후보 하나 이상이 면적의 50% 이상 그 정답 박스 안 (in50과 같은 기준, 정답마다 따로 셈).

실행: python analysis/subtype_recall.py 이름=run_… [이름=run_…]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import RDD_DIR, list_images  # noqa: E402
from evaluate import _area, _inter, _xyxy  # noqa: E402
from paths import OUTPUT_DIR  # noqa: E402

KIND = {"longitudinal crack": "세로", "transverse crack": "가로", "alligator crack": "거북등"}


def main():
    _, _, images = list_images("rdd_dev")
    gts = {}
    for p in images:
        ann = RDD_DIR / "ann" / f"{p.name}.json"
        for o in (json.loads(ann.read_text(encoding="utf-8"))["objects"] if ann.exists() else []):
            if o["classTitle"] in KIND:
                (ax, ay), (bx, by) = o["points"]["exterior"]
                gts.setdefault(p.name, []).append((KIND[o["classTitle"]], (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))))
    n = {k: sum(g[0] == k for gs in gts.values() for g in gs) for k in KIND.values()}
    print("종류 (정답 수) " + "".join(f"{k + f' ({n[k]})':>12}" for k in KIND.values()))
    for arg in sys.argv[1:]:
        name, run = arg.split("=")
        dets = json.load(open(OUTPUT_DIR / "pipeline" / run / "detections.json"))
        hit, tot = {k: 0 for k in KIND.values()}, {k: 0 for k in KIND.values()}
        for img, gs in gts.items():
            ds = [_xyxy(d["bbox"]) for d in dets.get(img, []) if d["type"] == "crack"]
            for k, g in gs:
                tot[k] += 1
                hit[k] += any(_inter(d, g) / max(_area(d), 1.0) >= 0.5 for d in ds)
        print(f"{name:13} " + "".join(f"{hit[k] / tot[k]:>12.3f}" for k in KIND.values()))


if __name__ == "__main__":
    main()
