"""
제공 13장에 노면 마스크(ROI)를 파랑으로 입혀 저장 — 그림자 수정 전 · 후 비교.

수정 전 = configs/preprocessing.json에서 roi.region_growing.shadow.enabled=false · roi.seeds.max_shadow_darkening_to_best=25 (리포트 "수정 전과 같은 결과")
수정 후 = 설정 그대로
사진마다 4칸: 원본 · 수정 전 ROI · 수정 후 ROI · 바뀐 곳 (새로 들어온 곳 파랑 · 빠진 곳 빨강). 노란 점 = 시드
입력: CV_DATA_DIR/provided (기본 final0/data/provided)
출력: outputs/roi_shadow_overlay/<사진>.png · summary.png · summary.md
실행: python roi_shadow_overlay.py
      python roi_shadow_overlay.py --before --after roi.features.normalize.texture=true --name "질감 정규화" --out roi_r1_overlay
      (--before · --after = 전처리 설정 덮어쓰기. 빈 --before = 지금 설정)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from paths import OUTPUT_DIR, PROVIDED_DIR, imread  # noqa: E402
from preprocessing.config import load_config  # noqa: E402
from preprocessing.mask_validation import compute_mask_metrics, validate_road_mask  # noqa: E402
from preprocessing.road_mask import extract_road_mask  # noqa: E402

BEFORE = ["roi.region_growing.shadow.enabled=false", "roi.seeds.max_shadow_darkening_to_best=25"]
BLUE, RED = np.array([30, 90, 255]), np.array([230, 40, 40])   # RGB


def run_roi(image, cfg):
    auto = extract_road_mask(image, cfg["roi"])
    seeds = [(s["x"], s["y"]) for s in auto.seeds]
    status = validate_road_mask(compute_mask_metrics(auto.mask, seeds), cfg["validation"])["status"]
    return auto.mask > 0, seeds, status


def paint(rgb, mask, color, alpha=0.5):
    out = rgb.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + color * alpha
    return out.clip(0, 255).astype(np.uint8)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    font_manager.fontManager.addfont(font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()

    ap = argparse.ArgumentParser()
    ap.add_argument("--before", nargs="*", default=BEFORE)
    ap.add_argument("--after", nargs="*", default=[])
    ap.add_argument("--name", default="그림자 허용")
    ap.add_argument("--out", default="roi_shadow_overlay")
    a = ap.parse_args()
    cfg_b, cfg_a = load_config(overrides=a.before), load_config(overrides=a.after)
    out_dir = OUTPUT_DIR / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in PROVIDED_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    rows, thumbs = [], []
    for path in images:
        image = imread(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            mb, sb, vb = run_roi(image, cfg_b)
            ma, sa, va = run_roi(image, cfg_a)
        except Exception as exc:                       # 근접 촬영 등 ROI 자체 실패
            print(f"{path.name}: ROI 실패 — {exc}")
            rows.append((path.name, None, None, "실패", "실패", None, None))
            continue
        added, removed = ma & ~mb, mb & ~ma
        n = mb.size
        rb, ra = mb.sum() / n, ma.sum() / n
        rows.append((path.name, rb, ra, vb, va, added.sum() / n, removed.sum() / n))

        diff = paint(paint(rgb, added, BLUE, 0.65), removed, RED, 0.65)
        panels = [(rgb, "원본", []), (paint(rgb, mb, BLUE), f"수정 전 ROI  {rb:.1%} {vb}", sb),
                  (paint(rgb, ma, BLUE), f"수정 후 ROI ({a.name})  {ra:.1%} {va}", sa),
                  (diff, f"바뀐 곳  새로 들어옴 {added.sum() / n:.1%} (파랑) · 빠짐 {removed.sum() / n:.1%} (빨강)", [])]
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
        for ax, (im, title, seeds) in zip(axes, panels):
            ax.imshow(im)
            if seeds:
                ax.scatter([x for x, _ in seeds], [y for _, y in seeds], s=28, c="yellow", edgecolors="black", linewidths=0.6)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(path.name, fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"{path.stem}.png", dpi=110)
        plt.close(fig)
        thumbs.append((path.name, paint(rgb, mb, BLUE), paint(rgb, ma, BLUE), rb, ra, vb, va))
        print(f"{path.name}: {rb:.1%} {vb} → {ra:.1%} {va} · 새로 {added.sum() / n:.1%} · 빠짐 {removed.sum() / n:.1%}", flush=True)

    # 한눈에: 사진마다 한 줄 (수정 전 · 수정 후)
    fig, axes = plt.subplots(len(thumbs), 2, figsize=(10, 3.1 * len(thumbs)))
    for row, (name, b, a, rb, ra, vb, va) in zip(axes, thumbs):
        for ax, im, title in ((row[0], b, f"{name}  수정 전 {rb:.1%} {vb}"), (row[1], a, f"수정 후 {ra:.1%} {va}")):
            ax.imshow(im)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=90)
    plt.close(fig)

    L = ["| 사진 | 수정 전 ROI | 수정 후 ROI | 새로 들어옴 | 빠짐 |", "|---|---|---|---|---|"]
    for name, rb, ra, vb, va, ad, rm in rows:
        L.append(f"| {name} | 실패 | 실패 | - | - |" if rb is None else
                 f"| {name} | {rb:.1%} {vb} | {ra:.1%} {va} | {ad:.1%} | {rm:.1%} |")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
