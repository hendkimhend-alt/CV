"""
균열 최종 구성 B의 가짜 후보를 무작위 150개 뽑아 그림으로 (개발 세트 · 분석만, 10/10).

가짜 = 같은 종류 정답 박스 안에 면적 50% 이상 들어가지 않은 후보 (in50 — EVALUATION.md 2절).
무작위 150개: seed 20261010 · 가짜 전체(균열 + 포트홀)에서 고르게 — 사진당 제한 없음
(가짜가 몰린 사진의 비중이 표본에도 그대로 들어가게).
그림 한 칸 = 왼쪽: 사진 전체(빨간 박스가 그 가짜) · 오른쪽: 박스 주변 확대 (박스의 3배, 최소 160px)
번호 = results/analysis/b_fakes.csv의 id. 종류(kind)는 사람이 그림을 보고 채운다:
  A 노면 밖 · B 보닛 · 차체 · 유리 반사 · C 차선 · 노면 페인트 · D 그림자 경계 · E 노면 질감(골재, 뚜렷한 선 없음)
  F 이음매 · 패치 · 포장 경계 · G 물체(맨홀 · 배수구 · 차 등) · H 진짜 균열처럼 보임(정답 표시 없음)
  I 정답 박스에 걸침 · J 판별 어려움

실행: python analysis/b_fake_sample.py --run run_20261010_195301 --sheet 폴더
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import list_images, load_gt  # noqa: E402
from evaluate import _area, _inter, _xyxy  # noqa: E402
from paths import OUTPUT_DIR, imread  # noqa: E402

OUT = ROOT / "results" / "analysis" / "b_fakes.csv"
N, SEED, PER_SHEET = 150, 20261010, 25
KINDS = {"A": "노면 밖", "B": "보닛 · 차체 · 유리 반사", "C": "차선 · 노면 페인트", "D": "그림자 경계",
         "E": "노면 질감", "F": "이음매 · 패치 · 포장 경계", "G": "물체", "H": "진짜 균열처럼 보임 (정답 없음)",
         "I": "정답 박스에 걸침", "J": "판별 어려움"}


def collect(run):
    dets = json.load(open(OUTPUT_DIR / "pipeline" / run / "detections.json"))
    name, _, images = list_images("rdd_dev")
    gt_of = load_gt("rdd_dev", name)
    fakes = []
    for path in images:
        gts = gt_of(path) or []
        for d in dets.get(path.name, []):
            b = _xyxy(d["bbox"])
            same = [tuple(g[1:]) for g in gts if g[0] == d["type"]]
            fr = max((_inter(b, g) / max(_area(b), 1.0) for g in same), default=0.0)
            if fr < 0.5:
                fakes.append({"image": path.name, "type": d["type"], "x1": round(b[0]), "y1": round(b[1]),
                              "x2": round(b[2]), "y2": round(b[3]), "gt_overlap": round(fr, 3),
                              "n_fake_in_image": 0})
    cnt = {}
    for f in fakes:
        cnt[f["image"]] = cnt.get(f["image"], 0) + 1
    for f in fakes:
        f["n_fake_in_image"] = cnt[f["image"]]
    return fakes, {p.name: p for p in images}


def tile(img, f, size=300):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]
    t = max(2, W // 250)
    full = img.copy()
    cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), t + 1)
    s = size / max(H, W)
    full = cv2.resize(full, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(80, 1.5 * max(x2 - x1, y2 - y1))
    a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
    c, d = int(min(cx + half, W)), int(min(cy + half, H))
    zoom = img[b:d, a:c].copy()
    cv2.rectangle(zoom, (x1 - a, y1 - b), (x2 - a, y2 - b), (0, 0, 255), max(1, (c - a) // 150))
    z = size / max(zoom.shape[:2])
    zoom = cv2.resize(zoom, (int(zoom.shape[1] * z), int(zoom.shape[0] * z)),
                      interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA)
    out = np.full((size, 2 * size + 8, 3), 255, np.uint8)
    out[:full.shape[0], :full.shape[1]] = full
    out[:zoom.shape[0], size + 8:size + 8 + zoom.shape[1]] = zoom
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_road_pipeline.py 실행 이름 (detections.json이 있는 것)")
    ap.add_argument("--sheet", required=True, help="그림 저장 폴더")
    a = ap.parse_args()
    fakes, paths = collect(a.run)
    pick = [fakes[i] for i in np.random.default_rng(SEED).choice(len(fakes), N, replace=False)]
    pick.sort(key=lambda f: (f["image"], f["y1"]))
    sheet = Path(a.sheet)
    sheet.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(pick, 1):
        f["id"] = i
    cache = {}
    for s in range(0, N, PER_SHEET):
        tiles = []
        for f in pick[s:s + PER_SHEET]:
            if f["image"] not in cache:
                cache = {f["image"]: imread(paths[f["image"]])}
            t = tile(cache[f["image"]], f)
            lab = np.full((34, t.shape[1], 3), 255, np.uint8)
            cv2.putText(lab, f"#{f['id']} {f['type']} ov{f['gt_overlap']:.2f} n{f['n_fake_in_image']}",
                        (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            tiles.append(np.vstack([lab, t]))
        cols = 5
        rows = [np.hstack(tiles[r:r + cols] + [np.full_like(tiles[0], 255)] * (cols - len(tiles[r:r + cols])))
                for r in range(0, len(tiles), cols)]
        page = np.vstack([np.vstack([r, np.full((14, r.shape[1], 3), 255, np.uint8)]) for r in rows])
        cv2.imwrite(str(sheet / f"b_fakes_{s // PER_SHEET + 1}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 88])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "image", "type", "x1", "y1", "x2", "y2", "gt_overlap", "n_fake_in_image"]
    with open(OUT, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + ["kind", "kind_name", "labeler"], lineterminator="\n")
        w.writeheader()
        for f in pick:
            w.writerow({**{k: f[k] for k in cols}, "kind": "", "kind_name": "", "labeler": ""})
    print(f"가짜 {len(fakes)}개 중 {N}개 → {OUT.relative_to(ROOT)} · 그림 {sheet}")
    print("균열", sum(f["type"] == "crack" for f in pick), "· 포트홀", sum(f["type"] == "pothole" for f in pick),
          "· 정답에 걸침(ov>0)", sum(f["gt_overlap"] > 0 for f in pick))


if __name__ == "__main__":
    main()
