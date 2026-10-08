"""
두 실행(이상)의 Recall · Precision을 같은 사진끼리 짝지어 비교한다 — 짝지은 부트스트랩 95% 범위.

같은 데이터 세트를 돌린 실행끼리만 비교한다 (사진 이름으로 짝지음). 첫 번째 실행이 기준.
판정 기준은 in50 (src/evaluate.py) — run_pipeline.py의 summary.csv · results/runs.csv와 같은 계산.
사진을 다시 뽑아(복원 추출, 1000번) 두 실행의 차이를 매번 다시 재고, 그 차이의 가운데 95%가
0을 포함하지 않으면 ✱ (= 우연으로 보기 어려운 차이).

실행 (저장소 맨 위 폴더에서):
  python analysis/compare_runs.py run_20261008_005056 run_20261008_134345
  python analysis/compare_runs.py 기준=run_20261008_005056 새것=run_20261008_134345 --kind pothole
  인자 = outputs/pipeline/ 아래 실행 폴더 이름 또는 경로 (이름=폴더로 표의 이름을 붙일 수 있음)
  실행 안에 조건 × 검출기가 여러 개면 --pick 조건/검출기 로 고른다 (예: --pick none/D1hv)
  --split: 개발 세트를 튜닝 394 · 검증 169로 나눠서 (labels/split_rdd_dev.csv) — 여러 값 중 고를 때 튜닝에서 고르고
           검증에서 확인 (10/8 실험과 같은 계산: 2000번 · seed 20261008)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paths import OUTPUT_DIR  # noqa: E402
from stats import row_tags  # noqa: E402

GROUPS = ("all", "blur", "local_illumination", "normal")
SPLIT_CSV = ROOT / "labels" / "split_rdd_dev.csv"


def load(arg, pick):
    name, _, folder = arg.rpartition("=")
    path = Path(folder) if Path(folder).is_dir() else OUTPUT_DIR / "pipeline" / folder
    rows = [r for r in csv.DictReader(open(path / "results.csv", encoding="utf-8-sig"))
            if r["has_gt"] in ("True", "1", "true")]
    combos = list(dict.fromkeys((r["condition"], r["detector"]) for r in rows))
    if pick:
        combos = [c for c in combos if "/".join(c) == pick]
    if len(combos) != 1:
        raise SystemExit(f"error:{path.name} — 조건 × 검출기가 {len(combos)}개 {combos}. --pick 조건/검출기 로 하나를 고르세요")
    rows = [r for r in rows if (r["condition"], r["detector"]) == combos[0]]
    return name or path.name, {r["image"]: r for r in rows}


def stat(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / (tp + fn), tp / (tp + fp), 2 * tp / (2 * tp + fp + fn)


def main():
    p = argparse.ArgumentParser(description="실행끼리 Recall · Precision 짝지은 비교 (첫 번째가 기준)")
    p.add_argument("runs", nargs="+", help="실행 폴더 이름 또는 이름=폴더 (2개 이상)")
    p.add_argument("--kind", default="crack", choices=("crack", "pothole"))
    p.add_argument("--pick", help="조건/검출기 (실행 안에 여러 개일 때)")
    p.add_argument("--split", action="store_true", help="그룹 대신 튜닝 · 검증으로 나눠 비교 (개발 세트 실행만)")
    p.add_argument("--n-boot", type=int, help="부트스트랩 횟수 (기본 1000, --split이면 2000)")
    p.add_argument("--seed", type=int, help="기본 20261006, --split이면 20261008")
    a = p.parse_args()
    n_boot = a.n_boot or (2000 if a.split else 1000)
    seed = a.seed or (20261008 if a.split else 20261006)
    if len(a.runs) < 2:
        raise SystemExit("error:비교할 실행이 2개 이상 필요")
    runs = [load(r, a.pick) for r in a.runs]
    base_name, base = runs[0]
    imgs = sorted(base)
    for name, data in runs[1:]:
        if set(data) != set(base):
            raise SystemExit(f"error:{name}의 사진 목록이 기준과 다름 (같은 데이터 세트 · 같은 --limit인지 확인)")
    if a.split:
        part = {r["image"]: r["split"] for r in csv.DictReader(open(SPLIT_CSV, encoding="utf-8"))}
        missing = [im for im in imgs if im not in part]
        if missing:
            raise SystemExit(f"error:--split은 rdd_dev 실행만 — 분할에 없는 사진 {len(missing)}장 (예: {missing[0]})")
        tags = {im: [part[im]] for im in imgs}
        groups = ("tune", "val")
    else:
        tags = {im: row_tags(base[im]) for im in imgs}
        groups = GROUPS
    k = a.kind
    print(f"종류 {k} · 판정 기준 in50 · 기준 = {base_name} · ✱ = 짝지은 95% 범위가 0을 포함하지 않음"
          + (" · 튜닝 · 검증으로 나눔" if a.split else ""))
    rng = None if a.split else np.random.default_rng(seed)
    for group in groups:
        gi = [im for im in imgs if group in tags[im]]
        if not gi:
            continue
        if a.split:                                 # 묶음마다 새로 (10/8 튜닝 · 검증 비교와 같은 계산)
            rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(gi), size=(n_boot, len(gi)))
        cnt = {n: np.array([[float(d[im][f"{k}_in50_{x}"]) for x in ("tp", "fp", "fn")] for im in gi]) for n, d in runs}
        bb = stat(cnt[base_name][idx].sum(1))
        b0 = stat(cnt[base_name].sum(0))
        print(f"\n[{group} · 사진 {len(gi)}장]")
        for n, _ in runs:
            R, P, F = stat(cnt[n].sum(0))
            line = (f"  {n:24} R {R:.3f}  P {P:.3f}  F1 {F:.3f}  맞힘 {cnt[n][:, 0].sum():4.0f}"
                    f"  가짜/장 {cnt[n][:, 1].sum() / len(gi):5.1f}")
            if n != base_name:
                b = stat(cnt[n][idx].sum(1))
                for lab, j, pt in (("R", 0, R), ("P", 1, P)):
                    lo, hi = np.nanpercentile(b[j] - bb[j], [2.5, 97.5])
                    line += f"  Δ{lab} {pt - b0[j]:+.3f} ({lo:+.3f}~{hi:+.3f}){'✱' if lo > 0 or hi < 0 else ' '}"
            print(line)


if __name__ == "__main__":
    main()
