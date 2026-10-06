"""
F1 오차 범위 — 사진 단위 부트스트랩 (짝지은 비교).

- 같은 그룹의 사진들을 복원 추출(B번) → 뽑힌 사진들의 TP·FP·FN 합으로 F1을 다시 계산 → 2.5 · 97.5 백분위 = 95% 범위
- 짝지은 비교: 한 번 뽑은 사진 묶음을 모든 조건(전처리 × 검출기)에 똑같이 쓴다
  → "기준 조건 대비 F1 차이"의 95% 범위가 0을 포함하지 않으면 의미 있는 차이(significant)
- 판정 기준 고르기 규칙("조건 간 차이를 오차 범위보다 크게 구분하는 기준 중 가장 엄격한 것")에 쓰는 숫자도 여기서 나온다
"""
from collections import defaultdict

import numpy as np

from evaluate import CRITERIA, KINDS

SEED = 20261006


def row_tags(r):
    """한 행이 속하는 집계 묶음: 전체 + 품질 그룹 + 시점(있으면 vp_<시점>)."""
    tags = ["all", *r["group"].split(";")]
    if r.get("viewpoint"):
        tags.append(f"vp_{r['viewpoint']}")
    return tags


def _f1(tp, fp, fn):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)


def bootstrap(rows, reference, n_boot=1000):
    """rows: run_pipeline의 사진×조건×검출기 행 (정답 있는 것만 씀), reference: (조건, 검출기)
    → (ci, compare)
       ci[(조건, 검출기, 그룹, 종류, 기준)] = (F1 하한, F1 상한)
       compare: 기준 조건 대비 F1 차이 행 목록"""
    rows = [r for r in rows if r["has_gt"]]
    configs = sorted({(r["condition"], r["detector"]) for r in rows})
    by_group = defaultdict(set)
    for r in rows:
        for g in row_tags(r):
            by_group[g].add(r["image"])
    table = {(r["image"], r["condition"], r["detector"]): r for r in rows}
    rng = np.random.default_rng(SEED)
    ci, compare = {}, []
    for group, imgs in sorted(by_group.items()):
        imgs = sorted(imgs)
        idx = rng.integers(0, len(imgs), size=(n_boot, len(imgs)))
        for kind in KINDS:
            for c in CRITERIA:
                boot, point = {}, {}
                for cfg in configs:
                    cnt = np.array([[table[(im, *cfg)][f"{kind}_{c}_{k}"] for k in ("tp", "fp", "fn")]
                                    for im in imgs], dtype=float)
                    point[cfg] = float(_f1(*cnt.sum(0)))
                    boot[cfg] = _f1(*(cnt[idx].sum(1).T))
                    if np.isfinite(boot[cfg]).any():
                        ci[(*cfg, group, kind, c)] = tuple(float(v) for v in np.nanpercentile(boot[cfg], [2.5, 97.5]))
                if reference not in boot:
                    continue
                for cfg in configs:
                    d = boot[cfg] - boot[reference]
                    if cfg == reference or not np.isfinite(d).any():
                        continue
                    lo, hi = np.nanpercentile(d, [2.5, 97.5])
                    compare.append({"group": group, "kind": kind, "criterion": c,
                                    "condition": cfg[0], "detector": cfg[1],
                                    "reference": f"{reference[0]}/{reference[1]}", "n_images": len(imgs),
                                    "f1": round(point[cfg], 4), "f1_reference": round(point[reference], 4),
                                    "delta": round(point[cfg] - point[reference], 4),
                                    "delta_lo": round(float(lo), 4), "delta_hi": round(float(hi), 4),
                                    "significant": bool(lo > 0 or hi < 0)})
    return ci, compare
