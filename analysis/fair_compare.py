"""
공정 비교 (10/12) — 찾기만 Black-hat ↔ Hessian, 기준값을 여러 점 돌린 곡선으로 "같은 가짜/장에서 Recall" 비교.
결정 규칙: results/experiments.md "공정 비교 — 찾기만 Black-hat ↔ Hessian" (실험 전에 적음).

- 곡선: results/runs.csv의 설정 이름 fair[0]_bh_* · fair[0]_h* (+ Hessian × 1.0 = 기존 실행)
- 비교 수준: 가짜/장 2 · 4 · 6 (두 곡선이 모두 닿는 것만) — 곡선 위 직선 보간 (가짜/장 순으로 정렬)
- 유의: 가짜/장 차이 ≤ 1.0인 가장 가까운 두 실행을 같은 사진끼리 짝지어 Recall 차이의 95% 범위 (부트스트랩 2000번)
- 그림: 조건마다 Recall-가짜/장 곡선

실행: python analysis/fair_compare.py --stage 2   (2 = 버리기 다 붙임 · 1 = 맞춘 버리기 끔)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from paths import OUTPUT_DIR  # noqa: E402

LEVELS = (2.0, 4.0, 6.0)
H10 = {(2, "M"): "b_road", (2, "F"): "t_c10_post"}       # 2단계 Hessian × 1.0 = 이미 있는 실행


def load_rows(run):
    rows = [r for r in csv.DictReader(open(OUTPUT_DIR / "pipeline" / run / "results.csv", encoding="utf-8-sig"))
            if r["has_gt"] in ("True", "1", "true")]
    return {r["image"]: r for r in rows}


def curve(runs_csv, prefix, cond, family, stage):
    pts = []
    for r in runs_csv:
        c = r["config"]
        ok = c.startswith(prefix) and c.endswith("_" + cond) and (("_bh_" in c) == (family == "bh"))
        if family == "h" and (stage, cond) in H10 and c == H10[(stage, cond)]:
            ok = True
        if ok:
            pts.append({"config": c, "run": r["run"], "R": float(r["crack_recall"]), "P": float(r["crack_precision"]),
                        "F1": float(r["crack_f1"]), "fake": float(r["crack_fake_per_img"]),
                        "frag": float(r["crack_frag_per_hit"] or 0)})
    return sorted(pts, key=lambda p: p["fake"])


def interp1(pts, x):
    """한 기준 계열(가짜/장 순) 위에서 x의 Recall — 직선 보간, 범위 밖이면 None."""
    pts = sorted(pts, key=lambda p: p["fake"])
    if len(pts) < 1 or x < pts[0]["fake"] or x > pts[-1]["fake"]:
        return None
    for p, q in zip(pts, pts[1:]):
        if p["fake"] <= x <= q["fake"]:
            return p["R"] if q["fake"] == p["fake"] else p["R"] + (x - p["fake"]) / (q["fake"] - p["fake"]) * (q["R"] - p["R"])
    return pts[0]["R"]


def interp(pts, x):
    """Black-hat은 두 기준 계열(고정 · 사진마다 %) 중 더 높은 쪽 — Black-hat에 유리하게."""
    fams = {}
    for p in pts:
        fam = "abs" if "_abs" in p["config"] else "pct" if "_pct" in p["config"] else "h"
        fams.setdefault(fam, []).append(p)
    vals = [v for v in (interp1(f, x) for f in fams.values()) if v is not None]
    return max(vals) if vals else None


def paired(run_a, run_b, n_boot=2000, seed=20261012):
    A, B = load_rows(run_a), load_rows(run_b)
    ims = sorted(set(A) & set(B))
    ta = np.array([[float(A[i]["crack_in50_tp"]), float(A[i]["crack_in50_fn"])] for i in ims])
    tb = np.array([[float(B[i]["crack_in50_tp"]), float(B[i]["crack_in50_fn"])] for i in ims])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ims), size=(n_boot, len(ims)))
    ra = ta[idx, 0].sum(1) / ta[idx].sum((1, 2)); rb = tb[idx, 0].sum(1) / tb[idx].sum((1, 2))
    d = ra - rb
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(ta[:, 0].sum() / ta.sum() - tb[:, 0].sum() / tb.sum()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--plot", help="그림 저장 경로 (png)")
    a = ap.parse_args()
    runs_csv = list(csv.DictReader(open(ROOT / "results" / "runs.csv", encoding="utf-8-sig")))
    prefix = "fair0_" if a.stage == 1 else "fair_"
    out = {}
    for cond in ("M", "F"):
        bh, h = curve(runs_csv, prefix, cond, "bh", a.stage), curve(runs_csv, prefix, cond, "h", a.stage)
        out[cond] = (bh, h)
        print(f"\n=== {a.stage}단계 · 조건 {cond} ({'노면 마스크만' if cond == 'M' else '최종 전처리'}) ===")
        for name, pts in (("Black-hat", bh), ("Hessian", h)):
            print(f"  {name}:")
            for p in pts:
                print(f"    {p['config']:28} 가짜/장 {p['fake']:5.2f}  R {p['R']:.3f}  P {p['P']:.3f}  F1 {p['F1']:.3f}  조각/맞힘 {p['frag']:.2f}")
        print("  같은 가짜/장에서 Recall (직선 보간):")
        for x in LEVELS:
            rb, rh = interp(bh, x), interp(h, x)
            s = f"    가짜 {x:.0f}/장: Black-hat {('-' if rb is None else f'{rb:.3f}'):>6} · Hessian {('-' if rh is None else f'{rh:.3f}'):>6}"
            if rb is not None and rh is not None:
                s += f"  차이 {rh - rb:+.3f}"
                pb = min(bh, key=lambda p: abs(p["fake"] - x)); ph = min(h, key=lambda p: abs(p["fake"] - x))
                if abs(pb["fake"] - ph["fake"]) <= 1.0:
                    dlt, lo, hi = paired(ph["run"], pb["run"])
                    s += f"  | 가까운 두 실행 {ph['config']} vs {pb['config']} (가짜 {ph['fake']:.1f} vs {pb['fake']:.1f}): ΔR {dlt:+.3f} ({lo:+.3f}~{hi:+.3f}){'✱' if lo > 0 or hi < 0 else ''}"
                else:
                    s += "  | 가짜/장 차이 ≤ 1.0인 실행 쌍 없음"
            print(s)
        for name, pts in (("Black-hat", bh), ("Hessian", h)):
            if pts:
                m = max(pts, key=lambda p: p["R"]); f = max(pts, key=lambda p: p["F1"])
                print(f"  {name} 최고 Recall {m['R']:.3f} (가짜 {m['fake']:.1f}/장) · 최고 F1 {f['F1']:.3f} ({f['config']})")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
        plt.rcParams["font.family"] = "Apple SD Gothic Neo"
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        for ax, cond in zip(axes, ("M", "F")):
            bh, h = out[cond]
            for pts, lab, col, mk in ((bh, "Black-hat (찾기만 바꿈)", "#dc2626", "o"), (h, "Hessian", "#2563eb", "s")):
                for fam in ("abs", "pct", ""):
                    sub = [p for p in pts if (f"_{fam}" in p["config"]) == bool(fam)] if fam else ([] if lab.startswith("Black") else pts)
                    if lab.startswith("Black") and fam:
                        sub = sorted(sub, key=lambda p: p["fake"])
                        ax.plot([p["fake"] for p in sub], [p["R"] for p in sub], marker=mk, color=col, alpha=1 if fam == "abs" else 0.55,
                                ls="-" if fam == "abs" else "--", label=f"{lab} · {'고정 기준' if fam == 'abs' else '사진마다 상위 %'}")
                    elif not lab.startswith("Black") and not fam:
                        ax.plot([p["fake"] for p in sub], [p["R"] for p in sub], marker=mk, color=col, label=lab)
            for x in LEVELS:
                ax.axvline(x, color="#cbd5e1", lw=1, zorder=0)
            ax.set_title(f"{'노면 마스크만 (M)' if cond == 'M' else '최종 전처리 (F)'}")
            ax.set_xlabel("가짜/장"); ax.grid(alpha=0.3)
        axes[0].set_ylabel("Recall"); axes[0].legend(loc="lower right", fontsize=9)
        fig.suptitle(f"공정 비교 {a.stage}단계 — {'맞춘 버리기 끔 (찾기 + 잇기만)' if a.stage == 1 else '버리기 다 붙임 (실전 조건)'} · 같은 가짜/장에서 위에 있는 곡선이 낫다")
        fig.tight_layout(); fig.savefig(a.plot, dpi=130)
        print("그림 →", a.plot)


if __name__ == "__main__":
    main()
