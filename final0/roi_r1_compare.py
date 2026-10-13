"""
R1 사진별 정규화 — 결정 규칙대로 비교 (docs/experiments_roi.md R1).

입력: outputs/roi_r1/<이름>/run_*/results.csv (run_final.py --no-images)
출력: outputs/roi_r1/compare.md
실행: python roi_r1_compare.py
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "roi_r1"
NAMES = ["base", "G", "T", "GT", "GTdown"]
LABEL = {"base": "R1-0 기준", "G": "R1-G 기울기", "T": "R1-T 질감", "GT": "R1-GT 둘 다", "GTdown": "R1-GT↓ 줄이기만"}
KINDS = ("crack", "pothole")


def load(name):
    fp = sorted((BASE / name).glob("run_*/results.csv"))[-1]
    return {r["image"]: r for r in csv.DictReader(open(fp, encoding="utf-8-sig"))}


def per_image(rows, names, kind):
    """사진마다 (tp, fp, fn, 후보 수)."""
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else 0.0
    return np.array([[f(rows[n], f"{kind}_in50_tp"), f(rows[n], f"{kind}_in50_fp"), f(rows[n], f"{kind}_in50_fn"),
                      f(rows[n], f"n_{kind}")] for n in names])


def metrics(a):
    tp, fp, fn, nd = a.sum(0)
    return {"R": tp / (tp + fn) if tp + fn else float("nan"), "P": (nd - fp) / nd if nd else float("nan"),
            "fake": fp / len(a), "tp": tp}


def boot(a, b, rng, what):
    ds = []
    for _ in range(2000):
        i = rng.integers(0, len(a), len(a))
        ds.append(metrics(a[i])[what] - metrics(b[i])[what])
    return np.percentile(ds, [2.5, 97.5])


def main():
    split = {r["image"]: r["split"] for r in csv.DictReader(open(ROOT / "labels" / "split_rdd_dev.csv", encoding="utf-8"))}
    runs = {n: load(n) for n in NAMES}
    imgs = sorted(runs["base"])
    tune = [n for n in imgs if split.get(n) == "tune"]
    val = [n for n in imgs if split.get(n) == "val"]
    M = {(n, s, k): metrics(per_image(runs[n], ns, k)) for n in NAMES for s, ns in (("tune", tune), ("val", val), ("all", imgs)) for k in KINDS}
    fmt = lambda m: f"{m['R']:.3f} · {m['P']:.3f} · {m['fake']:.2f}"
    L = ["# R1 사진별 정규화 — 결과 (개발 563 · 튜닝 394 / 검증 169)", "",
         "균열 · 포트홀 각각 R · P · 가짜/장 (in50)", "",
         "| 안 | 마스크 바뀐 사진 (노면 비율 ±0.5%p 넘게) | 검증 FAIL | 노면 비율 평균 | 튜닝 균열 | 튜닝 포트홀 | 검증 균열 | 검증 포트홀 |",
         "|---|---|---|---|---|---|---|---|"]
    rr = lambda r: float(r["road_ratio"]) if r["road_ratio"] not in ("", None) else 0.0
    for n in NAMES:
        changed = sum(abs(rr(runs[n][i]) - rr(runs["base"][i])) > 0.005 for i in imgs)
        fail = sum(runs[n][i]["mask_check"] == "FAIL" for i in imgs)
        ratio = np.mean([rr(runs[n][i]) for i in imgs])
        L.append(f"| {LABEL[n]} | {changed} | {fail} | {ratio:.3f} | {fmt(M[n, 'tune', 'crack'])} | {fmt(M[n, 'tune', 'pothole'])} | "
                 f"{fmt(M[n, 'val', 'crack'])} | {fmt(M[n, 'val', 'pothole'])} |")
    b = lambda s, k: M["base", s, k]
    ok = lambda n, s: (b(s, "crack")["R"] - M[n, s, "crack"]["R"] <= 0.01
                       and M[n, s, "crack"]["fake"] - b(s, "crack")["fake"] <= 0.2
                       and M[n, s, "pothole"]["fake"] - b(s, "pothole")["fake"] <= 0.1)
    L += ["", "**1 · 2단계 (튜닝)** 조건: 균열 R 하락 ≤ 0.01 · 균열 가짜/장 증가 ≤ 0.2 · 포트홀 가짜/장 증가 ≤ 0.1", ""]
    for n in NAMES[1:]:
        dR = M[n, "tune", "crack"]["R"] - b("tune", "crack")["R"]
        dF = M[n, "tune", "crack"]["fake"] - b("tune", "crack")["fake"]
        dP = M[n, "tune", "pothole"]["fake"] - b("tune", "pothole")["fake"]
        L.append(f"- {'✅' if ok(n, 'tune') else '❌'} {LABEL[n]}: 균열 R {dR:+.3f} · 균열 가짜/장 {dF:+.2f} · 포트홀 가짜/장 {dP:+.2f}")
    cands = [n for n in NAMES[1:] if ok(n, "tune")]
    if not cands:
        L += ["", "- **결론 (규칙대로): R1 안 넣음 — 튜닝에서 조건을 만족하는 안 없음**"]
    else:
        pick = max(cands, key=lambda n: (M[n, "tune", "crack"]["R"], -M[n, "tune", "crack"]["fake"]))
        L += ["", f"- 고른 안: **{LABEL[pick]}**", "", "**3단계 (검증 169)**", "",
              "| 지표 | 기준 | 고른 안 | 차이 (95% 범위 · 참고) |", "|---|---|---|---|"]
        rng = np.random.default_rng(20261013)
        for k in KINDS:
            a, c = per_image(runs[pick], val, k), per_image(runs["base"], val, k)
            for w, nm in (("R", "R"), ("P", "P"), ("fake", "가짜/장")):
                lo, hi = boot(a, c, rng, w)
                L.append(f"| {k} {nm} | {b('val', k)[w]:.3f} | {M[pick, 'val', k][w]:.3f} | "
                         f"{M[pick, 'val', k][w] - b('val', k)[w]:+.3f} ({lo:+.3f} ~ {hi:+.3f}) |")
        L += ["", f"- **결론 (규칙대로): {'R1 넣음 (' + LABEL[pick] + ')' if ok(pick, 'val') else 'R1 안 넣음 — 검증에서 조건 실패'}**"]
    (BASE / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
