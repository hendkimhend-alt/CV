"""
개발 세트 563장을 튜닝(tune) 70% / 검증(val) 30%로 다시 나눈다 → labels/split_rdd_dev.csv (한 번 만들고 고정).

왜: 테스트 세트는 최종 설정(G2) 확인에 이미 한 번 썼다 (10/8). 그 뒤의 검출기 개선은 테스트를 다시 보지 않고 고르기 위해
개발 세트 안에서 "고르는 사진(tune)"과 "처음 보는 사진처럼 확인하는 사진(val)"을 나눈다. 테스트 241장은 그대로 둔다.
방법: make_split.py와 같다 — 층(출처 × 품질 그룹 × 손상 유무)별 같은 비율 · seed 고정 셔플 · 층을 넘어 이어지는 계통 추출.
층 정보는 labels/split_rdd.csv의 것을 그대로 쓴다.

실행: python analysis/make_devsplit.py   (이미 있으면 덮어쓰지 않음, --force로 재생성)
"""
import argparse, csv, os, random, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import LABELS_DIR

SEED = 20261008
VAL_FRAC = 0.3
SRC = LABELS_DIR / "split_rdd.csv"
OUT = LABELS_DIR / "split_rdd_dev.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if OUT.exists() and not a.force:
        print(f"이미 있음: {OUT} (재생성은 --force)")
        return
    rows = [r for r in csv.DictReader(SRC.open(encoding="utf-8")) if r["split"] == "dev"]
    strata = defaultdict(list)
    for r in rows:
        strata[(r["source"], r["group"], r["has_damage"])].append(r)
    rng = random.Random(SEED)
    acc = rng.random()
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda r: r["image"])
        rng.shuffle(members)
        for r in members:
            acc += VAL_FRAC
            r["split"] = "val" if acc >= 1 else "tune"
            acc -= int(acc)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image"]))
    for col in ("source", "group", "has_damage"):
        c = Counter((r[col], r["split"]) for r in rows)
        print(f"\n[{col}]")
        for k in sorted({r[col] for r in rows}):
            t, v = c[(k, "tune")], c[(k, "val")]
            print(f"  {str(k):18} tune {t:4}  val {v:4}  val 비율 {v / (t + v):.2f}")
    for kind in ("n_crack", "n_pothole"):
        t = sum(int(r[kind]) for r in rows if r["split"] == "tune")
        v = sum(int(r[kind]) for r in rows if r["split"] == "val")
        print(f"  {kind:18} tune {t:4}  val {v:4}  val 비율 {v / (t + v):.2f}")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
