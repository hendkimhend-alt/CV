"""
RDD 804장을 개발(dev) 70% / 테스트(test) 30%로 나눈다 → labels/split_rdd.csv (팀 공용, 한 번 만들고 고정).

막 나누지 않고 층(stratum)별로 같은 비율이 되게 나눈다. 층 = 출처 × 품질 그룹 × 손상 유무
- 출처(7개)  = 카메라 시점 (드론·오토바이 vs 차량) — ROI 실험에서 결과가 정반대로 갈린 축
- 품질 그룹  = 흐림 / 국소 조도 / 정상 (P0 노면 영역 기준, run_pipeline과 같은 분류). 흐림+국소 조도 1장은 흐림으로
- 손상 유무  = 균열·포트홀 라벨이 하나라도 있나 (없는 사진 296장 = 오검출만 재는 음성 사진)
층 안에서는 seed 고정 셔플 후 계통 추출 → 전체 30%가 층마다 고르게 퍼진다 (작은 층도 한쪽에 몰리지 않게).
번호가 가까운 사진(차이 ≤ 2, 23쌍)은 육안 확인 결과 다른 장면 → 따로 묶지 않음.

실행: python analysis/make_split.py   (이미 있으면 덮어쓰지 않음, --force로 재생성)
"""
import argparse, csv, json, os, random, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data import RDD_TYPES, list_images
from metrics import measure_quality
from paths import LABELS_DIR, RDD_DIR, imread
from preprocess import classify_quality, geometry_preprocess, validate_config

SEED = 20261006
TEST_FRAC = 0.3
OUT = LABELS_DIR / "split_rdd.csv"


def describe(path, cfg):
    ann = json.loads((RDD_DIR / "ann" / f"{path.name}.json").read_text(encoding="utf-8"))
    kinds = Counter(RDD_TYPES.get(o["classTitle"]) for o in ann["objects"])
    ref, _ = geometry_preprocess(imread(path), cfg, path.name)
    tags = classify_quality(measure_quality(ref))
    group = "blur" if "blur" in tags else tags[0]
    return {"image": path.name, "source": re.sub(r"_\d+$", "", path.stem), "group": group,
            "has_damage": int(kinds["crack"] + kinds["pothole"] > 0),
            "n_crack": kinds["crack"], "n_pothole": kinds["pothole"]}


def split(rows):
    strata = defaultdict(list)
    for r in rows:
        strata[(r["source"], r["group"], r["has_damage"])].append(r)
    rng = random.Random(SEED)
    acc = rng.random()                       # 계통 추출 시작점 — 층을 넘어 이어져 전체 비율이 정확히 맞는다
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda r: r["image"])
        rng.shuffle(members)
        for r in members:
            acc += TEST_FRAC
            r["split"] = "test" if acc >= 1 else "dev"
            acc -= int(acc)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if OUT.exists() and not a.force:
        print(f"이미 있음: {OUT} (재생성은 --force)")
        return
    cfg = validate_config({})
    _, _, images = list_images("rdd")
    rows = split([describe(p, cfg) for p in images])
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image"]))
    # 균형 확인
    for col in ("source", "group", "has_damage"):
        c = Counter((r[col], r["split"]) for r in rows)
        print(f"\n[{col}]")
        for k in sorted({r[col] for r in rows}):
            d, t = c[(k, "dev")], c[(k, "test")]
            print(f"  {str(k):18} dev {d:4}  test {t:4}  test 비율 {t / (d + t):.2f}")
    for kind in ("n_crack", "n_pothole"):
        d = sum(r[kind] for r in rows if r["split"] == "dev")
        t = sum(r[kind] for r in rows if r["split"] == "test")
        print(f"  {kind:18} dev {d:4}  test {t:4}  test 비율 {t / (d + t):.2f}")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
