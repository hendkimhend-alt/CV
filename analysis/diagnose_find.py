"""
찾기 진단 (E8) — 찾기 방식 비교: Black-hat(지금, 어두운 정도) vs line(Hessian 선 모양).
개발 세트만. ROI auto · 보정 없음. 묶기 · 거르기 없이 "흔적 지도"만 본다.

공정한 비교: 기준값을 바꿔 흔적 양(사진 대비 흔적 픽셀 %)을 여러 단계로 만들고, 흔적 양이 비슷한 지점끼리 비교.
둘 다 1px 선으로 맞춤 (Black-hat은 이중 임계값 → 닫힘 → 뼈대, line은 중심선 → 이중 임계값 → 뼈대).

진단 (선 균열 = 세로 + 가로 정답 박스가 주 대상, 거북등은 참고)
  A. 이어서 덮음: 정답 박스 중 "박스 안 가장 긴 흔적(픽셀 수) ≥ 박스 긴 변 × 0.5" 비율
  B. 질감 구분: "박스 안 가장 긴 흔적 ÷ 긴 변"으로 정답 박스 vs 비교 박스를 가르는 정도 (AUC)
     비교 박스 = 같은 사진 · 같은 높이, 어떤 정답 박스와도 안 겹치고 80% 이상 노면인 같은 크기 박스
  C. 흔적 없음: 정답 박스 안에 흔적이 하나도 없는 비율
  D. 번짐: 흔적 중 가장 큰 덩어리의 몫 (사진별 중앙값)
출력: outputs/analysis/diagnose_find/
"""
import json, os, sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_images
from detect import DEFAULT_CFG, _hysteresis, _kernel, _to_gray, line_centerline, line_score, seeded
from diagnose_d1 import road_mask
from evaluate import transform_gt
from paths import OUTPUT_DIR, RDD_DIR, imread, imwrite
from preprocess import geometry_preprocess, validate_config

OUT = OUTPUT_DIR / "analysis" / "diagnose_find"
LINE_TYPES = {"longitudinal crack": "세로", "transverse crack": "가로"}
REF_TYPES = {"alligator crack": "거북등"}
BH_PCTS = (99.5, 99, 98, 97, 95, 92, 88)        # Black-hat 강한 기준 백분위 (지금 97)
LINE_PCTS = (99.5, 99, 98, 97, 95, 90, 80, 70, 60)      # 중심선 점수 강한 기준 백분위
COVER = 0.5


def bh_traces(gray, cfg):
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _kernel(cfg["blackhat_ksize"]))
    out = {}
    for p in BH_PCTS:
        b, _ = _hysteresis(bh, {**cfg, "crack_hi_pct": p})
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, _kernel(cfg["crack_close_ksize"]))
        out[p] = cv2.ximgproc.thinning(b) > 0
    return out


def line_traces(gray, cfg):
    score, angle = line_score(gray, cfg["line_sigmas"])
    center = line_centerline(score, angle)
    vals = center[center > 0]
    out = {}
    for p in LINE_PCTS:
        hi = max(float(np.percentile(vals, p)), 1e-6) if vals.size else 1.0
        out[p] = cv2.ximgproc.thinning(seeded(center >= hi * cfg["line_lo_ratio"], center >= hi)) > 0
    return out


def longest_in(trace, box):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    H, W = trace.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.nan, True
    crop = trace[y1:y2, x1:x2].astype(np.uint8)
    if not crop.any():
        return 0.0, True
    n, _, st, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    return st[1:, 4].max() / max(x2 - x1, y2 - y1), False


def control_box(gt_all, box, W, road, rng):
    x1, y1, x2, y2 = box
    w = x2 - x1
    if W <= w:
        return None
    for _ in range(30):
        nx = rng.uniform(0, W - w)
        b = (nx, y1, nx + w, y2)
        if any(not (b[2] <= g[0] or b[0] >= g[2] or b[3] <= g[1] or b[1] >= g[3]) for g in gt_all):
            continue
        r = road[max(0, int(y1)):int(y2), int(nx):int(nx + w)]
        if r.size and r.mean() >= 0.8:
            return b
    return None


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, images = list_images("rdd_dev")
    images = images[:limit] if limit else images
    pcfg = validate_config({"roi": "auto"})
    cfg = {**DEFAULT_CFG, "crack_lo_ratio": 0.4}
    rng = np.random.default_rng(20261007)
    # per[method][pct] = {"gt": [(ratio, empty, type)], "ctl": [ratio], "frac": [], "big": []}
    per = {m: {p: defaultdict(list) for p in pcts} for m, pcts in (("blackhat", BH_PCTS), ("line", LINE_PCTS))}
    examples = []
    for i, path in enumerate(images):
        ann = json.loads((RDD_DIR / "ann" / f"{path.name}.json").read_text(encoding="utf-8"))
        objs = [(o["classTitle"], *o["points"]["exterior"][0], *o["points"]["exterior"][1]) for o in ann["objects"]]
        ref, geo = geometry_preprocess(imread(path), pcfg, path.name)
        gray = _to_gray(ref)
        H, W = gray.shape
        boxes = [(o[0], min(o[1], o[3]), min(o[2], o[4]), max(o[1], o[3]), max(o[2], o[4])) for o in objs]
        kept = transform_gt(boxes, geo)
        if len(kept) != len(boxes):
            continue                                   # ROI가 정답을 자른 사진 (드묾) — 상자 순서를 맞출 수 없어 뺌
        gt_all = [tuple(k[1:]) for k in kept]
        targets = [(o[0], tuple(k[1:])) for o, k in zip(objs, kept) if o[0] in LINE_TYPES or o[0] in REF_TYPES]
        road = road_mask(ref)
        ctls = [control_box(gt_all, b, W, road, rng) for t, b in targets if t in LINE_TYPES]
        traces = {"blackhat": bh_traces(gray, cfg), "line": line_traces(gray, cfg)}
        for m, tr in traces.items():
            for p, t in tr.items():
                d = per[m][p]
                d["frac"].append(t.mean())
                n, _, st, _ = cv2.connectedComponentsWithStats(t.astype(np.uint8), connectivity=8)
                if n > 1:
                    d["big"].append(st[1:, 4].max() / st[1:, 4].sum())
                for typ, b in targets:
                    r, empty = longest_in(t, b)
                    if not np.isnan(r):
                        d["gt"].append((r, empty, typ))
                for c in ctls:
                    if c is not None:
                        r, _ = longest_in(t, c)
                        if not np.isnan(r):
                            d["ctl"].append(r)
        if len(examples) < 4 and any(t in LINE_TYPES for t, _ in targets) and rng.random() < 0.15:
            examples.append((ref, [b for t, b in targets if t in LINE_TYPES], traces))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(images)}", flush=True)

    rows = []
    for m in per:
        for p, d in per[m].items():
            line = [(r, e) for r, e, t in d["gt"] if t in LINE_TYPES.values() or t in LINE_TYPES]
            g = np.array([r for r, e, t in d["gt"] if t in LINE_TYPES])
            emp = np.array([e for r, e, t in d["gt"] if t in LINE_TYPES])
            al = np.array([r for r, e, t in d["gt"] if t in REF_TYPES])
            c = np.array(d["ctl"])
            rows.append({"method": m, "pct": p, "frac": float(np.mean(d["frac"])) * 100,
                         "A": float((g >= COVER).mean()), "A_ctl": float((c >= COVER).mean()),
                         "B": auc(g, c), "C": float(emp.mean()), "D": float(np.median(d["big"])),
                         "A_allig": float((al >= COVER).mean()), "g": g, "c": c})
    # 흔적 양이 비슷한 지점끼리 비교 (±25%) — 박스 단위 짝지은 부트스트랩
    brng = np.random.default_rng(20261006)
    comp = []
    for rb in [r for r in rows if r["method"] == "blackhat"]:
        rl = min((r for r in rows if r["method"] == "line"), key=lambda r: abs(np.log(r["frac"] / rb["frac"])))
        if abs(rl["frac"] / rb["frac"] - 1) > 0.25:
            comp.append((rb, rl, None))
            continue
        ng, nc = len(rb["g"]), len(rb["c"])
        dA, dB = [], []
        for _ in range(500):
            ig, ic = brng.integers(0, ng, ng), brng.integers(0, nc, nc)
            dA.append((rl["g"][ig] >= COVER).mean() - (rb["g"][ig] >= COVER).mean())
            dB.append(auc(rl["g"][ig], rl["c"][ic]) - auc(rb["g"][ig], rb["c"][ic]))
        comp.append((rb, rl, (np.percentile(dA, [2.5, 97.5]), np.percentile(dB, [2.5, 97.5]))))

    L = ["# 찾기 진단 (E8) — Black-hat vs line(Hessian), 개발 세트", "",
         f"선 균열 정답 박스 {len(rows[0]['g'])}개 · 비교 박스 {len(rows[0]['c'])}개 · 거북등 {len(per['blackhat'][BH_PCTS[0]]['gt']) - len(rows[0]['g'])}개", "",
         "| 찾기 | 강한 기준 백분위 | 흔적 양 % | A 이어서 덮음 | (비교 박스) | **B 질감 구분 AUC** | C 흔적 없음 | D 번짐 | (거북등 A) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['method']} | {r['pct']} | {r['frac']:.2f} | {r['A']:.3f} | {r['A_ctl']:.3f} | {r['B']:.3f} | {r['C']:.3f} | {r['D']:.2f} | {r['A_allig']:.3f} |")
    L += ["", "## 흔적 양이 비슷한 지점끼리 (line − blackhat, 95% 범위 · ✱ = 0을 안 포함)", "",
          "| Black-hat 기준 | 흔적 양 % (BH / line) | ΔA 이어서 덮음 | ΔB 질감 구분 | ΔD 번짐 |", "|---|---|---|---|---|"]
    for rb, rl, ci in comp:
        if ci is None:
            L.append(f"| {rb['pct']} | {rb['frac']:.2f} / (맞는 지점 없음) | | | |")
            continue
        (a_lo, a_hi), (b_lo, b_hi) = ci
        sa = "✱" if a_lo > 0 or a_hi < 0 else ""
        sb = "✱" if b_lo > 0 or b_hi < 0 else ""
        L.append(f"| {rb['pct']} | {rb['frac']:.2f} / {rl['frac']:.2f} (line {rl['pct']}) | {rl['A'] - rb['A']:+.3f}{sa} [{a_lo:+.3f}, {a_hi:+.3f}] | "
                 f"{rl['B'] - rb['B']:+.3f}{sb} [{b_lo:+.3f}, {b_hi:+.3f}] | {rl['D'] - rb['D']:+.2f} |")
    md = "\n".join(L) + "\n"
    (OUT / "diagnose_find.md").write_text(md, encoding="utf-8")
    print(md)

    # 예시: 흔적 양이 비슷한 한 쌍 (Black-hat 97 = 지금)
    rb = next(r for r in rows if r["method"] == "blackhat" and r["pct"] == 97)
    rl = min((r for r in rows if r["method"] == "line"), key=lambda r: abs(np.log(r["frac"] / rb["frac"])))
    tiles = []
    for ref, bxs, traces in examples:
        row = []
        for title, t in (("original", None), (f"blackhat 97", traces["blackhat"][97]), (f"line {rl['pct']}", traces["line"][rl["pct"]])):
            im = ref.copy()
            if t is not None:
                im = (im * 0.45).astype(np.uint8)
                im[cv2.dilate(t.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (0, 255, 255)
            for b in bxs:
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 2)
            cv2.putText(im, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            f = 420 / im.shape[1]
            row.append(cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
        h = max(x.shape[0] for x in row)
        tiles.append(np.hstack([np.pad(x, ((0, h - x.shape[0]), (0, 0), (0, 0))) for x in row]))
    if tiles:
        imwrite(str(OUT / "examples.jpg"), np.vstack([np.pad(t, ((0, 0), (0, max(x.shape[1] for x in tiles) - t.shape[1]), (0, 0))) for t in tiles]))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
