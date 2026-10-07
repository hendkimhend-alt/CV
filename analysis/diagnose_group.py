"""
묶기 진단 — 개발 세트만. ROI auto · 보정 없음 · 찾기 = Hessian (고정 기준 32.5, 약한 기준 × 0.4).

G1 (틈 · 방향): 정답 선 균열 박스 안 조각마다, 같은 박스 안 다른 조각까지 가장 가까운 끝점 거리와 방향 차이.
    손상 없는 비교 박스(질감)도 같이 → 균열 조각과 질감 조각이 틈 · 방향으로 갈리는지
G2 (잇기 거리): 잇기 없음 · 10 · 20 · 30 · 45px (각도 30°)
    M  맞출 수 있는 정답 = 박스 안에 50% 이상(상자 기준) 들어가면서 박스 긴 변의 절반 이상을 가로지르는 묶음이 있는 비율
    M 비교 박스 = 손상 없는 비교 박스에서 같은 조건의 묶음 비율 · M′ = M − M 비교 박스
    넘침 = 박스를 절반 이상 가로지르는 묶음은 있는데 박스 밖으로 넘쳐 50% 안을 못 지킨 비율
    사진당 긴 묶음 = 어떤 정답 박스에도 50% 이상 안 들어간 길이 60px 이상 묶음 수
출력: outputs/analysis/diagnose_group/
"""
import json, os, sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_images
from detect import DEFAULT_CFG, _to_gray, line_trace, link_fragments
from diagnose_d1 import road_mask
from diagnose_find import LINE_TYPES, control_box
from evaluate import _area, _inter, transform_gt
from paths import OUTPUT_DIR, RDD_DIR, imread, imwrite
from preprocess import geometry_preprocess, validate_config

OUT = OUTPUT_DIR / "analysis" / "diagnose_group"
DISTS = (0, 10, 20, 30, 45)
CFG = {**DEFAULT_CFG, "crack_find": "line", "line_hi_abs": 32.5, "link_angle": 30}
NB = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int16)
LONG = 60


def clip(b, W, H):
    return max(0, int(b[0])), max(0, int(b[1])), min(W, int(np.ceil(b[2]))), min(H, int(np.ceil(b[3])))


def pieces(trace, box):
    """박스 안 조각(5px 이상)마다 (끝점 좌표들, 방향 각도[0,180))."""
    H, W = trace.shape
    x1, y1, x2, y2 = clip(box, W, H)
    crop = (trace[y1:y2, x1:x2] > 0).astype(np.uint8)
    if crop.sum() < 5:
        return []
    nb = cv2.filter2D(crop, cv2.CV_16S, NB, borderType=cv2.BORDER_CONSTANT)
    n, lab = cv2.connectedComponents(crop, connectivity=8)
    out = []
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        if len(xs) < 5:
            continue
        e = (crop[ys, xs] == 1) & (nb[ys, xs] == 1)
        ends = np.stack([xs[e], ys[e]], 1) if e.any() else np.stack([xs, ys], 1)
        c = np.cov(np.stack([xs, ys]).astype(float))
        w, v = np.linalg.eigh(c)
        ang = np.degrees(np.arctan2(v[1, 1], v[0, 1])) % 180
        out.append((ends.astype(float), ang))
    return out


def gaps(ps):
    """조각마다 가장 가까운 다른 조각: (끝점 거리, 방향 차이)."""
    res = []
    for i, (ei, ai) in enumerate(ps):
        best = None
        for j, (ej, aj) in enumerate(ps):
            if i == j:
                continue
            d = np.sqrt(((ei[:, None, :] - ej[None, :, :]) ** 2).sum(-1)).min()
            if best is None or d < best[0]:
                da = abs(ai - aj) % 180
                best = (d, min(da, 180 - da))
        if best:
            res.append(best)
    return res


def groups(binary):
    n, lab, st, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    return lab, [(st[k, 0], st[k, 1], st[k, 0] + st[k, 2], st[k, 1] + st[k, 3], st[k, 4]) for k in range(1, n)]


def box_state(lab, gs, box, W, H):
    """정답(또는 비교) 박스 하나: 'M'(맞출 수 있음) / '넘침' / '짧음'."""
    x1, y1, x2, y2 = clip(box, W, H)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    side = max(x2 - x1, y2 - y1)
    crop = lab[y1:y2, x1:x2]
    over = False
    for k in np.unique(crop[crop > 0]):
        ys, xs = np.nonzero(crop == k)
        span = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / side
        if span < 0.5:
            continue
        g = gs[k - 1]
        if _inter(g[:4], box) >= 0.5 * max(_area(g[:4]), 1.0):
            return "M"
        over = True
    return "넘침" if over else "짧음"


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, images = list_images("rdd_dev")
    images = images[:limit] if limit else images
    pcfg = validate_config({"roi": "auto"})
    rng = np.random.default_rng(20261007)
    g1 = {"gt": [], "ctl": []}
    st = {d: {"gt": [], "ctl": [], "long": []} for d in DISTS}
    frac = []
    for i, path in enumerate(images):
        ann = json.loads((RDD_DIR / "ann" / f"{path.name}.json").read_text(encoding="utf-8"))
        objs = [(o["classTitle"], *o["points"]["exterior"][0], *o["points"]["exterior"][1]) for o in ann["objects"]]
        ref, geo = geometry_preprocess(imread(path), pcfg, path.name)
        boxes = [(o[0], min(o[1], o[3]), min(o[2], o[4]), max(o[1], o[3]), max(o[2], o[4])) for o in objs]
        kept = transform_gt(boxes, geo)
        if len(kept) != len(boxes):
            continue
        H, W = ref.shape[:2]
        gt_all = [tuple(k[1:]) for k in kept]
        lines = [tuple(k[1:]) for o, k in zip(objs, kept) if o[0] in LINE_TYPES]
        road = road_mask(ref)
        ctls = [c for c in (control_box(gt_all, b, W, road, rng) for b in lines) if c is not None]
        trace, _, _ = line_trace(_to_gray(ref), CFG)
        frac.append((trace > 0).mean())
        for b in lines:
            g1["gt"] += gaps(pieces(trace, b))
        for c in ctls:
            g1["ctl"] += gaps(pieces(trace, c))
        for d in DISTS:
            binary = link_fragments(trace, {**CFG, "link_dist": d}) if d else trace
            lab, gs = groups(binary)
            st[d]["gt"] += [box_state(lab, gs, b, W, H) for b in lines]
            st[d]["ctl"] += [box_state(lab, gs, c, W, H) for c in ctls]
            st[d]["long"].append(sum(1 for g in gs if g[4] >= LONG and not any(
                _inter(g[:4], b) >= 0.5 * max(_area(g[:4]), 1.0) for b in gt_all)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(images)}", flush=True)

    L = ["# 묶기 진단 — 개발 세트 (찾기 = Hessian 고정 32.5)", "", f"흔적 양 평균 {np.mean(frac) * 100:.2f}%", "",
         "## G1 — 조각 사이 틈 · 방향 (가장 가까운 다른 조각)", "",
         "| 어디 | 조각 수 | 끝점 거리 px (10 · 50 · 90%) | ≤10 | ≤20 | ≤30 | ≤45 | 방향 차이 ° (10 · 50 · 90%) | ≤30° |", "|---|---|---|---|---|---|---|---|---|"]
    for k, name in (("gt", "정답 선 균열 박스"), ("ctl", "손상 없는 비교 박스")):
        a = np.array(g1[k])
        L.append(f"| {name} | {len(a)} | {' · '.join(f'{np.percentile(a[:, 0], p):.0f}' for p in (10, 50, 90))} | "
                 + " | ".join(f"{(a[:, 0] <= t).mean() * 100:.0f}%" for t in (10, 20, 30, 45))
                 + f" | {' · '.join(f'{np.percentile(a[:, 1], p):.0f}' for p in (10, 50, 90))} | {(a[:, 1] <= 30).mean() * 100:.0f}% |")
    L += ["", "## G2 — 잇기 거리", "",
          "| 잇기 | M 맞출 수 있는 정답 | M 비교 박스 | **M′** | 넘침 | 짧음 | 사진당 긴 묶음 |", "|---|---|---|---|---|---|---|"]
    arr = {}
    for d in DISTS:
        g = np.array([s for s in st[d]["gt"] if s]); c = np.array([s for s in st[d]["ctl"] if s])
        arr[d] = (g, c)
        M, Mc = (g == "M").mean(), (c == "M").mean()
        L.append(f"| {d or '없음'} | {M:.3f} | {Mc:.3f} | {M - Mc:.3f} | {(g == '넘침').mean():.3f} | {(g == '짧음').mean():.3f} | {np.mean(st[d]['long']):.1f} |")
    L += ["", "잇기 없음 대비 (95% 범위, 박스 단위 짝지은 부트스트랩 · ✱ = 0을 안 포함)", "",
          "| 잇기 | ΔM′ | Δ넘침 |", "|---|---|---|"]
    brng = np.random.default_rng(20261006)
    g0, c0 = arr[0]
    for d in DISTS[1:]:
        g, c = arr[d]
        dm, dov = [], []
        for _ in range(1000):
            ig, ic = brng.integers(0, len(g), len(g)), brng.integers(0, len(c), len(c))
            dm.append(((g[ig] == "M").mean() - (c[ic] == "M").mean()) - ((g0[ig] == "M").mean() - (c0[ic] == "M").mean()))
            dov.append((g[ig] == "넘침").mean() - (g0[ig] == "넘침").mean())
        (a, b), (e, f) = np.percentile(dm, [2.5, 97.5]), np.percentile(dov, [2.5, 97.5])
        pm = ((g == "M").mean() - (c == "M").mean()) - ((g0 == "M").mean() - (c0 == "M").mean())
        po = (g == "넘침").mean() - (g0 == "넘침").mean()
        L.append(f"| {d} | {pm:+.3f}{'✱' if a > 0 or b < 0 else ''} [{a:+.3f}, {b:+.3f}] | {po:+.3f}{'✱' if e > 0 or f < 0 else ''} [{e:+.3f}, {f:+.3f}] |")
    md = "\n".join(L) + "\n"
    (OUT / "diagnose_group.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
