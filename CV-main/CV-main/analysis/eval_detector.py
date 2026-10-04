"""
detect()를 RDD 라벨로 평가한다 — 박스 단위.

정답 박스(균열/포트홀)마다 바로 옆 노면에 같은 크기의 대조 박스를 둔다 (좌·우·아래·위, 어떤 라벨과도 안 겹침).
- 잡은 비율(TPR) = 정답 박스 중 detect()가 그 종류 후보를 하나라도 내놓은 비율
- 헛잡은 비율(FPR) = 대조 박스 중 같은 종류 후보가 나온 비율
- 구분력 = TPR − FPR (0 = 손상과 노면을 못 가림, 1 = 완벽)
- 진단: 박스 안 가장 길쭉한 후보(판정 전 전부 포함)의 세장비로 정답/대조를 가르는 AUC
  → ④가 쓸 모양 차이가 ①②에서 만들어지고 있는지
후보는 bbox의 50% 이상이 박스 안이면 그 박스 소속 (MEMBER_FRAC=0 환경변수면 조금만 겹쳐도 소속 —
조각 묶기처럼 후보가 커지는 설정이 50% 기준에서 불리해지는 것 확인용). 박스 안 잡음 개수에 휘둘리지 않는다.
주의: RDD 라벨은 빠진 손상이 있어 대조 박스에 실제 손상이 있을 수 있음 → FPR은 과대. 설정끼리 상대 비교용.

실행: python analysis/eval_detector.py ['{"blackhat_ksize": 7}']
"""
import glob, json, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import OUTPUT_DIR, PROVIDED_DIR, RDD_DIR, imread, imwrite
from detect import detect

ROOT = str(RDD_DIR)
LONG_SIDE = 1024
GT = {"crack": {"longitudinal crack", "transverse crack", "alligator crack"}, "pothole": {"pothole"}}
MIN_BOX_AREA = 400


def prepare(gray, boxes):
    h, w = gray.shape
    s = LONG_SIDE / max(h, w)
    gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    top = gray.shape[0] // 2
    roi = gray[top:, :]
    H, W = roi.shape
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, int(x1 * s)), min(W, int(x2 * s))
        y1, y2 = max(0, int(y1 * s - top)), min(H, int(y2 * s - top))
        if x2 > x1 and y2 > y1:
            out.append((cls, x1, y1, x2, y2))
    return roi, out


def control_box(box, boxes, W, H):
    _, x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    for cx, cy in [(x1 - bw, y1), (x2, y1), (x1, y2), (x1, y1 - bh)]:
        if cx < 0 or cy < 0 or cx + bw > W or cy + bh > H:
            continue
        if all(cx + bw <= b[1] or cx >= b[3] or cy + bh <= b[2] or cy >= b[4] for b in boxes):
            return (cx, cy, cx + bw, cy + bh)
    return None


MEMBER_FRAC = float(os.environ.get("MEMBER_FRAC", 0.5))   # 0이면 조금만 겹쳐도 소속 (큰 후보에 불리하지 않게)


def members(dets, region):
    rx1, ry1, rx2, ry2 = region
    out = []
    for d in dets:
        x, y, w, h = d["bbox"]
        iw = max(0, min(x + w, rx2) - max(x, rx1))
        ih = max(0, min(y + h, ry2) - max(y, ry1))
        if iw * ih > 0 and iw * ih >= MEMBER_FRAC * max(w * h, 1):
            out.append(d)
    return out


def auc(pos, neg):
    v = np.concatenate([pos, neg])
    order = v.argsort(kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    for u in np.unique(v):              # 동점 평균 순위
        m = v == u
        ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(cfg):
    stats = {k: {"hit": [], "fhit": [], "elong": [], "felong": []} for k in GT}
    n_dets = []
    for ann in sorted(glob.glob(f"{ROOT}/ann/*.json")):
        d = json.load(open(ann))
        raw = []
        for o in d["objects"]:
            (x1, y1), (x2, y2) = o["points"]["exterior"]
            raw.append((o["classTitle"], min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        if not any(c in GT["crack"] | GT["pothole"] for c, *_ in raw):
            continue
        gray = imread(f"{ROOT}/img/{os.path.basename(ann)[:-5]}", cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        roi, boxes = prepare(gray, raw)
        H, W = roi.shape
        dets = detect(roi, {**cfg, "keep_noise": True})
        n_dets.append(sum(x["type"] == "crack" for x in dets))
        for kind, classes in GT.items():
            for b in boxes:
                if b[0] not in classes or (b[3] - b[1]) * (b[4] - b[2]) < MIN_BOX_AREA:
                    continue
                ctl = control_box(b, boxes, W, H)
                if ctl is None:
                    continue
                for region, hk, ek in [(b[1:], "hit", "elong"), (ctl, "fhit", "felong")]:
                    m = members(dets, region)
                    stats[kind][hk].append(any(x["type"] == kind for x in m))
                    stats[kind][ek].append(max([x["elong"] for x in m], default=0.0))
    res = {"crack_dets_per_img": float(np.mean(n_dets))}
    for kind, s in stats.items():
        tpr, fpr = np.mean(s["hit"]), np.mean(s["fhit"])
        res[kind] = {"n": len(s["hit"]), "TPR": tpr, "FPR": fpr, "sep": tpr - fpr,
                     "elong_auc": auc(np.array(s["elong"]), np.array(s["felong"]))}
    return res


def fmt(res):
    c, p = res["crack"], res["pothole"]
    return (f"균열 n={c['n']} TPR {c['TPR']:.2f} FPR {c['FPR']:.2f} 구분력 {c['sep']:+.2f} 세장비AUC {c['elong_auc']:.3f} | "
            f"포트홀 n={p['n']} TPR {p['TPR']:.2f} FPR {p['FPR']:.2f} 구분력 {p['sep']:+.2f} | "
            f"사진당 균열 후보 {res['crack_dets_per_img']:.1f}")


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(fmt(evaluate(cfg)))
