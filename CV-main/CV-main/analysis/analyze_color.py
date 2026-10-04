"""
색 채널별로 손상(균열·포트홀) 정보가 얼마나 담겨 있는지 측정한다.
"흑백으로 바꿔도 되는가"의 근거용.

방법 (RDD2020_train 라벨 박스 사용):
1. 채널마다 국소 편차 z = (채널 - 31px median 배경) / 노면(하단 절반) 편차의 robust std(MAD)
   → "주변보다 튀는 픽셀". 채널마다 자기 퍼짐으로 나누므로 채널끼리 공정 비교
2. 박스마다 튀는 픽셀 비율: 어두운 쪽(z < -3), 양방향(|z| > 3)
3. 대조군: 같은 크기, 손상 박스 바로 옆 노면(좌·우·아래·위 이동, 라벨과 안 겹침)
4. 손상 박스 vs 대조 박스를 그 비율로 얼마나 잘 가르는지 AUC
   0.5 = 정보 없음(동전 던지기), 1.0 = 완벽히 구분
채널: Gray / B G R / HSV의 S·V / Lab의 L·a·b (H는 무채색 노면에서 정의가 불안정해 제외)

참고: 처음엔 "박스 평균 vs 주변 평균"(Cohen's d)으로 쟀으나, 대조군과 비교하니 손상 박스가 더 큰 경우가
54%뿐이라 폐기 — 균열은 박스 면적의 일부라 평균에 묻힌다. 그래서 픽셀 단위로 바꿈.
또 대조군을 하단 절반 무작위 위치로 뽑았을 땐 AUC가 0.5 미만(차선·차량·연석에 떨어짐) → 바로 옆 노면으로 바꿈.
출력: outputs/analysis/color_channels.md, out/color_channels.csv
"""
import csv, glob, json, os
import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import OUTPUT_DIR, PROVIDED_DIR, RDD_DIR, imread, imwrite

ROOT = str(RDD_DIR)
OUT = str(OUTPUT_DIR / "analysis")
os.makedirs(OUT, exist_ok=True)
CLASSES = ["longitudinal crack", "transverse crack", "alligator crack", "pothole"]
CHANNELS = ["Gray", "B", "G", "R", "S", "V", "L", "a", "b"]
COLOR = ["S", "a", "b"]
BG_KSIZE = 31
Z_THRESH = 3
MIN_PIXELS = 200


def planes(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return dict(zip(CHANNELS, [gray, b, g, r, hsv[..., 1], hsv[..., 2], lab[..., 0], lab[..., 1], lab[..., 2]]))


def zmap(p):
    dev = p.astype(np.float32) - cv2.medianBlur(p, BG_KSIZE).astype(np.float32)
    road = dev[dev.shape[0] // 2:]
    mad = np.median(np.abs(road - np.median(road))) * 1.4826
    return dev / max(mad, 0.5)


def features(Z, x1, y1, x2, y2):
    r = {}
    for ch, z in Z.items():
        zz = z[y1:y2, x1:x2]
        r[f"{ch}_dark"] = float((zz < -Z_THRESH).mean())
        r[f"{ch}_abs"] = float((np.abs(zz) > Z_THRESH).mean())
    return r


def rankdata(v):
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    ranks = rankdata(np.concatenate([pos, neg]))
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


rows = []
for ann in sorted(glob.glob(f"{ROOT}/ann/*.json")):
    d = json.load(open(ann))
    if not any(o["classTitle"] in CLASSES for o in d["objects"]):
        continue
    name = os.path.basename(ann)[:-5]
    bgr = imread(f"{ROOT}/img/{name}")
    if bgr is None:
        continue
    H, W = bgr.shape[:2]
    Z = {ch: zmap(p) for ch, p in planes(bgr).items()}

    labeled = np.zeros((H, W), bool)
    boxes = []
    for o in d["objects"]:
        (x1, y1), (x2, y2) = o["points"]["exterior"]
        x1, x2 = sorted((max(0, x1), min(W, x2)))
        y1, y2 = sorted((max(0, y1), min(H, y2)))
        labeled[y1:y2, x1:x2] = True
        boxes.append((o["classTitle"], x1, y1, x2, y2))

    for cls, x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if cls not in CLASSES or bw * bh < MIN_PIXELS:
            continue
        # 대조 박스: 바로 옆 노면 (좌·우·아래·위로 박스 크기만큼 이동, 라벨과 안 겹치는 첫 위치)
        for cx, cy in [(x1 - bw, y1), (x2, y1), (x1, y2), (x1, y1 - bh)]:
            if 0 <= cx and cx + bw <= W and 0 <= cy and cy + bh <= H and not labeled[cy:cy + bh, cx:cx + bw].any():
                rows.append({"image": name, "class": cls, "group": "damage", **features(Z, x1, y1, x2, y2)})
                rows.append({"image": name, "class": cls, "group": "control", **features(Z, cx, cy, cx + bw, cy + bh)})
                break

with open(f"{OUT}/color_channels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)


def split(rs, key):
    return ([r[key] for r in rs if r["group"] == "damage"], [r[key] for r in rs if r["group"] == "control"])


def auc_of(rs, key):
    return auc(*split(rs, key))


n = sum(r["group"] == "damage" for r in rows)
lines = ["# 색 채널별 손상 정보량\n",
         f"RDD2020_train 라벨 박스 {n}개(균열 3종 + 포트홀) vs 바로 옆 노면의 같은 크기 대조 박스 {n}개. "
         "재현: `python analysis/analyze_color.py`\n",
         f"값 = AUC: 박스 안 '주변보다 튀는 픽셀'(|z|>{Z_THRESH}) 비율로 손상/대조를 가르는 정확도. "
         "**0.5 = 정보 없음, 1.0 = 완벽**. 채널마다 자기 퍼짐으로 정규화해 공정 비교.\n",
         "## 1. 채널별 AUC\n",
         "| 기준 | " + " | ".join(CHANNELS) + " |",
         "|---|" + "---|" * len(CHANNELS)]
for label, suffix in [("어두운 쪽 (z<-3)", "_dark"), ("양방향 (|z|>3)", "_abs")]:
    lines.append(f"| {label} | " + " | ".join(f"{auc_of(rows, ch + suffix):.3f}" for ch in CHANNELS) + " |")

lines += ["\n## 2. 클래스별 AUC (양방향)\n",
          "| 클래스 | n | " + " | ".join(CHANNELS) + " |",
          "|---|---|" + "---|" * len(CHANNELS)]
for cls in CLASSES:
    rs = [r for r in rows if r["class"] == cls]
    lines.append(f"| {cls} | {len(rs) // 2} | " + " | ".join(f"{auc_of(rs, ch + '_abs'):.2f}" for ch in CHANNELS) + " |")

# 흑백에 색을 더하면 나아지는가: 픽셀이 Gray 또는 색 채널 중 하나라도 튀면 카운트하는 대신, 박스 단위 max로 근사
dmg, ctl = [r for r in rows if r["group"] == "damage"], [r for r in rows if r["group"] == "control"]
comb = lambda r: max(r["Gray_abs"], *(r[c + "_abs"] for c in COLOR))
lines += ["\n## 3. 흑백에 색을 더하면?\n",
          f"- Gray 단독 AUC: **{auc_of(rows, 'Gray_abs'):.3f}**",
          f"- Gray + S·a·b 결합 (박스마다 네 채널 중 최댓값) AUC: **{auc([comb(r) for r in dmg], [comb(r) for r in ctl]):.3f}**",
          f"- 색만 (S·a·b 최댓값) AUC: **{auc([max(r[c + '_abs'] for c in COLOR) for r in dmg], [max(r[c + '_abs'] for c in COLOR) for r in ctl]):.3f}**\n"]
open(f"{OUT}/color_channels.md", "w").write("\n".join(lines))
print("\n".join(lines))
