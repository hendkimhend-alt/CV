"""
손상 정의 — 정답(RDD, 개발 세트)으로 "라벨러가 무엇을 손상으로 봤나"를 확인.
① 종류별 박스 모양 (긴 변 1024 기준): 폭 · 높이 · 짧은 변 · 세장비 · 면적 · 세로 위치 · 사진당 개수
② 같은 종류 박스끼리 붙어 있는 비율 (긴 손상을 여러 박스로 나눠 그렸나)
③ 밝기 대비 (대략): 박스 안 어두운 픽셀(5%) · 중앙값과 박스 바깥 띠의 중앙값 비교
   비교 기준 = 같은 사진 · 같은 높이에서 좌우로 옮긴 같은 크기 박스 (정답 박스와 안 겹치는 곳) — 손상 없는 노면도 어두운 5%는 바깥보다 어둡다
④ 종류별 예시 그림 (맥락 · 1:1 확대)
출력: outputs/analysis/define_damage/
"""
import csv, json, os, sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import OUTPUT_DIR, RDD_DIR, imread, imwrite

LONG = 1024
OUT = OUTPUT_DIR / "analysis" / "define_damage"
KO = {"longitudinal crack": "세로 균열", "transverse crack": "가로 균열", "alligator crack": "거북등 균열",
      "pothole": "포트홀", "other corruption": "기타(페인트 흐림)"}


def dev_rows():
    return [r for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         "labels", "split_rdd.csv"), encoding="utf-8-sig"))
            if r["split"] == "dev"]


def load(name):
    d = json.loads((RDD_DIR / "ann" / f"{name}.json").read_text(encoding="utf-8"))
    s = LONG / max(d["size"]["width"], d["size"]["height"])
    objs = []
    for o in d["objects"]:
        (x1, y1), (x2, y2) = o["points"]["exterior"]
        detail = "; ".join(str(t.get("value")) for t in o.get("tags", []))
        objs.append((o["classTitle"], detail, *(v * s for v in (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))))
    return s, round(d["size"]["height"] * s), objs


def contrast(gray, x1, y1, x2, y2):
    H, W = gray.shape
    x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    m = max(8, int(0.3 * min(x2 - x1, y2 - y1)))
    X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    ring = np.ones((Y2 - Y1, X2 - X1), bool)
    ring[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
    inside = gray[y1:y2, x1:x2].astype(float)
    out = gray[Y1:Y2, X1:X2][ring].astype(float)
    if out.size < 20:
        return None
    ref = np.median(out)
    return np.percentile(inside, 5) - ref, np.median(inside) - ref, ref


def control_box(objs, x1, y1, x2, y2, W, rng):
    """같은 높이에서 좌우로 옮긴 같은 크기 박스 — 어떤 정답 박스와도 안 겹치면 반환."""
    w = x2 - x1
    for _ in range(20):
        nx = rng.uniform(0, W - w) if W > w else None
        if nx is None:
            return None
        if all(nx + w <= o[2] or nx >= o[4] or y2 <= o[3] or y1 >= o[5] for o in objs):
            return nx, y1, nx + w, y2
    return None


def touching(objs, gap=10):
    """같은 종류 박스끼리 gap px 안에 있는 박스의 비율 (긴 손상을 나눠 그렸나)."""
    hit = set()
    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i < j and a[0] == b[0]:
                dx = max(0, max(a[2], b[2]) - min(a[4], b[4]))
                dy = max(0, max(a[3], b[3]) - min(a[5], b[5]))
                if dx <= gap and dy <= gap:
                    hit.update((i, j))
    return hit


def sheet(items, path, tile=300, cols=4):
    tiles = []
    for img, (x1, y1, x2, y2), label in items:
        H, W = img.shape[:2]
        m = int(0.3 * max(x2 - x1, y2 - y1)) + 10
        X1, Y1, X2, Y2 = max(0, int(x1) - m), max(0, int(y1) - m), min(W, int(x2) + m), min(H, int(y2) + m)
        crop = img[Y1:Y2, X1:X2].copy()
        cv2.rectangle(crop, (int(x1) - X1, int(y1) - Y1), (int(x2) - X1, int(y2) - Y1), (0, 255, 255), 1)
        f = tile / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST)
        t = np.zeros((tile, tile, 3), np.uint8)
        t[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(t, label, (4, tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def zoom_sheet(items, path, half=48, scale=4, cols=4):
    """1:1(긴 변 1024 기준) 확대 — 박스 중앙 96×96을 4배로. 균열 폭을 눈으로 재기 위해."""
    tiles = []
    for img, (x1, y1, x2, y2), label in items:
        H, W = img.shape[:2]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        X1, Y1 = min(max(0, cx - half), W - 2 * half), min(max(0, cy - half), H - 2 * half)
        crop = img[Y1:Y1 + 2 * half, X1:X1 + 2 * half]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for k in range(0, crop.shape[1], 10 * scale):          # 10px 눈금
            cv2.line(crop, (k, 0), (k, 6), (0, 0, 255), 1)
        cv2.putText(crop, label, (4, crop.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(crop)
    t = tiles[0].shape[0]
    while len(tiles) % cols:
        tiles.append(np.zeros((t, t, 3), np.uint8))
    imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = dev_rows()
    stat = defaultdict(list); details = defaultdict(lambda: defaultdict(int)); per_img = defaultdict(list)
    touch = defaultdict(lambda: [0, 0]); samples = defaultdict(list); ctrl = defaultdict(list)
    crng = np.random.default_rng(0)
    for r in rows:
        s, H, objs = load(r["image"])
        img = imread(str(RDD_DIR / "img" / r["image"]))
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hit = touching(objs)
        cnt = defaultdict(int)
        for i, (cls, det, x1, y1, x2, y2) in enumerate(objs):
            w, h = x2 - x1, y2 - y1
            c = contrast(gray, x1, y1, x2, y2)
            stat[cls].append((w, h, min(w, h), max(w, h) / max(1, min(w, h)), w * h / (img.shape[0] * img.shape[1]),
                              (y1 + y2) / 2 / img.shape[0], *(c if c else (np.nan,) * 3)))
            details[cls][det] += 1
            touch[cls][0] += i in hit; touch[cls][1] += 1
            cnt[cls] += 1
            samples[cls].append((r["image"], (x1, y1, x2, y2), r["source"]))
            cb = control_box(objs, x1, y1, x2, y2, img.shape[1], crng)
            cc = contrast(gray, *cb) if cb else None
            if cc:
                ctrl[cls].append(cc[:2])
        for cls in KO:
            if cnt[cls]:
                per_img[cls].append(cnt[cls])

    q = lambda a: " · ".join(f"{np.nanpercentile(a, p):.0f}" for p in (10, 50, 90))
    q2 = lambda a: " · ".join(f"{np.nanpercentile(a, p):.2f}" for p in (10, 50, 90))
    L = ["# 손상 정의 — 정답 박스 통계 (개발 세트, 긴 변 1024 기준)", "",
         "값은 10% · 중앙값 · 90%", "",
         "| 종류 | 개수 | 세부 정의(RDD 태그) | 폭 px | 높이 px | 짧은 변 px | 세장비 | 면적(사진 대비 %) | 세로 위치(0 위 · 1 아래) | 사진당 개수 | 같은 종류와 붙은 박스 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cls in KO:
        a = np.array(stat[cls])
        det = " / ".join(f"{k} {v}" for k, v in details[cls].items())
        L.append(f"| {KO[cls]} | {len(a)} | {det} | {q(a[:, 0])} | {q(a[:, 1])} | {q(a[:, 2])} | {q2(a[:, 3])} | "
                 f"{q2(a[:, 4] * 100)} | {q2(a[:, 5])} | {np.median(per_img[cls]):.0f} (최대 {max(per_img[cls])}) | "
                 f"{touch[cls][0] / touch[cls][1] * 100:.0f}% |")
    L += ["", "## 밝기 대비 (박스 안 − 박스 바깥 띠, 회색조 0~255)", "",
          "비교 = 같은 사진 · 같은 높이의 손상 없는 같은 크기 박스", "",
          "| 종류 | 안쪽 어두운 5% − 바깥 | (비교) | 안쪽 중앙값 − 바깥 | (비교) | 안쪽 중앙값이 바깥보다 밝은 비율 |", "|---|---|---|---|---|---|"]
    for cls in KO:
        a = np.array(stat[cls]); c = np.array(ctrl[cls])
        L.append(f"| {KO[cls]} | {q(a[:, 6])} | {q(c[:, 0])} | {q(a[:, 7])} | {q(c[:, 1])} | {np.nanmean(a[:, 7] > 0) * 100:.0f}% |")
    (OUT / "define_damage.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    rng = np.random.default_rng(20261007)
    cache = {}
    def get(name):
        if name not in cache:
            s, _, _ = load(name)
            im = imread(str(RDD_DIR / "img" / name))
            cache[name] = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cache[name]
    for cls in KO:
        pick = rng.choice(len(samples[cls]), min(16, len(samples[cls])), replace=False)
        items = [(get(samples[cls][i][0]), samples[cls][i][1], samples[cls][i][2]) for i in pick]
        key = cls.split()[0]
        sheet(items, OUT / f"{key}.jpg")
        if "crack" in cls:
            zoom_sheet(items[:8], OUT / f"{key}_zoom.jpg")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
