"""
RDD2020_train (Supervisely 포맷, datasetninja 배포본) 구조·라벨·품질 분석.
출력: outputs/analysis/rdd_summary.md, out/rdd_quality.csv, out/rdd_boxes_sample.png
"""
import json, glob, os, math, csv, random
import cv2, numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import OUTPUT_DIR, PROVIDED_DIR, RDD_DIR, imread, imwrite

ROOT = str(RDD_DIR)
OUT = str(OUTPUT_DIR / "analysis")
os.makedirs(OUT, exist_ok=True)

def noise_sigma(gray):
    h, w = gray.shape
    k = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], float)
    conv = cv2.filter2D(gray.astype(float), -1, k, borderType=cv2.BORDER_REFLECT)
    return math.sqrt(math.pi/2)/(6*(w-2)*(h-2))*np.abs(conv).sum()

anns = sorted(glob.glob(f"{ROOT}/ann/*.json"))
recs = []
cls_count = {}
cls_country = {}
res_country = {}
box_stats = {}  # class -> list of (w,h,aspect, area_ratio)
for a in anns:
    name = os.path.basename(a)[:-5]  # strip .json -> xxx.jpg
    d = json.load(open(a))
    country = name.rsplit("_", 1)[0]
    H, W = d["size"]["height"], d["size"]["width"]
    res_country.setdefault(country, set()).add((W, H))
    objs = d["objects"]
    recs.append(dict(name=name, country=country, W=W, H=H, n=len(objs),
                     classes=[o["classTitle"] for o in objs]))
    for o in objs:
        c = o["classTitle"]
        cls_count[c] = cls_count.get(c, 0) + 1
        cls_country.setdefault(c, {}).setdefault(country, 0)
        cls_country[c][country] += 1
        (x1, y1), (x2, y2) = o["points"]["exterior"]
        bw, bh = abs(x2-x1), abs(y2-y1)
        box_stats.setdefault(c, []).append((bw, bh, max(bw,bh)/max(1,min(bw,bh)), bw*bh/(W*H)))

countries = sorted(set(r["country"] for r in recs))
lines = []
lines.append(f"# RDD2020_train 분석\n")
lines.append(f"- 이미지 {len(recs)}장, 라벨 박스 {sum(cls_count.values())}개, 클래스 {len(cls_count)}종, 국가 {len(countries)}종")
lines.append(f"- 손상 없음(박스 0개) 이미지: {sum(1 for r in recs if r['n']==0)}장")
lines.append(f"- 이미지당 박스 수: 평균 {np.mean([r['n'] for r in recs]):.2f}, 최대 {max(r['n'] for r in recs)}\n")

lines.append("## 국가별 장수·해상도\n")
lines.append("| 국가 | 장수 | 해상도 | 박스 수 |")
lines.append("|---|---|---|---|")
for c in countries:
    rs = [r for r in recs if r["country"] == c]
    lines.append(f"| {c} | {len(rs)} | {', '.join(f'{w}x{h}' for w,h in sorted(res_country[c]))} | {sum(r['n'] for r in rs)} |")

lines.append("\n## 클래스별 박스 수 (국가별)\n")
lines.append("| 클래스 | 합계 | " + " | ".join(countries) + " |")
lines.append("|---|---|" + "---|"*len(countries))
for c, n in sorted(cls_count.items(), key=lambda x: -x[1]):
    lines.append(f"| {c} | {n} | " + " | ".join(str(cls_country[c].get(k, 0)) for k in countries) + " |")

lines.append("\n## 클래스별 박스 형태 (중앙값)\n")
lines.append("| 클래스 | 폭 px | 높이 px | 세장비(장변/단변) | 면적 비율(박스/영상) |")
lines.append("|---|---|---|---|---|")
for c in sorted(box_stats):
    arr = np.array(box_stats[c])
    med = np.median(arr, axis=0)
    lines.append(f"| {c} | {med[0]:.0f} | {med[1]:.0f} | {med[2]:.2f} | {med[3]*100:.1f}% |")

# 품질 지표 (전체 804장) — 제공 13장과 같은 지표
qrows = []
for r in recs:
    img = imread(f"{ROOT}/img/{r['name']}", cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    h, w = img.shape
    road = img[h//2:, :]
    bm = np.array([[img[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4].mean() for j in range(4)] for i in range(4)])
    qrows.append(dict(name=r["name"], country=r["country"], n=r["n"],
                      mean=img.mean(), dark=(img<40).mean()*100, bright=(img>215).mean()*100,
                      block_std=bm.std(), lap=cv2.Laplacian(road, cv2.CV_64F).var(),
                      noise=noise_sigma(road), sat=((img==0)|(img==255)).mean()*100))
with open(f"{OUT}/rdd_quality.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(qrows[0].keys())); wr.writeheader()
    for q in qrows: wr.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in q.items()})

lap = np.array([q["lap"] for q in qrows]); bs = np.array([q["block_std"] for q in qrows])
mean = np.array([q["mean"] for q in qrows]); dark = np.array([q["dark"] for q in qrows]); sat = np.array([q["sat"] for q in qrows])
lines.append("\n## 품질 지표 분포 (노면 하단 절반 기준, 제공 13장과 동일 지표)\n")
lines.append("| 지표 | 최소 | 10% | 중앙값 | 90% | 최대 | 제공 13장 범위 |")
lines.append("|---|---|---|---|---|---|---|")
for nm, arr, ref in [("Laplacian 분산", lap, "3 ~ 16,964"), ("블록 밝기 std", bs, "9 ~ 74"), ("밝기 평균", mean, "81 ~ 206"), ("어두운 픽셀 %", dark, "0 ~ 50"), ("포화 픽셀 %", sat, "0.0 ~ 15.4")]:
    p = np.percentile(arr, [0, 10, 50, 90, 100])
    lines.append(f"| {nm} | {p[0]:.1f} | {p[1]:.1f} | {p[2]:.1f} | {p[3]:.1f} | {p[4]:.1f} | {ref} |")
lines.append("")
lines.append(f"- 흐림 후보 (Lap분산 < 50): {int((lap<50).sum())}장 / 국소조도 후보 (블록std > 44): {int((bs>44).sum())}장 / 저조도 후보 (어두운 픽셀 > 30%): {int((dark>30).sum())}장 / 과노출 후보 (밝기 평균 > 190): {int((mean>190).sum())}장 / 포화 > 10%: {int((sat>10).sum())}장")

open(f"{OUT}/rdd_summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# 박스 그려서 샘플 몽타주 (클래스별 1장씩 + 무작위)
random.seed(0)
colors = {"longitudinal crack": (0,0,255), "transverse crack": (0,200,0), "alligator crack": (0,220,255),
          "pothole": (255,120,0), "repair": (90,60,180), "other corruption": (0,120,255), "block crack": (120,200,0)}
picked = []
for c in sorted(cls_count):
    cand = [r for r in recs if c in r["classes"] and r["W"] <= 720]
    if cand: picked.append(random.choice(cand))
picked += random.sample([r for r in recs if r["n"] > 0 and r["W"] <= 720], 9 - len(picked)) if len(picked) < 9 else []
tiles = []
for r in picked[:9]:
    img = imread(f"{ROOT}/img/{r['name']}")
    d = json.load(open(f"{ROOT}/ann/{r['name']}.json"))
    for o in d["objects"]:
        (x1,y1),(x2,y2) = o["points"]["exterior"]; c = o["classTitle"]
        cv2.rectangle(img, (x1,y1), (x2,y2), colors.get(c,(255,255,255)), 3)
        cv2.putText(img, c, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(c,(255,255,255)), 2)
    img = cv2.resize(img, (400,400))
    cv2.rectangle(img,(0,0),(400,22),(0,0,0),-1); cv2.putText(img, r["name"][:-4], (3,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    tiles.append(img)
while len(tiles) % 3: tiles.append(np.zeros((400,400,3),np.uint8))
grid = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
imwrite(f"{OUT}/rdd_boxes_sample.png", grid)
