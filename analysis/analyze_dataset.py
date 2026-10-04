"""
제공 데이터셋(PBL 모듈 1 dataset) 13장의 품질 특성을 정량 측정한다.
- 해상도, 밝기 평균/표준편차(대비), 어두운/밝은 픽셀 비율 (조도 문제)
- 블록별 밝기 편차 (조도 불균일)
- Laplacian 분산 (흐림 정도; 낮을수록 흐림)
- Immerkaer 잡음 추정 (noise sigma)
- Canny 에지 비율 (에지가 얼마나 검출되는지)
출력: outputs/analysis/stats.csv, out/stats.md, out/hist_<name>.png, out/montage.png
"""
import csv, glob, os, sys, math
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from paths import OUTPUT_DIR, PROVIDED_DIR, RDD_DIR, imread, imwrite

DATA_DIR = str(PROVIDED_DIR)
OUT_DIR = str(OUTPUT_DIR / "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def noise_sigma(gray):
    """Immerkaer (1996) fast noise variance estimation."""
    h, w = gray.shape
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, k, borderType=cv2.BORDER_REFLECT)
    return math.sqrt(math.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(conv).sum()


def block_means(gray, n=4):
    h, w = gray.shape
    bh, bw = h // n, w // n
    return np.array([[gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw].mean() for j in range(n)] for i in range(n)])


rows = []
for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg"))):
    name = os.path.splitext(os.path.basename(path))[0]
    bgr = imread(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    g = gray.astype(np.float64)

    mean, std = g.mean(), g.std()
    p1, p99 = np.percentile(g, [1, 99])
    dark = (gray < 40).mean() * 100
    bright = (gray > 215).mean() * 100
    bm = block_means(gray, 4)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sigma = noise_sigma(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = (edges > 0).mean() * 100
    # 하단 절반(노면 영역)만 따로
    road = gray[h//2:, :]
    road_mean, road_std = road.mean(), road.std()
    road_lap = cv2.Laplacian(road, cv2.CV_64F).var()

    rows.append(dict(name=name, w=w, h=h, mean=mean, std=std, p1=p1, p99=p99,
                     dark=dark, bright=bright,
                     block_std=bm.std(), block_min=bm.min(), block_max=bm.max(),
                     lap_var=lap_var, noise=sigma, edge=edge_ratio,
                     road_mean=road_mean, road_std=road_std, road_lap=road_lap))

    # 히스토그램 저장
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_img = np.full((200, 256, 3), 255, np.uint8)
    hist = hist / hist.max() * 190
    for x in range(256):
        cv2.line(hist_img, (x, 199), (x, 199 - int(hist[x])), (80, 80, 80), 1)
    imwrite(os.path.join(OUT_DIR, f"hist_{name}.png"), hist_img)

keys = ["name", "w", "h", "mean", "std", "p1", "p99", "dark", "bright", "block_std",
        "block_min", "block_max", "lap_var", "noise", "edge", "road_mean", "road_std", "road_lap"]
with open(os.path.join(OUT_DIR, "stats.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=keys)
    wr.writeheader()
    for r in rows:
        wr.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})

# markdown 표
hdr = ["이미지", "해상도", "밝기평균", "밝기std", "p1-p99", "어두운%", "밝은%", "블록std", "Lap분산", "노이즈σ", "에지%", "노면평균", "노면Lap"]
lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for r in rows:
    lines.append("| {name} | {w}x{h} | {mean:.1f} | {std:.1f} | {p1:.0f}-{p99:.0f} | {dark:.1f} | {bright:.1f} | {block_std:.1f} | {lap_var:.0f} | {noise:.2f} | {edge:.2f} | {road_mean:.1f} | {road_lap:.0f} |".format(**r))
md = "\n".join(lines)
with open(os.path.join(OUT_DIR, "stats.md"), "w") as f:
    f.write(md + "\n")
print(md)

# 요약 통계
arr = {k: np.array([r[k] for r in rows]) for k in keys[3:]}
print()
for k in ["mean", "std", "dark", "bright", "block_std", "lap_var", "noise", "edge"]:
    print(f"{k:10s} min={arr[k].min():8.2f} max={arr[k].max():8.2f} mean={arr[k].mean():8.2f}")

# 몽타주 (썸네일 + 이름)
thumbs = []
for r, path in zip(rows, sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))):
    im = imread(path)
    im = cv2.resize(im, (300, 300))
    cv2.rectangle(im, (0, 0), (300, 22), (0, 0, 0), -1)
    cv2.putText(im, r["name"][:28], (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    thumbs.append(im)
while len(thumbs) % 4:
    thumbs.append(np.zeros((300, 300, 3), np.uint8))
grid = np.vstack([np.hstack(thumbs[i:i+4]) for i in range(0, len(thumbs), 4)])
imwrite(os.path.join(OUT_DIR, "montage.png"), grid)
