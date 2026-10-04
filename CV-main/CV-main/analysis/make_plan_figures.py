"""
1차 과제수행계획서용 그림 생성.
출력: docs/images/fig1_problems.png (문제가 실제로 어떻게 보이나)
      docs/images/fig2_pipeline.png (전체 흐름과 각 단계의 이유)
      docs/images/fig3_detect_idea.png (검출 아이디어 단계별)
실행: python analysis/make_plan_figures.py
"""
import os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "src"))
from paths import PROVIDED_DIR, imread
import detect as D
from run_detect import prepare

DATA = str(PROVIDED_DIR)
OUT = os.path.join(BASE, "docs", "images")
os.makedirs(OUT, exist_ok=True)
# 한글 글꼴: 설치된 것 중 첫 번째 (macOS Pretendard·Apple SD Gothic Neo / Windows 맑은 고딕)
from matplotlib import font_manager
_installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic",
                                                "NanumGothic") if f in _installed), "sans-serif")
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED = "#1f2328", "#57606a"
C_FIX, C_FIND, C_MEASURE = "#2f6fbf", "#d9822b", "#2e8b57"
BG = {C_FIX: "#e8f0fb", C_FIND: "#fdf1e4", C_MEASURE: "#e7f4ec"}


def load_gray(name):
    return cv2.cvtColor(imread(f"{DATA}/{name}.jpg"), cv2.COLOR_BGR2GRAY)


# ---------- 그림 1: 문제가 실제로 어떻게 보이나 ----------

def fig_problems():
    items = [
        ("United_States_004830", "조도 불균일 — 한 장 안의 그림자", "13장 중 9장 · 블록 밝기 편차 47"),
        ("China_MotorBike_002209", "흐림 — 에지가 사라짐", "13장 중 4장 · 선명도 4 (선명한 장 1,700~17,000)"),
        ("Japan_001959", "노이즈처럼 보이는 골재 질감", "잡음 추정 σ 20 · 에지 37% 대부분이 골재"),
        ("United_States_005996", "찾아야 할 것 — 어둡고 가는 선", "균열 = 노면보다 어둡고 가는 선"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (name, title, sub) in zip(axes, items):
        g = load_gray(name)
        h = g.shape[0]
        road = g[h // 2:, :]
        if name == "United_States_005996":      # 균열 부분 확대
            p = prepare(imread(f"{DATA}/{name}.jpg"))
            road = p[150:350, 130:530]
        ax.imshow(road, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=8)
        ax.text(0.5, -0.07, sub, transform=ax.transAxes, ha="center", va="top", fontsize=11, color=MUTED)
        ax.axis("off")
    fig.suptitle("과제가 말한 문제는 실제 사진에서 이렇게 나타난다 (제공 13장, 노면 하단 절반)",
                 fontsize=15, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_problems.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- 그림 2: 전체 흐름 ----------

def box(ax, x, y, w, h, title, why, color, title_size=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=1.6, edgecolor=color, facecolor="white"))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, why, ha="center", va="center", fontsize=10.2, color=MUTED)


def group(ax, x, y, w, h, label, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                linewidth=0, facecolor=BG[color]))
    ax.text(x + 0.012, y + h - 0.035, label, ha="left", va="center", fontsize=15, fontweight="bold", color=color)
    ax.text(x + 0.012, y + h - 0.072, sub, ha="left", va="center", fontsize=10.5, color=MUTED)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=color))


def fig_pipeline():
    fig = plt.figure(figsize=(17, 9.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 맨 위: 출발 생각
    ax.add_patch(FancyBboxPatch((0.03, 0.885), 0.94, 0.085, boxstyle="round,pad=0.006,rounding_size=0.015",
                                linewidth=0, facecolor="#f3f4f6"))
    ax.text(0.5, 0.945, "출발 생각:  균열 = 주변보다 어둡고 가는 선   ·   포트홀 = 주변보다 어두운 덩어리",
            ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(0.5, 0.905, "그런데 조도·흐림·노이즈 때문에 그 \"어둡고 가는 선\"이 사진에서 안 보인다  →  ① 보이게 고치고  ② 찾고  ③ 진짜 찾았는지 잰다",
            ha="center", va="center", fontsize=12, color=MUTED)

    # 그룹
    gy, gh = 0.12, 0.735
    group(ax, 0.105, gy, 0.255, gh, "① 사진 고치기", "개발 A · 손상이 보이게", C_FIX)
    group(ax, 0.385, gy, 0.3, gh, "② 찾기", "개발 B · 어둡고 가는 선 / 덩어리", C_FIND)
    group(ax, 0.71, gy, 0.255, gh, "③ 재기", "개발 A · 진짜 찾았나, 누구 덕인가", C_MEASURE)

    # 입력
    box(ax, 0.01, 0.46, 0.08, 0.1, "입력", "도로 사진", MUTED)
    arrow(ax, 0.09, 0.51, 0.115, 0.51)

    # ① 고치기
    fx, fw, bh = 0.12, 0.225, 0.076
    fix = [("노면 영역 · 크기 맞춤", "하늘·건물 제외 / 해상도 8배 차이"),
           ("감마", "사진 전체가 너무 밝거나 어두움"),
           ("CLAHE", "한 장 안의 그림자 (13장 중 9장)"),
           ("언샤프 — 흐린 사진만", "에지가 사라진 4장을 선명하게"),
           ("가우시안", "골재·잡음 점 줄이기")]
    ys = [0.67, 0.565, 0.46, 0.355, 0.25]
    for (t, w), y in zip(fix, ys):
        box(ax, fx, y, fw, bh, t, w, C_FIX)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, fx + fw / 2, y1, fx + fw / 2, y2 + bh)
    ax.text(fx + fw / 2, 0.175, "문제 하나에 보정 하나 · 서로 상쇄하는 조합 금지", ha="center", fontsize=10, color=C_FIX)
    arrow(ax, 0.355, 0.5, 0.39, 0.5, INK)

    # ② 찾기
    bx, bw = 0.4, 0.27
    box(ax, bx, 0.67, bw, bh, "흑백", "단서는 \"어둡다\" 하나 → 밝기만", C_FIND)
    hw = 0.13
    box(ax, bx, 0.47, hw, 0.15, "균열 경로", "Black-hat → 이중 임계값\n→ Closing\n가는 · 이어진 것만", C_FIND, 12)
    box(ax, bx + bw - hw, 0.47, hw, 0.15, "포트홀 경로", "큰 범위 이진화\n→ Opening\n덩어리만", C_FIND, 12)
    arrow(ax, bx + bw / 2, 0.67, bx + hw / 2, 0.62)
    arrow(ax, bx + bw / 2, 0.67, bx + bw - hw / 2, 0.62)
    box(ax, bx, 0.355, bw, bh, "연결 요소 → 모양 재기", "붙은 픽셀을 후보로 묶고 세장비·면적을 잰다", C_FIND)
    arrow(ax, bx + hw / 2, 0.47, bx + bw / 2 - 0.03, 0.355 + bh)
    arrow(ax, bx + bw - hw / 2, 0.47, bx + bw / 2 + 0.03, 0.355 + bh)
    box(ax, bx, 0.25, bw, bh, "판정", "길쭉하면 균열 · 크고 둥글면 포트홀 · 나머지 버림", C_FIND)
    arrow(ax, bx + bw / 2, 0.355, bx + bw / 2, 0.25 + bh)
    ax.text(bx + bw / 2, 0.175, "선은 이어 붙이고(Closing), 덩어리는 깎아 낸다(Opening)", ha="center", fontsize=10, color=C_FIND)
    arrow(ax, 0.685, 0.5, 0.72, 0.5, INK)

    # ③ 재기
    mx, mw = 0.725, 0.225
    meas = [("정답 박스와 비교", "후보가 늘어도 진짜 손상인지 모름\n→ hit / miss → precision · recall"),
            ("품질 지표", "고친 문제가 실제로 줄었나\n밝기 편차 · 선명도 · 잡음"),
            ("한 번에 하나만 바꾸기", "전처리 P0 / P1 / P1+  ×  검출 D0 / D1\n→ 효과가 누구 덕인지")]
    my = [0.62, 0.45, 0.28]
    for (t, w), y in zip(meas, my):
        box(ax, mx, y, mw, 0.125, t, w, C_MEASURE)
    for y1, y2 in zip(my[:-1], my[1:]):
        arrow(ax, mx + mw / 2, y1, mx + mw / 2, y2 + 0.125)
    ax.text(mx + mw / 2, 0.175, "흐림 / 그림자 / 정상 그룹별로 따로 본다", ha="center", fontsize=10, color=C_MEASURE)

    # 맨 아래: 결과
    ax.add_patch(FancyBboxPatch((0.3, 0.025), 0.4, 0.06, boxstyle="round,pad=0.006,rounding_size=0.015",
                                linewidth=1.4, edgecolor=INK, facecolor="white"))
    ax.text(0.5, 0.055, "결과: 손상 후보 박스  +  전처리 전·후 정량 비교표", ha="center", va="center",
            fontsize=13, fontweight="bold", color=INK)
    arrow(ax, mx + mw / 2, 0.12, 0.7, 0.07)

    fig.savefig(f"{OUT}/fig2_pipeline.png", dpi=160, facecolor="white")
    plt.close(fig)


# ---------- 그림 3: 검출 아이디어 단계별 ----------

def fig_detect_idea():
    name = "United_States_005996"
    img = prepare(imread(f"{DATA}/{name}.jpg"))
    cfg = D.DEFAULT_CFG
    y0, y1, x0, x1 = 120, 470, 180, 580
    crop = lambda a: a[y0:y1, x0:x1]

    canny = cv2.Canny(img, cfg["canny_low"], cfg["canny_high"])
    bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, D._kernel(cfg["blackhat_ksize"]))
    hyst, _ = D._hysteresis(bh, cfg)
    adaptive = cv2.adaptiveThreshold(cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, D._kernel(15)), 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for d in D.detect(img):
        if d["type"] == "crack":
            x, y, w, h = d["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (220, 40, 40), 2)

    panels = [
        (crop(img), "원본", "가운데 대각선의 가는 검은 선이 균열", "gray"),
        (crop(canny), "Canny (수업 기술)", "밝기가 바뀌는 곳을 다 잡음\n→ 그림자 경계만 잔뜩, 균열은 놓침", "gray"),
        (crop(np.clip(bh.astype(float) * 255 / 40, 0, 255)), "Black-hat", "\"어둡고 가는 것\"만 남김\n→ 균열이 빛남", "gray"),
        (crop(adaptive), "픽셀별 판단 (원안)", "흐린 부분은 끊기고 잡음 점은 남음\n→ 균열이 조각남", "gray"),
        (crop(hyst), "이중 임계값 (채택)", "진한 곳과 이어진 것만 살림\n→ 잡음 점이 빠짐", "gray"),
        (crop(vis), "판정 결과", "길쭉한 덩어리만 균열(빨강)", None),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(19, 4.4))
    for ax, (im, t, s, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=255) if cmap else ax.imshow(im)
        ax.set_title(t, fontsize=13.5, fontweight="bold", color=INK, pad=6)
        ax.text(0.5, -0.05, s, transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=MUTED)
        ax.axis("off")
    fig.suptitle("② 찾기의 아이디어 — \"어둡고 가는 선\"을 단계별로 좁혀 간다 (United_States_005996 일부 확대)",
                 fontsize=15, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_detect_idea.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_problems()
    fig_pipeline()
    fig_detect_idea()
    print("saved to", OUT)
