"""픽셀 특징 지도(Lab, 기울기, 질감)와 32×32 격자 칸별 특징."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FeatureMaps:
    lab: np.ndarray        # H×W×3 float32, L* 0~100
    gradient: np.ndarray   # H×W, L* 변화량 / px
    texture: np.ndarray    # H×W, |Laplacian| 국소 평균


@dataclass
class GridFeatures:
    cell_size: int
    rows: int
    cols: int
    height: int
    width: int
    features: dict          # 이름 → (rows, cols) 배열

    def cell_slice(self, r, c):
        cs = self.cell_size
        return slice(r * cs, min((r + 1) * cs, self.height)), slice(c * cs, min((c + 1) * cs, self.width))

    def cell_center(self, r, c):
        ys, xs = self.cell_slice(r, c)
        return (xs.start + xs.stop - 1) / 2.0, (ys.start + ys.stop - 1) / 2.0


def bgr_to_lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def color_distance(lab_a, lab_b, lightness_weight):
    """√((w·ΔL)² + Δa² + Δb²). 그림자는 L을 크게 바꾸므로 L은 w만큼만 본다."""
    diff = np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64)
    return np.sqrt((lightness_weight * diff[..., 0]) ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)


def _blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)


def compute_feature_maps(bgr, cfg_features):
    lab = bgr_to_lab(_blur(bgr, cfg_features["smooth_sigma"]))
    raw_l = bgr_to_lab(bgr)[..., 0]

    smooth_l = _blur(raw_l, cfg_features["gradient_sigma"])
    gx = cv2.Sobel(smooth_l, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(smooth_l, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gradient = np.sqrt(gx * gx + gy * gy) / 8.0      # Sobel 가중치 합 8 → 화소당 변화량

    laplacian = np.abs(cv2.Laplacian(raw_l, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REFLECT_101))
    texture = cv2.GaussianBlur(laplacian, (0, 0), cfg_features["texture_sigma"],
                               borderType=cv2.BORDER_REFLECT_101)
    return FeatureMaps(lab=lab, gradient=gradient.astype(np.float32), texture=texture.astype(np.float32))


def compute_grid_features(maps, cfg_grid, cfg_features):
    cs = cfg_grid["cell_size"]
    height, width = maps.gradient.shape
    rows, cols = -(-height // cs), -(-width // cs)
    names = ("mean_L", "mean_a", "mean_b", "std_L", "std_a", "std_b", "cx", "cy",
             "texture", "grad_mean", "edge_density", "fill")
    feats = {name: np.zeros((rows, cols), np.float64) for name in names}
    edge = maps.gradient > cfg_features["edge_threshold"]
    grid = GridFeatures(cs, rows, cols, height, width, feats)

    for r in range(rows):
        for c in range(cols):
            ys, xs = grid.cell_slice(r, c)
            lab = maps.lab[ys, xs].reshape(-1, 3).astype(np.float64)
            feats["mean_L"][r, c], feats["mean_a"][r, c], feats["mean_b"][r, c] = lab.mean(0)
            feats["std_L"][r, c], feats["std_a"][r, c], feats["std_b"][r, c] = lab.std(0)
            cx, cy = grid.cell_center(r, c)
            feats["cx"][r, c] = cx / max(width - 1, 1)
            feats["cy"][r, c] = cy / max(height - 1, 1)
            feats["texture"][r, c] = float(maps.texture[ys, xs].mean())
            feats["grad_mean"][r, c] = float(maps.gradient[ys, xs].mean())
            feats["edge_density"][r, c] = float(edge[ys, xs].mean())
            feats["fill"][r, c] = lab.shape[0] / float(cs * cs)

    feats["chroma"] = np.sqrt(feats["mean_a"] ** 2 + feats["mean_b"] ** 2)
    feats["neighbor_dist"] = _neighbor_distance(feats, cfg_features["lightness_weight"])
    return grid


def neighbor_offsets():
    return [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _neighbor_distance(feats, lightness_weight):
    """8-이웃 칸과의 평균 색 거리."""
    lab = np.stack([feats["mean_L"], feats["mean_a"], feats["mean_b"]], axis=-1)
    rows, cols = lab.shape[:2]
    total = np.zeros((rows, cols))
    count = np.zeros((rows, cols))
    for dr, dc in neighbor_offsets():
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        if r1 <= r0 or c1 <= c0:
            continue
        here = lab[r0:r1, c0:c1]
        there = lab[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        total[r0:r1, c0:c1] += color_distance(here, there, lightness_weight)
        count[r0:r1, c0:c1] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.inf)
