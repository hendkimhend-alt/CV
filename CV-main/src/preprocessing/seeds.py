"""격자 칸마다 도로다움 점수를 매기고, 서로 떨어진 상위 칸을 시드로 고른다."""
from dataclasses import dataclass, field

import numpy as np

from .grid_features import color_distance, neighbor_offsets

COMPONENTS = ("position", "neutral_color", "homogeneity", "texture", "brightness", "neighbor_similarity")
EPS = 1e-6


@dataclass
class Seed:
    row: int
    col: int
    x: float            # 작업 영상에서 칸 중심
    y: float
    score: float
    components: dict = field(default_factory=dict)


def _texture_score(values, lo, hi):
    """범위 안이면 1, 너무 매끈하면 (t/lo)², 너무 거칠면 hi/t."""
    below = np.clip(values / max(lo, EPS), 0, 1) ** 2
    above = np.clip(hi / np.maximum(values, EPS), 0, 1)
    return np.where(values < lo, below, np.where(values > hi, above, 1.0))


def _geometric_mean(parts, weights):
    # 산술평균과 달리 한 성분이 0에 가까우면 점수 전체가 낮아진다
    total_w = sum(weights[name] for name in parts)
    if total_w <= 0:
        return np.ones_like(next(iter(parts.values())))
    log_sum = sum(weights[name] * np.log(np.maximum(value, EPS)) for name, value in parts.items())
    return np.exp(log_sum / total_w)


def score_cells(grid, cfg_seeds, lightness_weight):
    f = grid.features
    pos = cfg_seeds["position"]
    comps = {
        # 아래쪽, 가운데일수록 큼
        "position": f["cy"] ** pos["vertical_power"]
                    * (1.0 - pos["horizontal_center_weight"] * np.abs(2.0 * f["cx"] - 1.0)),
        # 무채색(아스팔트)일수록 큼
        "neutral_color": np.exp(-(f["chroma"] / cfg_seeds["neutral_chroma_scale"]) ** 2),
        # 강한 경계(차량, 건물)가 적을수록 큼
        "homogeneity": np.exp(-f["edge_density"] / cfg_seeds["homogeneity_edge_scale"]),
        "texture": _texture_score(f["texture"], *cfg_seeds["texture_range"]),
    }
    l_lo, l_hi = cfg_seeds["brightness_range"]
    margin = cfg_seeds["brightness_margin"]
    too_dark = np.maximum(l_lo - f["mean_L"], 0) / margin
    too_bright = np.maximum(f["mean_L"] - l_hi, 0) / margin
    comps["brightness"] = np.clip(1.0 - too_dark - too_bright, 0.0, 1.0)

    weights = cfg_seeds["weights"]
    preliminary = _geometric_mean(comps, weights)
    comps["neighbor_similarity"] = _neighbor_support(grid, preliminary, lightness_weight,
                                                     cfg_seeds["neighbor_color_scale"])
    return _geometric_mean(comps, weights), comps


def _neighbor_support(grid, preliminary, lightness_weight, scale):
    """이웃 칸과 색이 비슷하고 그 이웃도 점수가 높을수록 큼."""
    f = grid.features
    lab = np.stack([f["mean_L"], f["mean_a"], f["mean_b"]], axis=-1)
    rows, cols = preliminary.shape
    total = np.zeros((rows, cols))
    count = np.zeros((rows, cols))
    for dr, dc in neighbor_offsets():
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        if r1 <= r0 or c1 <= c0:
            continue
        d = color_distance(lab[r0:r1, c0:c1], lab[r0 + dr:r1 + dr, c0 + dc:c1 + dc], lightness_weight)
        total[r0:r1, c0:c1] += np.exp(-d / scale) * preliminary[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        count[r0:r1, c0:c1] += 1
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def consistent_with_best(f, best, cell, cfg_seeds):
    """첫 시드와 채도, 밝기, 질감이 너무 다르면 다른 재질로 보고 뺀다."""
    (br, bc), (r, c) = best, cell
    limit = cfg_seeds["max_chroma_distance_to_best"]
    if limit is not None:
        chroma_d = np.hypot(f["mean_a"][r, c] - f["mean_a"][br, bc], f["mean_b"][r, c] - f["mean_b"][br, bc])
        if chroma_d > limit:
            return False
    limit = cfg_seeds["max_lightness_distance_to_best"]
    if limit is not None and abs(f["mean_L"][r, c] - f["mean_L"][br, bc]) > limit:
        return False
    limit = cfg_seeds["max_texture_ratio_to_best"]
    if limit is not None:
        t, t_best = max(f["texture"][r, c], EPS), max(f["texture"][br, bc], EPS)
        if max(t / t_best, t_best / t) > limit:
            return False
    return True


def select_seeds(grid, scores, components, cfg_seeds, cfg_grid):
    f = grid.features
    eligible = f["fill"] >= cfg_grid["min_cell_fill"]       # 가장자리에서 잘린 칸 제외
    flat = np.where(eligible, scores, -1.0).ravel()
    order = np.argsort(-flat, kind="stable")
    min_dist = cfg_seeds["min_distance_cells"]

    chosen, warnings = [], []
    for index in order:
        score = flat[index]
        if score < cfg_seeds["min_score"] or len(chosen) >= cfg_seeds["count_max"]:
            break
        r, c = divmod(int(index), grid.cols)
        if any(max(abs(r - s.row), abs(c - s.col)) < min_dist for s in chosen):
            continue
        if chosen and not consistent_with_best(f, (chosen[0].row, chosen[0].col), (r, c), cfg_seeds):
            continue
        x, y = grid.cell_center(r, c)
        comps = {name: round(float(components[name][r, c]), 4) for name in COMPONENTS}
        chosen.append(Seed(r, c, x, y, float(score), comps))

    if len(chosen) < cfg_seeds["count_min"]:
        warnings.append(f"seed_count_below_min:{len(chosen)}<{cfg_seeds['count_min']}")
    return chosen, warnings
