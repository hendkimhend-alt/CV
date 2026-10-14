"""시드마다 색, 밝기, 기울기, 질감 조건을 만족하는 연결 화소로 영역을 넓힌다."""
import cv2
import numpy as np


def seed_reference(maps, grid, seed):
    """시드 칸의 기준값: Lab 평균, a · b 퍼짐, 질감 중앙값."""
    ys, xs = grid.cell_slice(seed.row, seed.col)
    lab = maps.lab[ys, xs].reshape(-1, 3).astype(np.float64)
    std = lab.std(0)
    return {"lab": lab.mean(0),
            "sigma_ab": float(np.hypot(std[1], std[2])),
            "texture": float(np.median(maps.texture[ys, xs]))}


def color_threshold(reference, cfg_rg):
    """색 거리 기준. 시드 칸의 색 퍼짐만큼 넓히되 상한을 둔다."""
    t = cfg_rg["color_threshold"] + cfg_rg["seed_std_scale"] * reference["sigma_ab"]
    return min(t, cfg_rg["max_color_threshold"])


def acceptance_mask(maps, reference, cfg_rg, lightness_weight, seed_y):
    """시드와 비교해 노면으로 받을 화소 (색, 밝기, 그림자, 기울기, 질감 조건)."""
    lab = maps.lab
    d_l = lab[..., 0] - reference["lab"][0]
    d_a = lab[..., 1] - reference["lab"][1]
    d_b = lab[..., 2] - reference["lab"][2]
    # 밝기 차는 덜 봐야 그림자 진 노면도 들어온다
    distance = np.sqrt((lightness_weight * d_l) ** 2 + d_a ** 2 + d_b ** 2)

    accept = distance <= color_threshold(reference, cfg_rg)
    accept &= np.abs(d_l) <= cfg_rg["max_lightness_diff"]          # 흰 차선, 하늘

    shadow_cfg = cfg_rg["shadow"]
    # 그림자 진 곳은 어둡고 파래진다(b가 음수 쪽). 그쪽 색 차는 덜 본다
    d_b_eff = np.where(d_b < 0, d_b * shadow_cfg["blue_shift_weight"], d_b)
    shadow = ((d_l < 0) & (-d_l <= shadow_cfg["max_darkening"])
              & (np.hypot(d_a, d_b_eff) <= shadow_cfg["color_threshold"]))
    # 먼 경치도 어둡고 파래 보여서 시드보다 많이 위쪽은 그림자로 치지 않는다
    shadow[:max(0, int(seed_y - shadow_cfg["max_rise_ratio"] * lab.shape[0]))] = False
    accept |= shadow
    # 그림자 속은 질감도 약해서 하한을 밝기 비율만큼 낮춘다
    tex_scale = lab[..., 0] / max(reference["lab"][0], 1e-6)
    tex_scale = np.clip(tex_scale, shadow_cfg["min_texture_scale"], 1.0)

    accept &= maps.gradient <= cfg_rg["max_gradient"]              # 연석, 차량 윤곽
    lo, hi = cfg_rg["texture_ratio_range"]
    ref_tex = max(reference["texture"], 1e-6)
    accept &= (maps.texture >= lo * ref_tex * tex_scale) & (maps.texture <= hi * ref_tex)
    return accept


def grow_regions(maps, grid, seeds, cfg_rg, lightness_weight):
    """시드마다 받을 화소 중 시드 칸에 이어진 덩어리를 모아 노면 영역을 만든다."""
    union = np.zeros(maps.gradient.shape, bool)
    for seed in seeds:
        reference = seed_reference(maps, grid, seed)
        accept = acceptance_mask(maps, reference, cfg_rg, lightness_weight, seed.y)
        # 시드에서 BFS로 넓히는 것과 같다: 조건을 만족하는 덩어리 중 시드 칸에 가장 많이 걸친 것
        _, labels = cv2.connectedComponents(accept.astype(np.uint8),
                                            connectivity=cfg_rg["connectivity"])
        ys, xs = grid.cell_slice(seed.row, seed.col)
        cell_labels = labels[ys, xs]
        cell_labels = cell_labels[cell_labels > 0]
        if not cell_labels.size:
            continue
        values, counts = np.unique(cell_labels, return_counts=True)
        coverage = float(counts.max()) / float((ys.stop - ys.start) * (xs.stop - xs.start))
        if coverage >= cfg_rg["min_seed_cell_coverage"]:
            union |= labels == int(values[np.argmax(counts)])
    return union
