"""시드마다 색 · 밝기 · 기울기 · 질감 조건을 만족하는 연결 화소로 영역을 넓힌다."""
import cv2
import numpy as np


def seed_reference(maps, grid, seed):
    ys, xs = grid.cell_slice(seed.row, seed.col)
    lab = maps.lab[ys, xs].reshape(-1, 3).astype(np.float64)
    std = lab.std(0)
    return {"lab": lab.mean(0),
            "sigma_ab": float(np.hypot(std[1], std[2])),
            "texture": float(np.median(maps.texture[ys, xs]))}


def color_threshold(reference, cfg_rg):
    t = cfg_rg["color_threshold"] + cfg_rg["seed_std_scale"] * reference["sigma_ab"]
    return min(t, cfg_rg["max_color_threshold"])


def acceptance_mask(maps, reference, cfg_rg, lightness_weight):
    lab = maps.lab
    d_l = lab[..., 0] - reference["lab"][0]
    d_a = lab[..., 1] - reference["lab"][1]
    d_b = lab[..., 2] - reference["lab"][2]
    # 밝기 차이는 lightness_weight만큼만 반영 → 그림자 진 노면도 받아들일 수 있게
    distance = np.sqrt((lightness_weight * d_l) ** 2 + d_a ** 2 + d_b ** 2)

    accept = distance <= color_threshold(reference, cfg_rg)
    accept &= np.abs(d_l) <= cfg_rg["max_lightness_diff"]          # 흰 차선, 하늘
    if cfg_rg["max_gradient"] is not None:
        accept &= maps.gradient <= cfg_rg["max_gradient"]          # 연석, 차량 윤곽
    lo, hi = cfg_rg["texture_ratio_range"]
    ref_tex = max(reference["texture"], 1e-6)
    accept &= (maps.texture >= lo * ref_tex) & (maps.texture <= hi * ref_tex)
    return accept


def grow_regions(maps, grid, seeds, cfg_rg, lightness_weight):
    """→ (모든 시드 영역의 합집합, 시드별 기록)."""
    union = np.zeros(maps.gradient.shape, bool)
    records = []
    for seed in seeds:
        reference = seed_reference(maps, grid, seed)
        accept = acceptance_mask(maps, reference, cfg_rg, lightness_weight)
        # 조건을 만족하는 화소의 연결 요소 중 시드 칸에 걸친 것 = BFS로 넓힌 결과와 같음
        _, labels = cv2.connectedComponents(accept.astype(np.uint8), connectivity=cfg_rg["connectivity"])
        ys, xs = grid.cell_slice(seed.row, seed.col)
        cell_labels = labels[ys, xs]
        cell_labels = cell_labels[cell_labels > 0]

        record = {"row": seed.row, "col": seed.col, "grown": False, "pixels": 0, "cell_coverage": 0.0,
                  "color_threshold": round(color_threshold(reference, cfg_rg), 3)}
        if cell_labels.size:
            values, counts = np.unique(cell_labels, return_counts=True)
            best = int(values[np.argmax(counts)])
            coverage = float(counts.max()) / float((ys.stop - ys.start) * (xs.stop - xs.start))
            record["cell_coverage"] = round(coverage, 4)
            if coverage >= cfg_rg["min_seed_cell_coverage"]:
                region = labels == best
                union |= region
                record.update(grown=True, pixels=int(region.sum()))
        records.append(record)
    return union, records
