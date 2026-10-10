"""격자 특징 · 결정적 시드 선택 · Region Growing · 정리 · Road Mask 형식."""
import unittest

import numpy as np

from helpers import config, synthetic_scene
from preprocessing.grid_features import compute_feature_maps, compute_grid_features
from preprocessing.mask_refine import fill_holes, refine_mask
from preprocessing.region_growing import grow_regions
from preprocessing.road_mask import extract_road_mask
from preprocessing.seeds import score_cells, select_seeds


class GridAndSeedTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()["roi"]
        self.img, self.regions = synthetic_scene()
        self.maps = compute_feature_maps(self.img, self.cfg["features"])
        self.grid = compute_grid_features(self.maps, self.cfg["grid"], self.cfg["features"])

    def test_grid_shape_and_features(self):
        self.assertEqual((self.grid.rows, self.grid.cols), (8, 10))   # 240/32 → 8 (마지막 칸 잘림), 320/32 → 10
        f = self.grid.features
        self.assertAlmostEqual(f["fill"][-1, 0], (240 - 7 * 32) / 32)
        self.assertLess(f["texture"][0, 5], f["texture"][6, 5])        # 하늘이 노면보다 매끈
        self.assertGreater(f["chroma"][0, 5], f["chroma"][6, 5])       # 하늘이 노면보다 유채색
        for name in ("cx", "cy"):
            self.assertTrue(np.all((f[name] >= 0) & (f[name] <= 1)))

    def test_seed_selection_is_deterministic_and_on_road(self):
        runs = []
        for _ in range(2):
            scores, comps = score_cells(self.grid, self.cfg["seeds"], self.cfg["features"]["lightness_weight"])
            seeds, _ = select_seeds(self.grid, scores, comps, self.cfg["seeds"], self.cfg["grid"])
            runs.append([(s.row, s.col, s.score) for s in seeds])
        self.assertEqual(runs[0], runs[1])
        self.assertGreaterEqual(len(runs[0]), self.cfg["seeds"]["count_min"])
        self.assertLessEqual(len(runs[0]), self.cfg["seeds"]["count_max"])
        for row, col, _ in runs[0]:
            ys, xs = self.grid.cell_slice(row, col)
            self.assertGreater(self.regions["road"][ys, xs].mean(), 0.9, f"시드 ({row},{col})가 노면이 아님")
        spacing = self.cfg["seeds"]["min_distance_cells"]
        cells = [(r, c) for r, c, _ in runs[0]]
        for i, a in enumerate(cells):
            for b in cells[i + 1:]:
                self.assertGreaterEqual(max(abs(a[0] - b[0]), abs(a[1] - b[1])), spacing)

    def test_region_growing_stays_on_road(self):
        scores, comps = score_cells(self.grid, self.cfg["seeds"], self.cfg["features"]["lightness_weight"])
        seeds, _ = select_seeds(self.grid, scores, comps, self.cfg["seeds"], self.cfg["grid"])
        grown, records = grow_regions(self.maps, self.grid, seeds, self.cfg["region_growing"],
                                      self.cfg["features"]["lightness_weight"])
        self.assertTrue(all(r["grown"] for r in records))
        self.assertLess((grown & self.regions["sky"]).sum() / self.regions["sky"].sum(), 0.01)
        self.assertLess((grown & self.regions["grass"]).sum() / self.regions["grass"].sum(), 0.05)
        self.assertGreater((grown & self.regions["road"]).sum() / self.regions["road"].sum(), 0.6)


class RefineAndMaskTests(unittest.TestCase):
    def test_fill_holes_respects_area_and_border(self):
        mask = np.zeros((50, 50), bool)
        mask[10:40, 10:40] = True
        mask[20:23, 20:23] = False      # 작은 구멍 9px
        mask[12:30, 30:38] = False      # 큰 구멍 144px
        mask[0:5, 0:50] = True
        mask[0:3, 20:25] = False        # 테두리에 닿은 배경 → 구멍 아님
        filled = fill_holes(mask, max_area=20)
        self.assertTrue(filled[21, 21])
        self.assertFalse(filled[20, 33])
        self.assertFalse(filled[1, 22])
        self.assertTrue(fill_holes(mask)[20, 33])

    def test_refine_keeps_separate_large_components(self):
        cfg = config()["roi"]["refine"]
        mask = np.zeros((100, 100), bool)
        mask[10:90, 5:40] = True        # 떨어진 두 노면
        mask[10:90, 60:95] = True
        mask[50, 50] = True             # 고립된 점
        out = refine_mask(mask, cfg)
        self.assertTrue(out[50, 20] and out[50, 80])
        self.assertFalse(out[50, 50])

    def test_extract_road_mask_format_and_coverage(self):
        img, regions = synthetic_scene()
        result = extract_road_mask(img, config()["roi"])
        mask = result.mask
        self.assertEqual(mask.shape, img.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask).tolist()) <= {0, 255})
        road = mask > 0
        self.assertGreater((road & regions["road"]).sum() / regions["road"].sum(), 0.85)
        self.assertLess((road & regions["sky"]).sum() / regions["sky"].sum(), 0.01)
        self.assertLess((road & regions["grass"]).sum() / regions["grass"].sum(), 0.05)
        # 노면 안쪽 균열은 정리 단계(closing · 구멍 채우기)로 마스크 안에 남는다
        self.assertTrue(road[150, 48 + 100 + int(10 * np.sin(150 / 9.0))])

    def test_downscaled_work_image_maps_back_to_original(self):
        img, regions = synthetic_scene(height=480, width=640)
        result = extract_road_mask(img, config(**{"roi.work_long_side": 320})["roi"])
        self.assertAlmostEqual(result.work_scale, 0.5)
        self.assertEqual(result.mask.shape, (480, 640))
        for seed in result.seeds:      # 시드 좌표는 원본 좌표
            self.assertTrue(regions["road"][int(seed["y"]), int(seed["x"])])


if __name__ == "__main__":
    unittest.main()
