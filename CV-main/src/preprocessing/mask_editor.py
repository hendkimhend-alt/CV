"""수동 도로 마스크: 다각형 편집기(matplotlib)와 외부 마스크 파일 읽기.

편집기 키: 클릭 꼭짓점 추가 · a 더하기 · d 빼기 · r 바꾸기 · e 외곽선 불러오기 · Esc 다시 그리기 · Enter 확정 · q 취소
"""
from pathlib import Path

import cv2
import numpy as np

from .image_io import ROAD_VALUE, read_mask, to_binary_mask

MODES = ("replace", "add", "subtract")
KEY_MODES = {"r": "replace", "a": "add", "d": "subtract"}


def polygon_to_mask(points, shape):
    mask = np.zeros(shape[:2], np.uint8)
    if len(points) >= 3:
        pts = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], ROAD_VALUE)
    return mask


def combine_masks(base, polygon_mask, mode):
    if mode not in MODES:
        raise ValueError(f"편집 모드는 {MODES} 중 하나")
    if mode == "replace":
        return polygon_mask.copy()
    if mode == "add":
        return to_binary_mask((base > 0) | (polygon_mask > 0))
    return to_binary_mask((base > 0) & ~(polygon_mask > 0))


def mask_to_polygon(mask, epsilon_ratio=0.003):
    """가장 큰 외곽선을 단순화한 다각형."""
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * cv2.arcLength(contour, True), True)
    return [(float(x), float(y)) for x, y in approx.reshape(-1, 2)]


def mask_file_candidates(mask_dir, image_name):
    stem = Path(image_name).stem
    mask_dir = Path(mask_dir)
    return [mask_dir / f"{stem}.png", mask_dir / f"{stem}_road_mask.png", mask_dir / stem / "road_mask.png"]


def find_external_mask(mask_dir, image_name):
    if not mask_dir:
        return None
    for path in mask_file_candidates(mask_dir, image_name):
        if path.is_file():
            return path
    return None


def load_external_mask(path, shape, cfg_manual):
    mask = read_mask(path, shape, cfg_manual["mask_threshold"])
    ratio = np.count_nonzero(mask) / mask.size
    if ratio < cfg_manual["min_area_ratio"]:
        raise ValueError(f"수동 마스크의 도로 면적 비율 {ratio:.5f}이 최소값 "
                         f"{cfg_manual['min_area_ratio']}보다 작음 ({path})")
    return mask


class MaskEditor:
    def __init__(self, image, initial_mask=None, title=""):
        self.image = image
        if initial_mask is None:
            self.base = np.zeros(image.shape[:2], np.uint8)
        else:
            self.base = to_binary_mask(initial_mask)
        self.polygon = []
        self.mode = "add"
        self.title = title
        self.result = None

    def set_polygon(self, points):
        self.polygon = [(float(x), float(y)) for x, y in points]

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"편집 모드는 {MODES} 중 하나")
        self.mode = mode

    def load_outline(self):
        self.polygon = mask_to_polygon(self.base)
        self.mode = "replace"
        return self.polygon

    def current_mask(self):
        # 다각형을 그리기 전에는 기존 마스크 그대로
        if len(self.polygon) < 3:
            return self.base.copy()
        return combine_masks(self.base, polygon_to_mask(self.polygon, self.image.shape), self.mode)

    def confirm(self):
        self.result = self.current_mask()
        return self.result

    def run(self):
        import matplotlib
        if matplotlib.get_backend().lower() == "agg":
            raise RuntimeError("GUI 백엔드가 없어 편집기를 열 수 없음 — --manual-mask-dir를 사용하세요")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import PolygonSelector

        font = _korean_font()
        rc = {"font.family": font, "axes.unicode_minus": False} if font else {}
        with plt.rc_context(rc):
            figure, axis = plt.subplots(figsize=(11, 8))
        axis.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        overlay = axis.imshow(self._overlay_rgba(), interpolation="nearest")
        axis.set_axis_off()

        if font:
            help_text = "Enter 확정 · q 취소 · a 더하기 / d 빼기 / r 바꾸기 · e 외곽선 불러오기 · Esc 다시 그리기"
        else:
            help_text = "Enter=confirm  q=cancel  a=add / d=subtract / r=replace  e=load outline  Esc=redraw"
        title_font = {"fontfamily": font} if font else {}

        def update_title():
            axis.set_title(f"{self.title}\n[{self.mode}] {help_text}", fontsize=10, **title_font)

        def refresh(*_):
            self.polygon = [(float(x), float(y)) for x, y in selector.verts]
            overlay.set_data(self._overlay_rgba())
            update_title()
            figure.canvas.draw_idle()

        selector = PolygonSelector(axis, refresh, useblit=False,
                                   props={"color": "yellow", "linewidth": 1.5},
                                   handle_props={"markersize": 5, "markerfacecolor": "yellow"})
        update_title()

        def on_key(event):
            if event.key in KEY_MODES:
                self.mode = KEY_MODES[event.key]
                refresh()
            elif event.key == "e":
                if len(self.load_outline()) >= 3:
                    selector.verts = self.polygon
                refresh()
            elif event.key == "enter":
                self.polygon = [(float(x), float(y)) for x, y in selector.verts]
                self.confirm()
                plt.close(figure)
            elif event.key == "q":
                self.result = None
                plt.close(figure)

        figure.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        return self.result

    def _overlay_rgba(self):
        mask = self.current_mask() > 0
        rgba = np.zeros(mask.shape + (4,), np.float32)
        rgba[mask] = (0.0, 1.0, 0.0, 0.35)
        return rgba


def _korean_font():
    try:
        from matplotlib import font_manager
    except ImportError:
        return None
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"):
        if name in names:
            return name
    return None


def edit_mask_interactive(image, initial_mask=None, title=""):
    """확정하면 마스크(uint8 0/255), 취소하면 None."""
    return MaskEditor(image, initial_mask, title).run()
