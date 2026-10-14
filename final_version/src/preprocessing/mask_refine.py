"""Region Growing 결과 정리: closing, opening, 작은 구멍 채우기, 작은 조각 제거."""
import cv2
import numpy as np


def _ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def fill_holes(mask, max_area=None):
    """테두리에 닿지 않는 배경 영역(구멍)을 채운다. max_area가 있으면 그보다 작은 구멍만."""
    background = (~mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    if count <= 1:
        return mask.copy()
    edges = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    border = set(np.unique(edges).tolist())
    filled = mask.copy()
    for label in range(1, count):
        if label in border:
            continue
        if max_area is None or stats[label, cv2.CC_STAT_AREA] <= max_area:
            filled |= labels == label
    return filled


def remove_small_components(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def refine_mask(mask, cfg_refine):
    area = mask.size
    out = mask.astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _ellipse(cfg_refine["close_kernel"]),
                           borderType=cv2.BORDER_REPLICATE)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _ellipse(cfg_refine["open_kernel"]),
                           borderType=cv2.BORDER_REPLICATE)
    out = fill_holes(out.astype(bool), cfg_refine["fill_holes_max_area_ratio"] * area)
    # 제일 큰 덩어리만 남기면 중앙분리대로 나뉜 도로 한쪽이 날아가서 작은 조각만 지운다
    return remove_small_components(out, cfg_refine["min_component_area_ratio"] * area)
