"""검토용 그림 — 원본 위에 도로 마스크(초록)와 손상 후보(균열 빨강 · 포트홀 파랑 · 도로 밖 후보 회색)를 그린다."""
from __future__ import annotations

import cv2
import numpy as np

COLORS = {"crack": (0, 0, 255), "pothole": (255, 0, 0)}
OFF_ROAD = (150, 150, 150)


def draw_result(image, road_mask, candidates, title="", max_side=1280, show_off_road=True):
    scale = min(1.0, max_side / max(image.shape[:2]))
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    vis = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    if road_mask is not None:
        road = cv2.resize(road_mask, size, interpolation=cv2.INTER_NEAREST) > 0
        tint = vis.copy()
        tint[road] = (0, 190, 0)
        vis = cv2.addWeighted(vis, 0.7, tint, 0.3, 0)
        contours, _ = cv2.findContours(road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)
    for c in candidates:
        if not c["on_road"] and not show_off_road:
            continue
        x, y, w, h = (v * scale for v in c["bbox_original"])
        color = COLORS.get(c["type"], (0, 255, 0)) if c["on_road"] else OFF_ROAD
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 2 if c["on_road"] else 1)
    if title:
        for i, line in enumerate(title.split("\n")):
            yy = 22 + 22 * i
            cv2.putText(vis, line, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vis, line, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


def review_sheet(images, columns=3, tile=420):
    tiles = []
    for vis in images:
        s = tile / max(vis.shape[:2])
        small = cv2.resize(vis, (max(1, round(vis.shape[1] * s)), max(1, round(vis.shape[0] * s))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[:small.shape[0], :small.shape[1]] = small
        tiles.append(canvas)
    while len(tiles) % columns:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
