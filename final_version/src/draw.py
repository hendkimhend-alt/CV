import cv2
import numpy as np

COLORS = {"crack": (0, 0, 255), "pothole": (255, 0, 0)}


def draw_result(image, road_mask, boxes, title="", max_side=1280):
    """boxes는 원본 좌표 [{"type", "bbox"}]. 큰 사진은 긴 변 max_side로 줄여서 그린다."""
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
    for b in boxes:
        x, y, w, h = (v * scale for v in b["bbox"])
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), COLORS[b["type"]], 2)
    if title:
        cv2.putText(vis, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis
