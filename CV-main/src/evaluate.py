"""
검출 결과를 정답 박스와 비교한다 — hit/miss. (계획서상 개발 A 담당 — 전체 실행 파일을 위해 B가 초안 작성)

계획서 2-2: 균열은 가늘어 겹친 면적(IoU)으로 재면 부당하게 낮다 → 후보 박스가 정답 박스에 **걸치면** hit.
- 정답 박스(종류별): 같은 종류 후보가 하나라도 걸치면 hit → recall = hit / 정답 수
- 후보: 같은 종류 정답 박스에 걸치면 맞음 → precision = 맞은 후보 / 후보 수
좌표는 둘 다 같은 좌표계(전처리 후 노면 영역)여야 한다 → transform_gt()로 맞춘다.
"""

import cv2
import numpy as np

if __package__:
    from .roi import bbox_intersection_area, geometry_mask
else:
    from roi import bbox_intersection_area, geometry_mask

KINDS = ("crack", "pothole")


def transform_gt(boxes, geometry, roi_mask=None):
    """기존 crop·scale 변환을 유지하고 사다리꼴과 전혀 겹치지 않는 GT도 제외한다.

    일부가 보이는 GT는 기존 직사각형 ROI처럼 유지한다. 25/50/75% 선정 임계값을 추가하지 않는다.
    """
    out = []
    W, H = geometry["output_width"], geometry["output_height"]
    mask = geometry_mask(geometry) if roi_mask is None else roi_mask
    integral = cv2.integral((mask != 0).astype(np.uint8), sdepth=cv2.CV_64F) if mask is not None else None
    for kind, x1, y1, x2, y2 in boxes:
        nx1 = max(0.0, (x1 - geometry["crop_x"]) * geometry["scale_x"])
        ny1 = max(0.0, (y1 - geometry["crop_y"]) * geometry["scale_y"])
        nx2 = min(W, (x2 - geometry["crop_x"]) * geometry["scale_x"])
        ny2 = min(H, (y2 - geometry["crop_y"]) * geometry["scale_y"])
        if nx2 > nx1 and ny2 > ny1:
            if integral is None or bbox_intersection_area(integral, (nx1, ny1, nx2, ny2)) > 0:
                out.append((kind, nx1, ny1, nx2, ny2))
    return out


def _overlaps(det_bbox, gt):
    x, y, w, h = det_bbox
    _, gx1, gy1, gx2, gy2 = gt
    return x < gx2 and x + w > gx1 and y < gy2 and y + h > gy1


def evaluate(detections, gt_boxes):
    """detections: detect() 결과, gt_boxes: transform_gt() 결과 → 종류별 개수와 precision·recall."""
    res = {}
    for kind in KINDS:
        dets = [d for d in detections if d["type"] == kind]
        gts = [g for g in gt_boxes if g[0] == kind]
        hit_gt = sum(any(_overlaps(d["bbox"], g) for d in dets) for g in gts)
        correct = sum(any(_overlaps(d["bbox"], g) for g in gts) for d in dets)
        res[kind] = {"n_gt": len(gts), "n_det": len(dets), "hit_gt": hit_gt, "correct_det": correct,
                     "recall": hit_gt / len(gts) if gts else None,
                     "precision": correct / len(dets) if dets else None}
    return res
