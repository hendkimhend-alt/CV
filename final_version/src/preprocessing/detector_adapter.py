"""검출기 설정값이 긴 변 1024 기준이라 노면 영역을 잘라 그 크기로 맞춘다.
박스는 나중에 원본 좌표로 되돌린다."""
import cv2


def road_bbox(road_mask, margin):
    ys, xs = (road_mask > 0).nonzero()
    h, w = road_mask.shape
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    return x0, y0, x1 - x0, y1 - y0


def to_detector_input(image, road_mask, long_side, margin):
    height, width = image.shape[:2]
    x0, y0, cw, ch = road_bbox(road_mask, margin)
    image_c = image[y0:y0 + ch, x0:x0 + cw]
    mask_c = road_mask[y0:y0 + ch, x0:x0 + cw]

    scale = long_side / max(ch, cw)
    size = (max(1, round(cw * scale)), max(1, round(ch * scale)))
    if size == (cw, ch):
        det_image, det_mask = image_c.copy(), mask_c.copy()
    else:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        det_image = cv2.resize(image_c, size, interpolation=interp)
        det_mask = cv2.resize(mask_c, size, interpolation=cv2.INTER_NEAREST)

    geometry = {"original_width": width, "original_height": height, "crop_x": x0, "crop_y": y0,
                "scale_x": size[0] / cw, "scale_y": size[1] / ch}
    return det_image, det_mask, geometry


def to_original_bbox(bbox, geometry):
    x, y, w, h = bbox
    sx, sy = geometry["scale_x"], geometry["scale_y"]
    return x / sx + geometry["crop_x"], y / sy + geometry["crop_y"], w / sx, h / sy
