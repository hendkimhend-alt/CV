"""전처리 결과를 검출기(detect.py) 입력으로 바꾼다. 검출기는 긴 변 1024 영상을 가정한다."""
import cv2

from .image_io import validate_image, validate_mask

DETECTOR_LONG_SIDE = 1024


def road_bbox(road_mask, margin=0):
    """도로 영역의 외접 사각형 (x, y, w, h). 도로가 없으면 None."""
    ys, xs = (road_mask > 0).nonzero()
    if xs.size == 0:
        return None
    h, w = road_mask.shape
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    return x0, y0, x1 - x0, y1 - y0


def to_detector_input(processed_image, road_mask, long_side=DETECTOR_LONG_SIDE, crop_to_road=False, crop_margin=0):
    """→ (검출기 영상, 검출기 마스크, geometry). crop_to_road면 도로 영역만 잘라서 맞춘다."""
    validate_image(processed_image)
    validate_mask(road_mask, processed_image.shape)
    height, width = processed_image.shape[:2]

    x0, y0, cw, ch = 0, 0, width, height
    if crop_to_road:
        box = road_bbox(road_mask, crop_margin)
        if box is None:
            raise ValueError("road_mask가 비어 있음")
        x0, y0, cw, ch = box
    image_c = processed_image[y0:y0 + ch, x0:x0 + cw]
    mask_c = road_mask[y0:y0 + ch, x0:x0 + cw]

    scale = long_side / max(ch, cw)
    size = (max(1, round(cw * scale)), max(1, round(ch * scale)))
    if size == (cw, ch):
        image, mask, method = image_c.copy(), mask_c.copy(), "none"
    else:
        method = "INTER_AREA" if scale < 1 else "INTER_CUBIC"
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        image = cv2.resize(image_c, size, interpolation=interp)
        mask = cv2.resize(mask_c, size, interpolation=cv2.INTER_NEAREST)

    geometry = {"original_width": width, "original_height": height,
                "crop_x": x0, "crop_y": y0, "crop_width": cw, "crop_height": ch,
                "output_width": size[0], "output_height": size[1],
                "scale_x": size[0] / cw, "scale_y": size[1] / ch,
                "resize_interpolation": method, "mask_interpolation": "INTER_NEAREST",
                "crop_to_road": bool(crop_to_road)}
    return image, mask, geometry


def to_original_bbox(bbox, geometry):
    """검출기 좌표 박스 (x, y, w, h) → 원본 좌표."""
    x, y, w, h = bbox
    sx, sy = geometry["scale_x"], geometry["scale_y"]
    return x / sx + geometry["crop_x"], y / sy + geometry["crop_y"], w / sx, h / sy
