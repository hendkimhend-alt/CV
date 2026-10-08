"""
개발 B - 검출 결과 시각화. 손상 종류별 색으로 박스를 그린다.
"""
import cv2

from paths import imwrite

COLORS = {"crack": (0, 0, 255), "pothole": (255, 0, 0), "noise": (128, 128, 128)}  # BGR


def draw_detections(img, detections, title=None):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        color = COLORS.get(det["type"], (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    if title:
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def save(path, img):
    imwrite(path, img)
