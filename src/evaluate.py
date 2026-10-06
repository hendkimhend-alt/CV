"""
검출 결과를 정답 박스와 비교한다 — 박스 단위 TP / FP / FN → precision · recall · F1.
(계획서상 개발 A 담당 — 전체 실행 파일을 위해 B가 초안 작성, 2차 평가 지표 재정의로 수정)

정답 판정 기준 3개 — 기준선(개선 전 방법)에서 "조건 간 차이를 오차 범위보다 크게 구분하는 기준 중
가장 엄격한 것"을 하나 골라 이후 모든 실험에 고정한다. 세 기준 결과는 모두 기록한다.
- iou50: IoU > 0.5 — 표준 (PASCAL VOC · RDD 대회 GRDDC 2020과 같은 정의)
- iou30: IoU > 0.3 — 중간
- in50 : 후보 면적의 절반 이상이 정답 박스 안이면 정답. 같은 정답 안의 추가 조각은 TP도 FP도 아님(제외)
         → 균열을 조각으로 잡는 검출기 특성은 봐주되, 큰 박스(조금만 걸쳐도 정답)나
           조각 수(잘게 쪼갤수록 정답 수 증가)로 점수를 올리는 꼼수는 막는다
공통: 정답 하나에 TP는 최대 하나 (겹침이 큰 쌍부터 짝짓기).
      iou50 · iou30은 표준대로 짝을 못 찾은 후보(중복 조각 포함)는 모두 FP.
      recall = TP / 정답 수, precision = TP / (TP + FP), F1 = 둘의 조화평균
좌표는 둘 다 같은 좌표계(전처리 후 노면 영역)여야 한다 → transform_gt()로 맞춘다.

진단용 — 구분력 (separation(), 판정에는 안 씀. 실패 원인 분석용)
- 정답 박스마다 바로 옆 노면(좌·우·아래·위, 어떤 정답과도 안 겹침)에 같은 크기 대조 박스를 둔다
- 후보 면적의 50% 이상이 박스 안이면 그 박스 소속 (analysis/eval_detector.py와 같은 규칙)
- 구분력 = 정답 박스에 같은 종류 후보가 나온 비율(TPR) − 대조 박스에 나온 비율(FPR)
  → 통계의 Youden's J(민감도 + 특이도 − 1)와 같은 형태. 0 = 손상과 옆 노면을 못 가림
"""

KINDS = ("crack", "pothole")
CRITERIA = ("iou50", "iou30", "in50")
IOU_THRESHOLDS = {"iou50": 0.5, "iou30": 0.3}
INSIDE_FRAC = 0.5           # in50: 후보 면적 중 정답 박스 안 비율
SEP_MIN_BOX_AREA = 400      # 구분력: 이보다 작은 정답 박스는 대조 박스를 두지 않음 (eval_detector와 같음)


def transform_gt(boxes, geometry):
    """원본 좌표 정답 → preprocess 출력 좌표 (노면 영역 자르기 + 크기 조절). 노면 영역 밖 박스는 버림."""
    out = []
    W, H = geometry["output_width"], geometry["output_height"]
    for kind, x1, y1, x2, y2 in boxes:
        nx1 = max(0.0, (x1 - geometry["crop_x"]) * geometry["scale_x"])
        ny1 = max(0.0, (y1 - geometry["crop_y"]) * geometry["scale_y"])
        nx2 = min(W, (x2 - geometry["crop_x"]) * geometry["scale_x"])
        ny2 = min(H, (y2 - geometry["crop_y"]) * geometry["scale_y"])
        if nx2 > nx1 and ny2 > ny1:
            out.append((kind, nx1, ny1, nx2, ny2))
    return out


def _overlaps(det_bbox, gt):
    """후보 bbox(x, y, w, h)가 정답 박스에 조금이라도 걸치나 (1차 판정 방식 — 오검출 분석 스크립트용으로 남김)."""
    x, y, w, h = det_bbox
    _, gx1, gy1, gx2, gy2 = gt
    return x < gx2 and x + w > gx1 and y < gy2 and y + h > gy1


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _pair_greedy(pairs):
    """(점수, 후보 i, 정답 j) 목록 → 점수가 큰 쌍부터 1:1로 짝지은 후보·정답 번호."""
    used_d, used_g = set(), set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_d and j not in used_g:
            used_d.add(i)
            used_g.add(j)
    return used_d, used_g


def _match_iou(dets, gts, threshold):
    pairs = []
    for i, d in enumerate(dets):
        for j, g in enumerate(gts):
            inter = _inter(d, g)
            if inter:
                iou = inter / (_area(d) + _area(g) - inter)
                if iou > threshold:
                    pairs.append((iou, i, j))
    _, used_g = _pair_greedy(pairs)
    tp = len(used_g)
    return tp, len(dets) - tp, len(gts) - tp


def _match_inside(dets, gts):
    pairs, inside_any = [], [False] * len(dets)
    for i, d in enumerate(dets):
        area = max(_area(d), 1.0)
        for j, g in enumerate(gts):
            frac = _inter(d, g) / area
            if frac >= INSIDE_FRAC:
                pairs.append((frac, i, j))
                inside_any[i] = True
    used_d, used_g = _pair_greedy(pairs)
    tp = len(used_g)
    # 짝을 못 찾았는데 정답 안에 있는 후보 = 이미 짝지어진 정답의 추가 조각 → 제외
    ignored = sum(inside_any[i] and i not in used_d for i in range(len(dets)))
    return tp, len(dets) - tp - ignored, len(gts) - tp


def _xyxy(bbox):
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def evaluate(detections, gt_boxes):
    """detections: detect() 결과, gt_boxes: transform_gt() 결과
    → {kind: {"n_gt", "n_det", 기준: {"tp", "fp", "fn"}}} (사진 한 장)"""
    res = {}
    for kind in KINDS:
        dets = [_xyxy(d["bbox"]) for d in detections if d["type"] == kind]
        gts = [tuple(g[1:]) for g in gt_boxes if g[0] == kind]
        r = {"n_gt": len(gts), "n_det": len(dets)}
        for c in CRITERIA:
            tp, fp, fn = (_match_inside(dets, gts) if c == "in50"
                          else _match_iou(dets, gts, IOU_THRESHOLDS[c]))
            r[c] = {"tp": tp, "fp": fp, "fn": fn}
        res[kind] = r
    return res


def prf(tp, fp, fn):
    """TP · FP · FN 합계 → (precision, recall, F1). 분모가 0이면 None."""
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    if p is None or r is None:
        return p, r, None
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _control_box(box, boxes, W, H):
    """정답 박스 바로 옆(좌·우·아래·위 순)에 같은 크기, 어떤 정답과도 안 겹치는 대조 박스. 없으면 None."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    for cx, cy in [(x1 - bw, y1), (x2, y1), (x1, y2), (x1, y1 - bh)]:
        if cx < 0 or cy < 0 or cx + bw > W or cy + bh > H:
            continue
        if all(cx + bw <= b[0] or cx >= b[2] or cy + bh <= b[1] or cy >= b[3] for b in boxes):
            return (cx, cy, cx + bw, cy + bh)
    return None


def _has_member(dets, region):
    """region 안에 면적의 50% 이상이 들어간 후보가 있나."""
    return any(_inter(d, region) >= INSIDE_FRAC * max(_area(d), 1.0) for d in dets)


def separation(detections, gt_boxes, W, H):
    """진단용 구분력 재료 (사진 한 장) → {kind: {"n": 대조 박스를 둔 정답 수, "hit": 정답 쪽 검출, "fhit": 대조 쪽 검출}}"""
    all_boxes = [tuple(g[1:]) for g in gt_boxes]
    res = {}
    for kind in KINDS:
        dets = [_xyxy(d["bbox"]) for d in detections if d["type"] == kind]
        n = hit = fhit = 0
        for g in gt_boxes:
            box = tuple(g[1:])
            if g[0] != kind or _area(box) < SEP_MIN_BOX_AREA:
                continue
            ctl = _control_box(box, all_boxes, W, H)
            if ctl is None:
                continue
            n += 1
            hit += _has_member(dets, box)
            fhit += _has_member(dets, ctl)
        res[kind] = {"n": n, "hit": hit, "fhit": fhit}
    return res
