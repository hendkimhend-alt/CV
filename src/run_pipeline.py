"""
전체 실행: ① 사진 고치기(A) → ② 찾기(B) → ③ 재기.

사진마다:
  노면 영역·1024 (A geometry_preprocess) → 보정 전 지표로 그룹 분류 (흐림 / 국소 조도 / 정상)
  → 조건 P0 / P1 / P1+ 보정 (A preprocess_condition) → 품질 지표 (A measure_quality)
  → 검출기 D0 / D1 (B detect)
  → 과제 비교 항목: 에지 수, 특징점 수(SIFT), 후보 수·면적, 처리 시간
  → 정답이 있으면 판정 기준 3개(iou50 / iou30 / in50)로 TP·FP·FN → precision·recall·F1 (evaluate)
    ROI 밖으로 잘린 정답은 놓침(FN)으로 센다 (count_cut)
  → 오검출: FPPI = 사진 1장당 FP (판정 기준별) / 깨끗한 노면 FPPI = 손상 라벨이 없는 사진 1장당 후보 수
  → 오차 범위: 사진 단위 부트스트랩으로 F1 95% 범위, 기준 조건(--reference, 기본 P1+/D1) 대비 차이 (stats)
  → 진단용 구분력: 정답 박스 TPR − 옆 노면 대조 박스 FPR (판정에는 안 씀, evaluate.separation)
  → 집계 묶음: 전체 / 품질 그룹 / RDD면 시점(vp_far · vp_road_full · vp_ambiguous, labels/viewpoint_rdd.csv)
    시점 라벨은 결과를 나눠 보는 데만 쓰고 검출 방법은 보지 않는다

출력 (outputs/pipeline/run_<시각>/):
  results.csv   사진 × 조건 × 검출기 한 줄씩
  summary.csv   조건 × 검출기 × 그룹 평균, precision·recall·F1은 TP·FP·FN 전체 합산 (판정 기준별), F1 95% 범위
  compare.csv   기준 조건 대비 F1 차이와 95% 범위, 의미 있는 차이인지 (그룹 × 종류 × 판정 기준)
  images/       결과 박스 그림 (--save-images N: 처음 N장, 0 = 저장 안 함, -1 = 전부)
  run_config.json  실행 설정·버전

실행:
  python src/run_pipeline.py                      # 제공 13장
  python src/run_pipeline.py --dataset rdd_dev    # RDD 개발 세트 563장 (정답 있음 → precision·recall·F1)
  python src/run_pipeline.py --dataset rdd --limit 50 --save-images 10
"""
import argparse
import csv
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

from data import list_images, load_gt, load_viewpoints
from detect import DEFAULT_CFG as DETECT_CFG, detect
from evaluate import CRITERIA, KINDS, count_cut, evaluate, prf, separation, transform_gt
from metrics import measure_quality
from paths import OUTPUT_DIR, imread, imwrite
from stats import bootstrap, row_tags
from preprocess import CONDITIONS, classify_quality, geometry_preprocess, preprocess_condition, validate_config
from visualize import draw_detections

DETECTORS = ("D0", "D1")
QUALITY_KEYS = ("gray_mean", "block_mean_std_4x4", "laplacian_variance", "noise_sigma", "saturation_ratio")
CANNY_EDGES = (100, 200)        # 에지 수 = 13장 실측(표1)과 같은 Canny 기준


def new_run_dir(parent):
    """run_<시각> 폴더를 새로 만든다. 같은 초에 이미 있으면 _2, _3 …을 붙여 앞 결과를 덮어쓰지 않는다."""
    base = datetime.now(timezone(timedelta(hours=9))).strftime("run_%Y%m%d_%H%M%S")
    parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        run_dir = parent / (base if n == 1 else f"{base}_{n}")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"error:결과 폴더를 만들 수 없음 {parent / base}")


def count_edges(gray):
    return int((cv2.Canny(gray, *CANNY_EDGES) > 0).sum())


def run(dataset="provided", limit=None, conditions=CONDITIONS, detectors=DETECTORS,
        keypoints=True, save_images=5, n_boot=1000, reference=("P1+", "D1"), roi="bottom_half"):
    name, folder, images = list_images(dataset)
    images = images[:limit] if limit else images
    gt_loader = load_gt(dataset, name)
    viewpoints = load_viewpoints(dataset)
    cfg_a = validate_config({"roi": roi})
    sift = cv2.SIFT_create() if keypoints else None

    run_dir = new_run_dir(OUTPUT_DIR / "pipeline")
    rows = []
    print(f"{name}: {len(images)}장 × 조건 {len(conditions)} × 검출기 {len(detectors)} · ROI {roi} → {run_dir}", flush=True)

    for i, path in enumerate(images, 1):
        try:
            img = imread(path)
            t0 = time.perf_counter()
            ref, geometry = geometry_preprocess(img, cfg_a, path.name)
            geometry_ms = (time.perf_counter() - t0) * 1000
            ref_quality = measure_quality(ref)
            tags = classify_quality(ref_quality)
        except (ValueError, cv2.error) as exc:
            print(f"  건너뜀 {path.name}: {exc}", flush=True)
            continue
        gt = gt_loader(path)
        gt_t = transform_gt(gt, geometry) if gt is not None else None
        cut = count_cut(gt, gt_t) if gt is not None else None     # ROI 밖으로 잘린 정답 → 놓침
        gamma_cache = {}

        for cond in conditions:
            t0 = time.perf_counter()
            fixed, _ = preprocess_condition(ref, cfg_a, cond, tags, gamma_cache)
            pre_ms = geometry_ms + (time.perf_counter() - t0) * 1000
            quality = ref_quality if cond == "P0_reference" else measure_quality(fixed)
            gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
            n_edges = count_edges(gray)
            n_kp = len(sift.detect(gray, None)) if sift else None

            for det_name in detectors:
                t0 = time.perf_counter()
                dets = detect(fixed, {"detector": det_name})
                det_ms = (time.perf_counter() - t0) * 1000
                row = {"image": path.name, "dataset": name, "group": ";".join(tags),
                       "viewpoint": viewpoints.get(path.name, ""), "condition": cond,
                       "detector": det_name, "preprocess_ms": round(pre_ms, 2), "detect_ms": round(det_ms, 2),
                       "n_edges": n_edges, "n_keypoints": n_kp, "has_gt": gt_t is not None,
                       "clean": gt is not None and len(gt) == 0}      # 원본에 손상 라벨이 하나도 없는 사진
                row.update({k: quality[k] for k in QUALITY_KEYS})
                for kind in KINDS:
                    ds = [d for d in dets if d["type"] == kind]
                    row[f"n_{kind}"] = len(ds)
                    row[f"area_{kind}"] = round(sum(d["area"] for d in ds))
                if gt_t is not None:
                    for kind, e in evaluate(dets, gt_t, cut).items():
                        row[f"{kind}_n_gt"] = e["n_gt"]
                        row[f"{kind}_n_cut"] = e["n_cut"]
                        for c in CRITERIA:
                            row.update({f"{kind}_{c}_{k}": v for k, v in e[c].items()})
                    for kind, e in separation(dets, gt_t, fixed.shape[1], fixed.shape[0]).items():
                        row.update({f"{kind}_sep_{k}": v for k, v in e.items()})
                rows.append(row)

                if save_images < 0 or i <= save_images:
                    vis = draw_detections(fixed, dets, f"{cond} {det_name}")
                    for _, x1, y1, x2, y2 in gt_t or []:     # 정답 = 초록
                        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 1)
                    imwrite(run_dir / "images" / f"{cond.replace('+', '_plus')}_{det_name}" / f"{path.stem}.png", vis)

        if i == 1 or i % 25 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] {path.name}", flush=True)

    if not rows:
        raise ValueError("error:처리된 사진 없음")
    summary = summarize(rows)
    compare = []
    if n_boot and any(r["has_gt"] for r in rows):
        ci, compare = bootstrap(rows, tuple(reference), n_boot)
        for item in summary:
            for kind in KINDS:
                for c in CRITERIA:
                    lo, hi = ci.get((item["condition"], item["detector"], item["group"], kind, c), (None, None))
                    item[f"{kind}_{c}_f1_lo"] = None if lo is None else round(lo, 4)
                    item[f"{kind}_{c}_f1_hi"] = None if hi is None else round(hi, 4)
    write_csv(run_dir / "results.csv", rows)
    write_csv(run_dir / "summary.csv", summary)
    if compare:
        write_csv(run_dir / "compare.csv", compare)
    (run_dir / "run_config.json").write_text(json.dumps({
        "dataset": name, "folder": str(folder), "n_images": len(images), "conditions": list(conditions),
        "detectors": list(detectors), "preprocess_cfg": cfg_a, "detect_cfg": DETECT_CFG,
        "edges": f"Canny{CANNY_EDGES}", "keypoints": "SIFT" if keypoints else None,
        "bootstrap": n_boot, "reference": list(reference),
        "versions": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(summary)
    print_compare(compare)
    print(f"완료 → {run_dir}")
    return run_dir


def summarize(rows):
    buckets = defaultdict(list)
    for r in rows:
        for g in row_tags(r):
            buckets[(r["condition"], r["detector"], g)].append(r)
    numeric = ["preprocess_ms", "detect_ms", "n_edges", "n_keypoints", *QUALITY_KEYS,
               *(f"{p}_{k}" for k in KINDS for p in ("n", "area"))]
    out = []
    for (cond, det, group), rs in sorted(buckets.items(), key=lambda kv: (CONDITIONS.index(kv[0][0]), kv[0][1], kv[0][2])):
        item = {"condition": cond, "detector": det, "group": group, "n_images": len(rs)}
        for k in numeric:
            vals = [r[k] for r in rs if r.get(k) is not None]
            item[f"mean_{k}"] = round(float(np.mean(vals)), 4) if vals else None
        with_gt = [r for r in rs if r["has_gt"]]
        item["n_images_gt"] = len(with_gt)
        clean = [r for r in with_gt if r["clean"]]
        item["n_images_clean"] = len(clean)
        for kind in KINDS:
            n_gt = sum(r[f"{kind}_n_gt"] for r in with_gt)
            n_cut = sum(r[f"{kind}_n_cut"] for r in with_gt)
            item[f"{kind}_n_gt"], item[f"{kind}_n_cut"] = n_gt, n_cut      # 정답 수 / 그중 ROI 밖으로 잘린 수
            item[f"{kind}_cut_ratio"] = round(n_cut / n_gt, 4) if n_gt else None
            for c in CRITERIA:
                tp, fp, fn = (sum(r[f"{kind}_{c}_{k}"] for r in with_gt) for k in ("tp", "fp", "fn"))
                for name, v in zip(("precision", "recall", "f1"), prf(tp, fp, fn)):
                    item[f"{kind}_{c}_{name}"] = None if v is None else round(v, 4)
                item[f"{kind}_{c}_fppi"] = round(fp / len(with_gt), 4) if with_gt else None
            item[f"{kind}_clean_fppi"] = round(sum(r[f"n_{kind}"] for r in clean) / len(clean), 4) if clean else None
            n = sum(r[f"{kind}_sep_n"] for r in with_gt)
            if n:
                tpr = sum(r[f"{kind}_sep_hit"] for r in with_gt) / n
                fpr = sum(r[f"{kind}_sep_fhit"] for r in with_gt) / n
                item.update({f"{kind}_sep_tpr": round(tpr, 4), f"{kind}_sep_fpr": round(fpr, 4),
                             f"{kind}_separation": round(tpr - fpr, 4)})
        out.append(item)
    return out


def write_csv(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def print_table(summary):
    fmt = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
    print(f"\n{'조건':<13}{'검출':<5}{'에지':>8}{'특징점':>8}{'균열후보':>8}"
          + "".join(f"{'균열F1 ' + c:>13}" for c in CRITERIA) + f"{'포트홀F1 in50':>14}{'균열FPPI in50':>14}{'깨끗한노면FPPI':>14}{'전처리ms':>9}")
    for s in summary:
        if s["group"] == "all":
            print(f"{s['condition']:<13}{s['detector']:<5}{fmt(s['mean_n_edges'], 0):>8}{fmt(s['mean_n_keypoints'], 0):>8}"
                  f"{fmt(s['mean_n_crack'], 1):>8}" + "".join(f"{fmt(s[f'crack_{c}_f1']):>13}" for c in CRITERIA)
                  + f"{fmt(s['pothole_in50_f1']):>14}{fmt(s['crack_in50_fppi'], 1):>14}{fmt(s['crack_clean_fppi'], 1):>14}"
                  f"{fmt(s['mean_preprocess_ms'], 1):>9}")


def print_compare(compare):
    """판정 기준 고르기용: 전체 그룹·균열에서 기준 조건과 F1 차이가 의미 있는 조건 수 (기준별)."""
    rows = [c for c in compare if c["group"] == "all" and c["kind"] == "crack"]
    if not rows:
        return
    print(f"\n기준 조건 {rows[0]['reference']} 대비 균열 F1 차이 (전체, 95% 범위가 0을 안 포함하면 *)")
    for crit in CRITERIA:
        rs = [c for c in rows if c["criterion"] == crit]
        cells = "  ".join(f"{c['condition']}/{c['detector']} {c['delta']:+.3f}{'*' if c['significant'] else ' '}" for c in rs)
        print(f"  {crit:<6} 의미 있는 차이 {sum(c['significant'] for c in rs)}/{len(rs)}  |  {cells}")


def main():
    p = argparse.ArgumentParser(description="전체 실행: 전처리(A) → 검출(B) → 평가")
    p.add_argument("--dataset", default="provided", help="provided / captured / rdd / rdd_dev / rdd_test 또는 폴더 경로")
    p.add_argument("--limit", type=int, help="처음 N장만")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    p.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=DETECTORS)
    p.add_argument("--no-keypoints", action="store_true", help="SIFT 특징점 수 생략 (빠르게)")
    p.add_argument("--save-images", type=int, default=5, help="결과 그림 저장 장수 (0 = 안 함, -1 = 전부)")
    p.add_argument("--bootstrap", type=int, default=1000, help="F1 오차 범위 부트스트랩 횟수 (0 = 안 함)")
    p.add_argument("--reference", default="P1+/D1", help="차이를 잴 기준 조건 '전처리/검출기' (기본: 1차 최종 후보)")
    p.add_argument("--roi", default="bottom_half",
                   help="노면 영역: bottom_half(기본) / full(전체) / bottom_<N>(아래쪽 N%%만, 예: bottom_60)")
    a = p.parse_args()
    run(a.dataset, a.limit, tuple(a.conditions), tuple(a.detectors), not a.no_keypoints, a.save_images,
        a.bootstrap, tuple(a.reference.split("/")), a.roi)


if __name__ == "__main__":
    main()
