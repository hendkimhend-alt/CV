"""도로 손상 검출 시스템 실행기 — 전처리(자동 Road Mask · 보정) → 검출 → 도로 위 후보만 집계.

사용 예 (저장소 맨 위 폴더, PowerShell)
  py src/run_road_detection.py --input data/RDD2020_train/train/img/Japan_000015.jpg
  py src/run_road_detection.py --dataset rdd_dev --limit 20
  py src/run_road_detection.py --input data/captured
  py src/run_road_detection.py --dataset rdd_dev --limit 20 --on-fail skip        # FAIL 마스크 사진은 건너뜀 (기본은 후보 마스크로 검출)
  py src/run_road_detection.py --input <사진> --detection-config configs/detection_p2bd.json   # opencv-contrib 필요

출력 (기본 outputs/road_detection/run_<시각>/)
  images/<image_id>/  processed_image.png · road_mask.png · analysis_mask.png · result.jpg(검출 그림) ·
                      detections.json · metadata.json(전처리 + 검출 기록)   (마스크 FAIL이면 road_mask_candidate.png · overlay.jpg)
  summary.csv         이미지당 한 줄 (상태 · 마스크 판정 · 도로 위 균열/포트홀 수 · 시간)
  detections.csv      후보당 한 줄 (종류 · 도로 위 여부 · 원본 좌표 박스 · 형태값)
  run_config.json · run_summary.json · review_sheet.jpg
rdd_test(최종 확인용)는 막혀 있다 (--allow-rdd-test 제외).
"""
import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from paths import REPO_ROOT
from preprocessing.config import ConfigError, config_hash, load_config
from preprocessing.image_io import read_image, write_image
from preprocessing.mask_editor import edit_mask_interactive, find_external_mask, load_external_mask
from preprocessing.pipeline import save_result
from road_detection.config import DetectionConfigError, check_runtime, load_detection_config
from road_detection.pipeline import KINDS, run_image
from road_detection.render import draw_result, review_sheet
from preprocessing.run_preprocess import collect_images, new_run_dir

DEFAULT_PRE = REPO_ROOT / "configs" / "preprocessing.json"
DEFAULT_DET = REPO_ROOT / "configs" / "detection_default.json"
SUMMARY_COLUMNS = ["image_id", "status", "auto_validation", "mask_source", "unverified_mask", "n_crack", "n_pothole", "n_off_road",
                   "gamma_applied", "gaussian_applied", "preprocess_ms", "detection_ms", "total_ms", "warnings", "error", "input_path"]
DET_COLUMNS = ["image_id", "id", "type", "on_road", "road_overlap", "x", "y", "w", "h", "area", "length", "width", "elong", "contrast"]


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


def main(argv=None):
    p = argparse.ArgumentParser(description="도로 손상 검출 시스템 (전처리 → 검출)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="이미지 파일 또는 폴더")
    src.add_argument("--dataset", help="rdd_dev / rdd_tune / rdd_val / provided / captured")
    p.add_argument("--images", nargs="+", help="입력 중 이 파일 이름만")
    p.add_argument("--limit", type=int)
    p.add_argument("--preprocess-config", default=str(DEFAULT_PRE))
    p.add_argument("--detection-config", default=str(DEFAULT_DET))
    p.add_argument("--set", nargs="+", default=[], metavar="키.경로=값", help="전처리 설정 덮어쓰기")
    p.add_argument("--on-fail", choices=["skip", "use_candidate"], help="자동 마스크 FAIL일 때 (설정 파일 값 덮어쓰기)")
    p.add_argument("--manual-mask", help="한 장 입력일 때 수동 road_mask PNG")
    p.add_argument("--manual-mask-dir", help="수동 마스크 폴더 (<이름>.png 등)")
    p.add_argument("--edit-failed", action="store_true", help="마스크 FAIL이면 다각형 편집기를 연다 (GUI 필요)")
    p.add_argument("--output")
    p.add_argument("--allow-rdd-test", action="store_true")
    args = p.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error:--limit은 1 이상")
    try:
        pre_cfg = load_config(args.preprocess_config, args.set)
        det = load_detection_config(args.detection_config)
        if args.on_fail:
            det["integration"]["on_fail"] = args.on_fail
        check_runtime(det)
    except (ConfigError, DetectionConfigError) as exc:
        raise SystemExit(f"error:{exc}") from None
    images = collect_images(args)
    if not images:
        raise SystemExit("error:처리할 이미지가 없음")
    if args.manual_mask and len(images) != 1:
        raise SystemExit("error:--manual-mask는 한 장 입력에서만 (여러 장은 --manual-mask-dir)")
    run_dir = new_run_dir(REPO_ROOT / "outputs" / "road_detection", args.output)
    (run_dir / "run_config.json").write_text(json.dumps({
        "program": "src/run_road_detection.py", "argv": sys.argv[1:] if argv is None else list(argv),
        "preprocess_config": str(Path(args.preprocess_config).resolve()), "preprocess_config_sha256": config_hash(pre_cfg),
        "preprocess": pre_cfg, "detection_config": str(Path(args.detection_config).resolve()), "detection": det,
        "n_images": len(images), "started_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__}},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"입력 {len(images)}장 · 검출 설정 {det['name']} ({det['detector_name']}) · FAIL 처리 {det['integration']['on_fail']} → {run_dir}", flush=True)
    rows, det_rows, sheet = [], [], []
    t_start = time.perf_counter()
    for index, (image_id, path) in enumerate(images, 1):
        try:
            image = read_image(path)
        except ValueError as exc:
            rows.append({"image_id": image_id, "status": "error_read", "error": str(exc), "input_path": str(path)})
            print(f"[{index}/{len(images)}] {image_id}: 읽기 실패 — {exc}", flush=True)
            continue
        manual_mask = manual_path = None
        try:
            manual_path = Path(args.manual_mask) if args.manual_mask else find_external_mask(args.manual_mask_dir, path.name)
            if manual_path is not None:
                manual_mask = load_external_mask(manual_path, image.shape, pre_cfg["manual_correction"])
        except ValueError as exc:
            print(f"  경고: 수동 마스크를 쓸 수 없음 ({exc})", flush=True)
            manual_mask = manual_path = None
        res = run_image(image, pre_cfg, det, image_id=image_id, input_path=path, manual_mask=manual_mask,
                        manual_source=manual_path, editor=edit_mask_interactive if args.edit_failed else None)
        pre, meta = res["preprocess"], res["preprocess"].metadata
        out_dir = run_dir / "images" / image_id
        save_result(pre, out_dir, pre_cfg["output"], image)
        counts = meta["detection"]["counts_on_road"]
        title = (f"{image_id}  mask {meta.get('validation', {}).get('status', '-')}{' (unverified)' if res['unverified_mask'] else ''}\n"
                 f"crack {counts['crack']}  pothole {counts['pothole']}  off-road {meta['detection']['n_off_road']}  [{res['status']}]")
        mask = pre.road_mask if pre.road_mask is not None else pre.auto_mask
        vis = draw_result(image, mask, res["candidates"], title)
        write_image(out_dir / "result.jpg", vis)
        (out_dir / "detections.json").write_text(json.dumps({"image_id": image_id, "status": res["status"], "geometry": res["geometry"],
                                                             "candidates": res["candidates"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        if len(sheet) < 36:
            sheet.append(vis)
        rows.append({"image_id": image_id, "status": res["status"], "auto_validation": meta.get("validation", {}).get("status"),
                     "mask_source": (meta.get("final_mask") or {}).get("source"), "unverified_mask": res["unverified_mask"],
                     "n_crack": counts["crack"], "n_pothole": counts["pothole"], "n_off_road": meta["detection"]["n_off_road"],
                     "gamma_applied": (meta.get("gamma") or {}).get("applied"), "gaussian_applied": (meta.get("gaussian") or {}).get("applied"),
                     "preprocess_ms": round(res["timing_ms"].get("preprocess") or 0, 1), "detection_ms": round(res["timing_ms"].get("detection") or 0, 1),
                     "total_ms": round(res["timing_ms"]["total"], 1), "warnings": ";".join(meta.get("warnings") or []),
                     "error": res["error"] or ";".join(e["message"] for e in meta.get("errors") or []), "input_path": str(path)})
        for c in res["candidates"]:
            x, y, w, h = c["bbox_original"]
            det_rows.append({"image_id": image_id, **c, "x": x, "y": y, "w": w, "h": h})
        print(f"[{index}/{len(images)}] {image_id}: {res['status']} · 마스크 {rows[-1]['auto_validation']}"
              f"{'(검증 안 됨)' if res['unverified_mask'] else ''} · 균열 {counts['crack']} · 포트홀 {counts['pothole']}"
              f" · 도로 밖 {meta['detection']['n_off_road']} · {res['timing_ms']['total']:.0f}ms", flush=True)
    write_csv(run_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    write_csv(run_dir / "detections.csv", det_rows, DET_COLUMNS)
    if sheet:
        cv2.imencode(".jpg", review_sheet(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(run_dir / "review_sheet.jpg"))
    done = [r for r in rows if r["status"] == "detected"]
    summary = {"n_images": len(rows), "detected": len(done),
               "skipped_mask_fail": sum(str(r["status"]).startswith("skipped_manual") for r in rows),
               "errors": sum(str(r["status"]).startswith("error") for r in rows),
               "unverified_mask_used": sum(bool(r.get("unverified_mask")) for r in rows),
               "candidates_on_road": {k: sum(r.get(f"n_{k}") or 0 for r in done) for k in KINDS},
               "candidates_off_road": sum(r.get("n_off_road") or 0 for r in done),
               "mean_total_ms": round(float(np.mean([r["total_ms"] for r in rows if r.get("total_ms") is not None])), 1) if rows else None,
               "wall_time_s": round(time.perf_counter() - t_start, 1)}
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: 검출 {summary['detected']}장 · 마스크 FAIL로 건너뜀 {summary['skipped_mask_fail']}장 · 오류 {summary['errors']}장"
          f" · 도로 위 균열 {summary['candidates_on_road']['crack']} · 포트홀 {summary['candidates_on_road']['pothole']}"
          f" · 도로 밖 후보 {summary['candidates_off_road']} → {run_dir}")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
