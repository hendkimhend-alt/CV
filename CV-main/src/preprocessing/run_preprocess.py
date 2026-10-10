"""전처리만 실행한다 (검출 없음).

  py src/preprocessing/run_preprocess.py --dataset rdd_dev --limit 20
  py src/preprocessing/run_preprocess.py --input <사진 또는 폴더> [--manual-mask-dir <폴더>] [--edit-failed]

결과는 outputs/preprocessing/run_<시각>/ 에 저장된다. rdd_test 이미지는 --allow-rdd-test 없이는 건너뛴다.
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

if __name__ == "__main__":
    # 직접 실행하면 이 폴더가 import 경로가 되므로 src/로 바꾼다
    sys.path[0] = str(Path(__file__).resolve().parents[1])

from data import list_images  # noqa: E402
from paths import LABELS_DIR, REPO_ROOT  # noqa: E402
from preprocessing.config import DEFAULT_CONFIG_PATH, ConfigError, config_hash, load_config  # noqa: E402
from preprocessing.image_io import IMAGE_EXTENSIONS, read_image  # noqa: E402
from preprocessing.mask_editor import edit_mask_interactive, find_external_mask, load_external_mask  # noqa: E402
from preprocessing.pipeline import ERROR, MANUAL_REQUIRED, SUCCESS, PreprocessResult  # noqa: E402
from preprocessing.pipeline import make_overlay, overlay_text, process_image, save_result  # noqa: E402
from preprocessing.quality import METRIC_NAMES  # noqa: E402

KST = timezone(timedelta(hours=9))
ALLOWED_DATASETS = ("rdd_dev", "rdd_tune", "rdd_val", "provided", "captured")
BLOCKED_DATASETS = ("rdd", "rdd_test")
SUMMARY_COLUMNS = (
    ["image_id", "input_path", "status", "auto_validation", "fail_reasons", "final_mask_source",
     "manual_required", "manual_applied", "mask_area_ratio", "largest_component_ratio", "hole_ratio",
     "seed_consistency", "n_seeds", "analysis_pixels", "analysis_sufficient",
     "gamma_applied", "gamma_value", "gamma_reason",
     "gaussian_applied", "gaussian_sigma", "gaussian_kernel", "gaussian_reason"]
    + [f"initial_{m}" for m in METRIC_NAMES]
    + [f"final_{m}" for m in METRIC_NAMES]
    + ["time_ms", "warnings", "errors"]
)


def test_split_names():
    path = LABELS_DIR / "split_rdd.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {row["image"] for row in csv.DictReader(f) if row["split"] == "test"}


def _is_image(path, root):
    hidden = any(part.startswith(".") for part in path.relative_to(root).parts)
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not hidden


def collect_images(args):
    """→ [(image_id, path)]"""
    if args.dataset:
        if args.dataset in BLOCKED_DATASETS and not args.allow_rdd_test:
            raise SystemExit(f"error:{args.dataset}에는 rdd_test가 포함되어 있어 막혀 있음. rdd_dev를 사용하세요")
        if args.dataset not in ALLOWED_DATASETS + BLOCKED_DATASETS:
            raise SystemExit(f"error:--dataset은 {ALLOWED_DATASETS} 중 하나 (폴더는 --input 사용)")
        _, folder, paths = list_images(args.dataset)
        items = [(p.relative_to(folder), p) for p in paths]
    else:
        source = Path(args.input)
        if source.is_file():
            items = [(Path(source.name), source)]
        elif source.is_dir():
            items = sorted((p.relative_to(source), p) for p in source.rglob("*") if _is_image(p, source))
        else:
            raise SystemExit(f"error:입력 경로 없음 {source}")

    if args.images:
        wanted = set(args.images)
        items = [item for item in items if item[1].name in wanted]
        missing = wanted - {p.name for _, p in items}
        if missing:
            print(f"경고: 입력에서 찾지 못한 이미지 {sorted(missing)}")

    if not args.allow_rdd_test:
        blocked = test_split_names()
        n_blocked = sum(p.name in blocked for _, p in items)
        if n_blocked:
            print(f"rdd_test 보호: test 분할 이미지 {n_blocked}장을 건너뜀")
        items = [item for item in items if item[1].name not in blocked]

    if args.limit:
        items = items[:args.limit]
    return [("__".join(rel.with_suffix("").parts), path) for rel, path in items]


def new_run_dir(root, explicit):
    if explicit:
        run_dir = Path(explicit)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"error:출력 폴더가 비어 있지 않음 {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    base = datetime.now(KST).strftime("run_%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        run_dir = root / (base if n == 1 else f"{base}_{n}")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise SystemExit("error:실행 폴더를 만들 수 없음")


def summary_row(result, path):
    meta = result.metadata
    validation = meta.get("validation") or {}
    metrics = validation.get("metrics") or {}
    quality = meta.get("quality") or {}
    gamma = meta.get("gamma") or {}
    gauss = meta.get("gaussian") or {}
    analysis = meta.get("analysis_mask") or {}
    manual = meta.get("manual_correction") or {}
    row = {
        "image_id": result.image_id,
        "input_path": str(path),
        "status": result.status,
        "auto_validation": validation.get("status"),
        "fail_reasons": ";".join(validation.get("fail_reasons") or []),
        "final_mask_source": (meta.get("final_mask") or {}).get("source"),
        "manual_required": manual.get("required"),
        "manual_applied": manual.get("applied"),
        "n_seeds": metrics.get("n_seeds"),
        "analysis_pixels": analysis.get("pixels"),
        "analysis_sufficient": analysis.get("sufficient"),
        "gamma_applied": gamma.get("applied"),
        "gamma_value": gamma.get("value_used"),
        "gamma_reason": gamma.get("reason"),
        "gaussian_applied": gauss.get("applied"),
        "gaussian_sigma": gauss.get("sigma_used"),
        "gaussian_kernel": gauss.get("kernel_used"),
        "gaussian_reason": gauss.get("reason"),
        "time_ms": meta["timing_ms"].get("total"),
        "warnings": ";".join(meta.get("warnings") or []),
        "errors": ";".join(f"{e['stage']}:{e['message']}" for e in meta.get("errors") or []),
    }
    for name in ("mask_area_ratio", "largest_component_ratio", "hole_ratio", "seed_consistency"):
        row[name] = metrics.get(name)
    for name in METRIC_NAMES:
        row[f"initial_{name}"] = (quality.get("initial") or {}).get(name)
        row[f"final_{name}"] = (quality.get("final") or {}).get(name)
    return row


def _time_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": round(float(arr.mean()), 1), "median": round(float(np.median(arr)), 1),
            "min": round(float(arr.min()), 1), "max": round(float(arr.max()), 1)}


def run_summary(rows):
    ok = [r for r in rows if r["status"] == SUCCESS]
    fail_counts = {}
    for r in rows:
        for reason in filter(None, (r["fail_reasons"] or "").split(";")):
            key = reason.split(" ")[0]
            fail_counts[key] = fail_counts.get(key, 0) + 1
    return {
        "n_images": len(rows),
        "success": len(ok),
        "auto_mask_pass": sum(r["auto_validation"] == "PASS" for r in rows),
        "auto_mask_fail": sum(r["auto_validation"] == "FAIL" for r in rows),
        "manual_correction_required": sum(r["status"] == MANUAL_REQUIRED for r in rows),
        "manual_correction_applied": sum(bool(r["manual_applied"]) for r in rows),
        "errors": sum(r["status"] == ERROR for r in rows),
        "gamma_applied": sum(bool(r["gamma_applied"]) for r in ok),
        "gaussian_applied": sum(bool(r["gaussian_applied"]) for r in ok),
        "mean_time_ms_all": _time_stats([r["time_ms"] for r in rows]),
        "mean_time_ms_success": _time_stats([r["time_ms"] for r in ok]),
        "fail_reason_counts": fail_counts,
    }


def write_review_sheet(path, overlays, columns=4, tile=360):
    if not overlays:
        return
    tiles = []
    for vis in overlays:
        scale = tile / max(vis.shape[:2])
        size = (max(1, round(vis.shape[1] * scale)), max(1, round(vis.shape[0] * scale)))
        canvas = np.zeros((tile, tile, 3), np.uint8)
        small = cv2.resize(vis, size, interpolation=cv2.INTER_AREA)
        canvas[:small.shape[0], :small.shape[1]] = small
        tiles.append(canvas)
    while len(tiles) % columns:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(str(path))


def _load_manual_mask(args, path, image, cfg):
    """→ (마스크, 경로). 없거나 못 쓰면 (None, None)."""
    if args.manual_mask:
        mask_path = Path(args.manual_mask)
    else:
        mask_path = find_external_mask(args.manual_mask_dir, path.name)
    if mask_path is None:
        return None, None
    try:
        return load_external_mask(mask_path, image.shape, cfg["manual_correction"]), mask_path
    except ValueError as exc:
        print(f"  경고: 수동 마스크를 쓸 수 없음 ({exc})", flush=True)
        return None, None


def _log_line(index, total, result):
    meta = result.metadata
    detail = meta.get("validation", {}).get("status", "-")
    if result.status == SUCCESS:
        detail += (f" · mask={meta['final_mask']['source']} · gamma={meta['gamma']['value_used']}"
                   f" · gauss={meta['gaussian']['sigma_used']}")
    elif result.status == ERROR:
        detail += " · " + "; ".join(e["message"] for e in meta["errors"])
    else:
        detail += " · " + "; ".join(meta["validation"]["fail_reasons"])
    return f"[{index}/{total}] {result.image_id}: {result.status} ({detail}) {meta['timing_ms']['total']:.0f}ms"


def parse_args(argv):
    p = argparse.ArgumentParser(description="도로 영상 전처리")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="이미지 파일 또는 폴더")
    source.add_argument("--dataset", help=f"{ALLOWED_DATASETS}")
    p.add_argument("--images", nargs="+", help="이 파일 이름만 처리")
    p.add_argument("--limit", type=int, help="처음 N장만")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--set", nargs="+", default=[], metavar="키.경로=값", help="설정 덮어쓰기")
    p.add_argument("--output", help="결과 폴더 (비어 있어야 함)")
    p.add_argument("--manual-mask", help="한 장 입력일 때 수동 road_mask PNG")
    p.add_argument("--manual-mask-dir", help="수동 마스크 폴더")
    p.add_argument("--edit-failed", action="store_true", help="FAIL이면 편집기를 연다")
    p.add_argument("--review", action="store_true", help="PASS도 편집기로 확인")
    p.add_argument("--allow-rdd-test", action="store_true", help="최종 확인 때만 사용")
    args = p.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("error:--limit은 1 이상")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, args.set)
    except ConfigError as exc:
        raise SystemExit(f"error:{exc}") from None
    images = collect_images(args)
    if not images:
        raise SystemExit("error:처리할 이미지가 없음")
    if args.manual_mask and len(images) != 1:
        raise SystemExit("error:--manual-mask는 한 장 입력에서만 사용 (여러 장은 --manual-mask-dir)")

    review = args.review or cfg["manual_correction"]["review_pass_results"]
    editor = edit_mask_interactive if (args.edit_failed or review) else None
    out_root = Path(cfg["output"]["root"])
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    run_dir = new_run_dir(out_root, args.output)

    run_info = {"program": "src/preprocessing/run_preprocess.py",
                "argv": sys.argv[1:] if argv is None else list(argv),
                "config_path": str(Path(args.config).resolve()), "config_sha256": config_hash(cfg),
                "config": cfg, "n_images": len(images),
                "started_at": datetime.now(KST).isoformat(timespec="seconds"),
                "versions": {"python": platform.python_version(), "opencv": cv2.__version__,
                             "numpy": np.__version__}}
    (run_dir / "run_config.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    verbose = cfg["output"]["log_level"] != "quiet"
    print(f"입력 {len(images)}장 → {run_dir}", flush=True)
    rows, overlays = [], []
    t_start = time.perf_counter()
    for index, (image_id, path) in enumerate(images, 1):
        try:
            image = read_image(path)
        except ValueError as exc:
            error = {"stage": "read", "type": type(exc).__name__, "message": str(exc)}
            meta = {"image_id": image_id, "status": ERROR, "timing_ms": {"total": None},
                    "warnings": [], "errors": [error]}
            rows.append(summary_row(PreprocessResult(image_id, ERROR, meta), path))
            print(f"[{index}/{len(images)}] {image_id}: error 읽기 실패 — {exc}", flush=True)
            continue

        manual_mask, manual_path = _load_manual_mask(args, path, image, cfg)
        result = process_image(image, cfg, image_id=image_id, input_path=path, manual_mask=manual_mask,
                               manual_source=manual_path, editor=editor, review=review)
        save_result(result, run_dir / "images" / image_id, cfg["output"], image)
        rows.append(summary_row(result, path))

        if cfg["output"]["review_sheet"] and len(overlays) < 48:
            mask = result.road_mask if result.road_mask is not None else result.auto_mask
            overlays.append(make_overlay(image, mask, result.analysis_mask, result.seeds,
                                         overlay_text(result), max_side=720))
        if verbose:
            print(_log_line(index, len(images), result), flush=True)

    with (run_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = run_summary(rows)
    summary["wall_time_s"] = round(time.perf_counter() - t_start, 2)
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if cfg["output"]["review_sheet"]:
        write_review_sheet(run_dir / "review_sheet.jpg", overlays)

    print(f"완료: 성공 {summary['success']} · PASS {summary['auto_mask_pass']} / FAIL {summary['auto_mask_fail']}"
          f" · 수동 수정 필요 {summary['manual_correction_required']}"
          f" · 수동 마스크 적용 {summary['manual_correction_applied']} · 오류 {summary['errors']}"
          f" · Gamma 적용 {summary['gamma_applied']}")
    if summary["mean_time_ms_all"]:
        print(f"평균 {summary['mean_time_ms_all']['mean']:.0f} ms/장 → {run_dir}")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
