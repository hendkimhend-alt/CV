"""
B 구성(균열 최종, 10/10 확정)을 팀 공통 방법(EVALUATION.md)으로 잰다 — B 위에 보정을 하나씩 얹어 시험하는 것도 여기서.

B = 팀원 전처리 v1.1 (자동 노면 마스크 · 조건부 감마) → 연결 (노면 외접 영역 자르기 · 긴 변 1024)
    → 검출기 p2bd → 도로 위 판정 (후보 박스의 50% 이상이 노면 마스크 안이면 남김)
실행 코드 · 설정은 CV-main/ 그대로 쓴다 (CV-main/configs/preprocessing.json · detection_p2bd.json).
채점은 맨 위 src/의 in50 (원본 좌표, ROI로 잘린 정답 없음) → outputs/pipeline/<run>/results.csv
(compare_runs.py가 읽는 형식) + results/runs.csv 자동 기록 (run_pipeline.py와 같은 칸).

B 위에 얹는 보정 (--extra, 기본 없음 = B 그대로):
  연결 뒤 검출기 입력(긴 변 1024)에서 노면 마스크 안에만 건다 — 크기(px)가 Hessian 크기(σ 1 · 1.5 · 2)와 같은 척도
  {"gaussian": {"ksize": 3, "sigma": 0.8}}                  가우시안 (1차 값)
  {"line_close": {"length": 11, "angles": 12}}              막대 닫힘: 여러 방향 막대로 닫힌 결과의 최솟값
                                                             → 막대보다 짧은 어두운 점 · 줄(골재)은 메우고, 긴 어두운 선(균열)은 남김
  {"unsharp": {"amount": 1.5, "sigma": 1.0, "apply_to": "blur"}}   언샤프 (apply_to: all · blur — 흐림 = labels/split_rdd.csv 그룹, 사진만으로 정한 분류)
  {"clahe": {"clip": 2.0, "tile": 8, "apply_to": "all"}}     CLAHE (Lab L 채널, apply_to: all · local_illumination 등 그룹)
  {"gamma": {"mode": "auto", "min": 0.5, "max": 2.0}}        사진마다 감마: 노면 회색 가운데값 m이 128이 되는 γ = ln 0.5 ÷ ln(m/255)
                                                             (auto_dark = γ ≤ 1만, 어두운 사진만 밝게 · fixed = "value")
  {"harris_adapt": {"win": 31, "thr": 2.0, "strong": {…질감 처리…}}}
                                                             거친 곳만 더 세게: Harris 코너 세기(수업 3-2)를 창(win)으로 평균 내 사진 노면의 중앙값으로 나눈 값이
                                                             thr 이상인 노면은 "strong" 질감 처리 결과로 바꿈 (판단은 보정 전 입력으로 · thr는 정답 없이
                                                             노면 분포의 분위수로 정함 — analysis/harris_density.py)
  여러 개를 함께 쓰면 "order"의 순서로 (기본 감마 → CLAHE → 가우시안 → 막대 닫힘 → 언샤프),
  예: {"line_close": …, "gaussian": …, "order": ["line_close", "gaussian"]} · 모든 단계에 "apply_to" 가능 (기본 all)
  보정은 Hessian 점수 크기를 바꾸므로 --calibrate로 강한 기준을 다시 맞춘다:
  B 그대로일 때 강한 흔적(중심선 ≥ 128.609375) 총수와 같아지는 값 (사진만 씀, 정답 안 봄 — analysis/calibrate.py와 같은 원리)

실행 (저장소 맨 위 폴더에서, opencv-contrib 필요):
  python analysis/run_road_pipeline.py --note "B 재현"                                    # 563장 약 15분
  python analysis/run_road_pipeline.py --limit 20 --no-log                                  # 빠른 확인
  python analysis/run_road_pipeline.py --cache 폴더 --note "…"                             # 전처리 · 연결 결과를 폴더에 저장 / 있으면 다시 씀
  python analysis/run_road_pipeline.py --cache 폴더 --extra '{"line_close": {"length": 11, "angles": 12}}' \
      --calibrate --config-name b_lc11 --note "B + 막대 닫힘 11"
  python analysis/run_road_pipeline.py --cache 폴더 --det-set guided_filter='{"r":4,"eps":"var","eps_scale":2.0}' --calibrate --config-name b_gf3
  python analysis/compare_runs.py B=<B 실행> 새것=<이 실행> --split
rdd_test는 --allow-rdd-test를 줄 때만 (EVALUATION.md 7절 — 쓰면 experiments.md에 기록).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "CV-main"
# CV-main/src와 맨 위 src/에 같은 이름의 모듈(paths · data · evaluate · preprocess · roi · detect)이 있다.
# 먼저 CV-main 쪽을 불러 실행 함수를 잡아 두고, 겹치는 이름을 지운 뒤 맨 위 src/를 불러 채점에 쓴다.
SHARED = ("paths", "data", "evaluate", "preprocess", "roi", "detect", "roadcue")
sys.path.insert(0, str(MAIN / "src"))
from detect import (DEFAULT_CFG, _line_kernel, _to_gray, detect, find_input, line_centerline,  # noqa: E402
                    line_centerline_interp, line_score, tensor_propagate)
from preprocessing.config import config_hash, load_config  # noqa: E402
from preprocessing.detector_adapter import to_detector_input, to_original_bbox  # noqa: E402
from preprocessing.pipeline import MANUAL_REQUIRED, SUCCESS, process_image  # noqa: E402
from road_detection.config import check_runtime, load_detection_config  # noqa: E402
from road_detection.pipeline import road_overlap  # noqa: E402
sys.path.remove(str(MAIN / "src"))
for name in SHARED:
    sys.modules.pop(name, None)
sys.path.insert(0, str(ROOT / "src"))
from data import list_images, load_gt, load_viewpoints  # noqa: E402
from evaluate import CRITERIA, KINDS, evaluate  # noqa: E402
from paths import OUTPUT_DIR, imread  # noqa: E402
from run_pipeline import git_version, log_run, new_run_dir, parse_overrides  # noqa: E402

PRE_CFG = MAIN / "configs" / "preprocessing.json"
DET_CFG = MAIN / "configs" / "detection_p2bd.json"
SPLIT_CSV = ROOT / "labels" / "split_rdd.csv"
CONDITION, DETECTOR = "v1.1_road", "D1hv"      # runs.csv의 조건 · 검출기 칸
KEEP = 200_000                                  # 강한 기준 맞추기: 사진마다 위쪽 중심선 점수만 보관


def prepare(path, pre_cfg, det, cache):
    """팀원 전처리 + 연결 → (기록, 검출기 입력 영상, 검출기 마스크, geometry). 전처리 실패면 영상 None.
    road_detection.pipeline.run_image · detect_on_road와 같은 순서 (검출 직전까지)."""
    if cache and (cache / f"{path.stem}.json").exists():
        meta = json.loads((cache / f"{path.stem}.json").read_text())
        if meta["status"] != "detected":
            return meta, None, None, None
        return (meta, cv2.imread(str(cache / f"{path.stem}_img.png"), cv2.IMREAD_COLOR),
                cv2.imread(str(cache / f"{path.stem}_mask.png"), cv2.IMREAD_GRAYSCALE), meta["geometry"])
    image = imread(path)
    pre = process_image(image, pre_cfg, image_id=path.name)
    unverified = False
    if pre.status == MANUAL_REQUIRED and det["integration"]["on_fail"] == "use_candidate" and pre.auto_mask is not None \
            and np.count_nonzero(pre.auto_mask):
        # 검증되지 않은 자동 후보 마스크로 전처리를 끝까지 진행 (run_image와 같음)
        pre = process_image(image, pre_cfg, image_id=path.name, manual_mask=pre.auto_mask,
                            manual_source="auto_candidate_unverified")
        unverified = True
    meta = {"status": pre.status, "unverified_mask": unverified,
            "gamma": bool(pre.metadata.get("gamma", {}).get("applied"))}
    img = mask = geo = None
    if pre.status != SUCCESS:
        meta["status"] = "skipped_" + pre.status
    else:
        integ = det["integration"]
        img, mask, geo = to_detector_input(pre.processed_image, pre.road_mask, integ["long_side"],
                                           crop_to_road=integ["input_mode"] == "crop_to_road",
                                           crop_margin=integ["crop_margin"])
        meta.update(status="detected", geometry=geo)
    if cache:
        if img is not None:
            cv2.imwrite(str(cache / f"{path.stem}_img.png"), img)
            cv2.imwrite(str(cache / f"{path.stem}_mask.png"), mask)
        (cache / f"{path.stem}.json").write_text(json.dumps(meta, default=float))
    return meta, img, mask, geo


def apply_extra(img, mask, extra, tags):
    """B 위에 얹는 보정 — 검출기 입력에서 노면 마스크 안에만, extra["order"] 순서로. → (영상, 적용했나)"""
    out, applied = img, False
    for step in extra.get("order", ("gamma", "clahe", "gaussian", "line_close", "unsharp")):
        if step not in extra:
            continue
        c = extra[step]
        if c.get("apply_to", "all") != "all" and c["apply_to"] not in tags:
            continue
        if step == "gamma":
            if c["mode"] == "fixed":
                g = float(c["value"])
            else:
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                m = float(np.clip(np.median(gray[mask > 0]) if np.any(mask > 0) else np.median(gray), 1, 254))
                g = float(np.clip(np.log(0.5) / np.log(m / 255.0), c.get("min", 0.5), c.get("max", 2.0)))
                g = min(g, 1.0) if c["mode"] == "auto_dark" else g
            out = cv2.LUT(out, np.rint(255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8))
        elif step == "clahe":
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=c["clip"], tileGridSize=(c.get("tile", 8),) * 2).apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif step == "gaussian":
            out = cv2.GaussianBlur(out, (c["ksize"], c["ksize"]), c["sigma"], borderType=cv2.BORDER_REFLECT_101)
        elif step == "line_close":
            closed = [cv2.morphologyEx(out, cv2.MORPH_CLOSE, _line_kernel(c["length"], 180.0 * i / c["angles"]),
                                       borderType=cv2.BORDER_REPLICATE) for i in range(c["angles"])]
            out = np.min(np.stack(closed), axis=0)
        elif step == "harris_adapt":
            resp = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), 2, 3, 0.04)
            energy = cv2.boxFilter(np.maximum(resp, 0), -1, (c["win"], c["win"]))
            energy /= max(float(np.median(energy[mask > 0])) if np.any(mask > 0) else 1.0, 1e-9)
            rough = (energy >= c["thr"]) & (mask > 0)
            strong, _ = apply_extra(img, mask, c["strong"], tags)
            out = np.where(rough[..., None], strong, out)
        elif step == "unsharp":
            src = out.astype(np.float32)
            blur = cv2.GaussianBlur(src, (0, 0), c["sigma"], borderType=cv2.BORDER_REFLECT_101)
            out = np.rint(np.clip(src + c["amount"] * (src - blur), 0, 255)).astype(np.uint8)
        else:
            raise SystemExit(f"error:모르는 보정 {step}")
        applied = True
    if not applied:
        return img, False
    return np.where((mask > 0)[..., None], out, img).astype(np.uint8), True


def center_map(img, cfg):
    """강한 기준과 비교하는 바로 그 중심선 점수 (detect.line_trace · analysis/calibrate.py와 같음)."""
    score, angle = line_score(find_input(_to_gray(img), cfg), cfg["line_sigmas"], cfg.get("line_scale_combine", "max"))
    if cfg.get("tensor_propagate", 0) > 0:
        field, nangle = tensor_propagate(score, angle, cfg)
        score, angle = (score * field, angle) if cfg.get("tensor_gate", False) else (field, nangle)
    nms = line_centerline_interp if cfg.get("nms_interp", False) else line_centerline
    return nms(score, angle)


def main():
    p = argparse.ArgumentParser(description="B 구성 (팀원 전처리 v1.1 + 연결 + 도로 위 판정 + p2bd)을 in50으로 채점")
    p.add_argument("--dataset", default="rdd_dev", choices=("rdd_dev", "rdd_test"))
    p.add_argument("--limit", type=int)
    p.add_argument("--allow-rdd-test", action="store_true", help="rdd_test는 최종 확인에만 (EVALUATION.md 7절)")
    p.add_argument("--pre-config", type=Path, default=PRE_CFG, help="전처리 설정 (기본 CV-main/configs/preprocessing.json)")
    p.add_argument("--det-config", type=Path, default=DET_CFG, help="검출 설정 (기본 CV-main/configs/detection_p2bd.json)")
    p.add_argument("--det-set", nargs="+", metavar="키=값", help="검출 설정 위에 덮어쓸 detect 설정 (예: guided_filter=…)")
    p.add_argument("--extra", default="{}", help="B 위에 얹는 보정 (JSON, 위 설명)")
    p.add_argument("--calibrate", action="store_true", help="강한 기준을 B의 강한 흔적 총수에 맞춤 (--cache 필요)")
    p.add_argument("--cache", type=Path, help="전처리 · 연결 결과 저장 폴더 (있으면 다시 씀 — 같은 전처리 설정일 때만)")
    p.add_argument("--config-name", default="b_road", help="runs.csv의 설정 칸 (설정을 바꾸면 이름도 바꿀 것)")
    p.add_argument("--note", default="")
    p.add_argument("--no-log", action="store_true", help="results/runs.csv에 남기지 않음")
    a = p.parse_args()
    if a.dataset == "rdd_test" and not a.allow_rdd_test:
        raise SystemExit("error:rdd_test는 --allow-rdd-test로만 (최종 확인용 — 쓰면 experiments.md 테스트 사용 기록에 적기)")
    if a.calibrate and not a.cache:
        raise SystemExit("error:--calibrate는 --cache가 필요 (B의 강한 흔적 수를 캐시에 둠)")

    pre_path, det_path = a.pre_config.resolve(), a.det_config.resolve()
    pre_cfg, det = load_config(pre_path), load_detection_config(det_path)
    check_runtime(det)
    base_cfg = {**DEFAULT_CFG, **det["detect_cfg"]}
    cfg = {**base_cfg, **parse_overrides(a.det_set)}
    extra = json.loads(a.extra)
    name, _, images = list_images(a.dataset)
    images = images[:a.limit] if a.limit else images
    gt_of, vps = load_gt(a.dataset, name), load_viewpoints(a.dataset)
    group = {r["image"]: r["group"] for r in csv.DictReader(open(SPLIT_CSV, encoding="utf-8"))}
    tags_of = lambda n: set(group.get(n, "").split(";"))
    if a.cache:
        a.cache.mkdir(parents=True, exist_ok=True)
        stamp = {"preprocessing_hash": config_hash(pre_cfg), "integration": det["integration"]}
        sp = a.cache / "_cache.json"
        if sp.exists() and json.loads(sp.read_text()) != stamp:
            raise SystemExit(f"error:{a.cache}는 다른 전처리 · 연결 설정의 캐시")
        sp.write_text(json.dumps(stamp))

    hi_used = cfg["line_hi_abs"]
    if a.calibrate:
        key = hashlib.sha256(json.dumps([a.dataset, a.limit, base_cfg["line_hi_abs"],
                                         {k: base_cfg[k] for k in sorted(base_cfg) if k.startswith(("line", "tensor"))}],
                                        sort_keys=True, default=str).encode()).hexdigest()[:12]
        ref_path = a.cache / f"_ref_count_{key}.json"
        if ref_path.exists():
            n_ref = json.loads(ref_path.read_text())["n_ref"]
        else:
            n_ref = 0
            for path in images:
                _, img, _, _ = prepare(path, pre_cfg, det, a.cache)
                if img is not None:
                    n_ref += int((center_map(img, base_cfg) >= base_cfg["line_hi_abs"]).sum())
            ref_path.write_text(json.dumps({"n_ref": n_ref, "hi": base_cfg["line_hi_abs"]}))
        vals = []
        for i, path in enumerate(images, 1):
            meta, img, mask, _ = prepare(path, pre_cfg, det, a.cache)
            if img is None:
                continue
            img, _ = apply_extra(img, mask, extra, tags_of(path.name))
            c = center_map(img, cfg)
            vals.append(np.sort(c[c > 0])[-KEEP:])
            if i % 100 == 0:
                print(f"  맞추기 [{i}/{len(images)}]", flush=True)
        allv = np.sort(np.concatenate(vals))[::-1]
        hi_used = float(allv[n_ref - 1])
        cfg["line_hi_abs"] = hi_used
        print(f"강한 기준 맞춤: B 강한 흔적 {n_ref:,}개 (≥ {base_cfg['line_hi_abs']}) → 이 설정의 강한 기준 {hi_used:.6g}", flush=True)

    run_dir = new_run_dir(OUTPUT_DIR / "pipeline")
    print(f"{a.dataset}: {len(images)}장 · B 구성 ({pre_path.relative_to(ROOT)} + {det_path.relative_to(ROOT)})"
          f"{' + ' + json.dumps(extra, ensure_ascii=False) if extra else ''} → {run_dir}", flush=True)
    integ = det["integration"]
    rows, dets_all = [], {}
    for i, path in enumerate(images, 1):
        t0 = time.perf_counter()
        meta, img, mask, geo = prepare(path, pre_cfg, det, a.cache)
        dets, applied = [], False
        if img is not None:
            img, applied = apply_extra(img, mask, extra, tags_of(path.name))
            for d in detect(img, cfg):
                if road_overlap(mask, d["bbox"]) >= integ["min_road_overlap"]:
                    x, y, w, h = to_original_bbox(d["bbox"], geo)
                    dets.append({"type": d["type"], "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)]})
        ms = (time.perf_counter() - t0) * 1000
        dets_all[path.name] = dets
        gt = gt_of(path)
        row = {"image": path.name, "dataset": name, "group": group.get(path.name, ""),
               "viewpoint": vps.get(path.name, ""), "condition": CONDITION, "detector": DETECTOR,
               "status": meta["status"], "unverified_mask": meta["unverified_mask"], "gamma": meta["gamma"],
               "extra_applied": applied, "det_scale": round(geo["scale_x"], 4) if geo else None,
               "total_ms": round(ms, 1), "has_gt": gt is not None, "clean": gt is not None and len(gt) == 0}
        for kind in KINDS:
            row[f"n_{kind}"] = sum(d["type"] == kind for d in dets)
        if gt is not None:
            for kind, e in evaluate(dets, gt, None).items():      # 원본 좌표끼리 — 잘린 정답 없음
                row[f"{kind}_n_gt"], row[f"{kind}_n_cut"] = e["n_gt"], e["n_cut"]
                for c in CRITERIA:
                    row.update({f"{kind}_{c}_{k}": v for k, v in e[c].items()})
        rows.append(row)
        if i % 50 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] {path.name}", flush=True)

    with open(run_dir / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    json.dump(dets_all, open(run_dir / "detections.json", "w"))
    json.dump({"dataset": a.dataset, "limit": a.limit, "preprocessing_config": str(pre_path.relative_to(ROOT)),
               "preprocessing_hash": config_hash(pre_cfg), "detection_config": str(det_path.relative_to(ROOT)),
               "det_set": a.det_set, "extra": extra, "line_hi_abs": hi_used, "calibrated": a.calibrate,
               "code": git_version(), "note": a.note}, open(run_dir / "run_config.json", "w"), ensure_ascii=False, indent=2)
    n_unv = sum(r["unverified_mask"] for r in rows)
    print(f"상태: { {s: sum(r['status'] == s for r in rows) for s in dict.fromkeys(r['status'] for r in rows)} }"
          f" · 검증 안 된 후보 마스크 {n_unv}장 · 감마 {sum(r['gamma'] for r in rows)}장 · 보정 적용 {sum(r['extra_applied'] for r in rows)}장")
    if not a.no_log:
        log_run(run_dir, rows, a.dataset, a.limit, a.config_name, a.note)
    print(f"완료 → {run_dir}")


if __name__ == "__main__":
    main()
