"""
최종 구성 실행기 (final0) — 사진 한 장마다:

  [전처리] 자동 노면 마스크 (FAIL이면 검증 안 된 후보 마스크로 계속)
  → [연결] 노면 외접 영역 자르기(여백 16px) · 긴 변 1024
  → [검출기 입력 보정] 노면 마스크 안에만 — 선명한 사진: 가우시안 σ0.8 → 막대 닫힘 7px → CLAHE 1.0
                                              아주 흐린 사진(노면 선명도 < 2.76): 언샤프 3.0 · σ3 → CLAHE 1.0
  → [균열] detect.py — Hessian 찾기 · 텐서 전파 · 버리기(길이 · 양쪽 확인 · 노면 단서 · 깊이)
  → [포트홀] pothole.py — 주변 노면 고리보다 거친 곳 → 색 · 어두운 부분 · 모양 · 위치로 버리기
  → [도로 위 판정] 균열 · 포트홀 공통: 후보 박스의 50% 이상이 노면 마스크 안이면 남김 → 원본 좌표
  → [채점] 정답이 있으면 in50 (원본 좌표끼리)

설정: configs/preprocessing.json (전처리) · configs/detection_final.json (균열 검출기 + 보정 + 연결) ·
      configs/pothole_final.json (포트홀)
원본: 저장소 analysis/run_road_pipeline.py --final-config configs/final.json 과 같은 처리 순서 · 같은 설정
      (실험용 옵션 --extra · --calibrate · --degrade · --cache · --det-set은 뺐다)

결과는 final-result/run_<시각>/ 에 바로 저장된다 (README.md 표 참고):
  images/       결과 그림 — 노면 마스크(초록) · 균열(빨강) · 포트홀(파랑) · 정답(흰색, 있을 때)
  road_mask/    노면 마스크 (흰색 = 노면)          road_view/  원본에서 노면 밖을 어둡게
  preprocessed/ 검출기에 실제로 들어간 영상 (노면 영역 자르기 · 긴 변 1024 · 보정)
  review_sheet.jpg · results.csv (사진별) · summary.csv (전체 · 그룹 · 폴더별 정확도와 측정값)
  detections.json · run_config.json

정답(정확도)을 찾는 순서: --gt 파일 → labels/<데이터 이름>.csv → 사진 폴더의 annotations.json · annotations.csv
  → (provided) data/road_damage_annotated/annotations.json → RDD는 data/RDD2020_train/train/ann/

실행 (final0 폴더에서, opencv-contrib-python 필요):
  python run_final.py                          # rdd_dev 563장 → final-result/run_<시각>/
  python run_final.py --limit 20               # 빠른 확인
  python run_final.py --dataset provided       # 제공 13장
  python run_final.py --dataset <폴더 경로>     # 아무 사진 폴더 (하위 폴더까지, 폴더별 요약도 냄)
  python run_final.py --dataset <폴더> --gt <정답 파일>       # 정답 파일 직접 지정 (.json · .csv)
  python run_final.py --dataset rdd_test --allow-rdd-test    # 테스트 세트 — 설정을 고정한 뒤 한 번만
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import list_images, load_gt, load_gt_file, load_viewpoints  # noqa: E402
from detect import detect  # noqa: E402
from evaluate import CRITERIA, KINDS, evaluate, prf, prf_in50  # noqa: E402
from paths import DATA_DIR, LABELS_DIR, imread, imwrite  # noqa: E402
from pothole import POTHOLE_DEFAULTS, texture_potholes  # noqa: E402
from preprocessing.config import config_hash, load_config  # noqa: E402
from preprocessing.detector_adapter import to_detector_input, to_original_bbox  # noqa: E402
from preprocessing.detector_corrections import apply_corrections, judge_tags, road_sharpness  # noqa: E402
from preprocessing.pipeline import MANUAL_REQUIRED, SUCCESS, process_image  # noqa: E402
from road_detection.config import check_runtime, load_detection_config  # noqa: E402
from road_detection.pipeline import road_overlap  # noqa: E402
from road_detection.render import draw_result, review_sheet  # noqa: E402

RESULT_DIR = ROOT / "final-result"
SHEET_MAX = 12                                  # review_sheet.jpg에 넣는 사진 수
PRE_CFG = ROOT / "configs" / "preprocessing.json"
DET_CFG = ROOT / "configs" / "detection_final.json"
POT_CFG = ROOT / "configs" / "pothole_final.json"
SPLIT_CSV = LABELS_DIR / "split_rdd.csv"
PROVIDED_GT = DATA_DIR / "road_damage_annotated" / "annotations.json"
PROTECTED = ("rdd", "rdd_test")                 # 테스트 세트가 들어 있는 데이터셋
GT_COLOR = (255, 255, 255)                      # 결과 그림의 정답 박스 (흰색)
CANNY_EDGES = (100, 200)                        # 에지 수 — 계획서 3-6 표와 같은 기준 (저장소 analysis/measure_prepost.py)


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


def prepare(path, pre_cfg, det, image=None):
    """전처리 + 연결 → (기록, 검출기 입력 영상, 검출기 마스크, geometry). 전처리 실패면 영상 None.
    기록: "road_mask" = 원본 크기 노면 마스크 (실패면 None) · "mask_check" = 자동 마스크 검증 PASS / FAIL."""
    image = imread(path) if image is None else image
    pre = process_image(image, pre_cfg, image_id=path.name)
    mask_check = pre.metadata.get("validation", {}).get("status")
    unverified = False
    if pre.status == MANUAL_REQUIRED and det["integration"]["on_fail"] == "use_candidate" and pre.auto_mask is not None \
            and np.count_nonzero(pre.auto_mask):
        # 검증되지 않은 자동 후보 마스크로 전처리를 끝까지 진행
        pre = process_image(image, pre_cfg, image_id=path.name, manual_mask=pre.auto_mask,
                            manual_source="auto_candidate_unverified")
        unverified = True
    meta = {"status": pre.status, "unverified_mask": unverified, "mask_check": mask_check,
            "gamma": bool(pre.metadata.get("gamma", {}).get("applied")), "road_mask": pre.road_mask}
    if pre.status != SUCCESS:
        meta["status"] = "skipped_" + pre.status
        return meta, None, None, None
    integ = det["integration"]
    img, mask, geo = to_detector_input(pre.processed_image, pre.road_mask, integ["long_side"],
                                       crop_to_road=integ["input_mode"] == "crop_to_road",
                                       crop_margin=integ["crop_margin"])
    meta["status"] = "detected"
    return meta, img, mask, geo


def detect_image(img, mask, geo, det, pothole_cfg, base_tags):
    """검출기 입력 한 장 → (원본 좌표 후보 목록, 보정 적용했나, 보정된 검출기 입력 영상, 태그).
    균열 · 포트홀을 같은 보정 영상에서 따로 찾는다."""
    corr = det["corrections"]
    tags = judge_tags(img, mask, geo, corr, base_tags)
    img, applied = apply_corrections(img, mask, corr, tags)
    cands = [d for d in detect(img, det["detect_cfg"]) if d["type"] != "pothole"] + \
            [d for d in texture_potholes(img, pothole_cfg, mask=mask) if d["type"] == "pothole"]
    dets = []
    for d in cands:                                  # 도로 위 판정은 균열 · 포트홀 공통
        if road_overlap(mask, d["bbox"]) >= det["integration"]["min_road_overlap"]:
            x, y, w, h = to_original_bbox(d["bbox"], geo)
            dets.append({"type": d["type"], "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)]})
    return dets, applied, img, tags


def measure(proc, mask, sift):
    """검출기 입력(보정 뒤)의 노면 마스크 안 에지 수 · 특징점 수 — 계획서 3-6 표의 '전처리 후' 값과 같은 방법."""
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    road = (mask > 0).astype(np.uint8)
    return {"edges": int(((cv2.Canny(gray, *CANNY_EDGES) > 0) & (road > 0)).sum()),
            "keypoints": len(sift.detect(gray, road)), "road_px_det": int(road.sum())}


def find_gt(a, name, folder):
    """정답 로더와 출처 설명. 순서: --gt → labels/<이름>.csv · RDD ann → 사진 폴더의 annotations.* → 제공 13장 초안 정답."""
    if a.gt:
        return load_gt_file(a.gt), str(a.gt)
    if str(a.dataset).startswith("rdd"):
        return load_gt(a.dataset, name), "RDD ann/*.json"
    if (LABELS_DIR / f"{name}.csv").exists():
        return load_gt(a.dataset, name), f"labels/{name}.csv"
    for cand in (folder / "annotations.json", folder / "annotations.csv"):
        if cand.exists():
            return load_gt_file(cand), str(cand)
    if name == "provided" and PROVIDED_GT.exists():
        return load_gt_file(PROVIDED_GT), str(PROVIDED_GT)
    return (lambda path: None), None


def summarize(rows):
    """한 묶음(전체 · 그룹 · 폴더)의 지표.
    정확도는 정답 있는 사진만 — Recall = 맞힌 정답 ÷ 정답 수 · Precision = 정답 박스 안 후보 ÷ 전체 후보
    · 가짜/장 = 가짜 ÷ 정답 있는 사진 수 · 조각/맞힘 = 정답 박스 안 후보 ÷ 맞힌 정답 (저장소 run_pipeline.run_log_metrics와 같은 계산)
    측정값은 검출까지 간 사진의 평균 (계획서 3-6 표 항목)."""
    rnd = lambda v, k=4: None if v is None else round(v, k)
    mean = lambda k, rs: rnd(float(np.mean([r[k] for r in rs])), 1) if rs else None
    done = [r for r in rows if r["status"] == "detected"]
    out = {"n_images": len(rows), "n_detected": len(done), "n_unverified_mask": sum(r["unverified_mask"] for r in rows),
           "n_blur_rt": sum(r["blur_judge"] == "blur_rt" for r in rows),
           "cands_per_img": mean("n_all", rows), "cand_box_area_per_img": mean("cand_box_area", rows),
           "edges_per_img": mean("edges", done), "keypoints_per_img": mean("keypoints", done),
           "road_ratio_mean": rnd(float(np.mean([r["road_ratio"] for r in done])), 3) if done else None,
           "pre_ms_per_img": mean("pre_ms", rows), "det_ms_per_img": mean("det_ms", rows),
           "total_ms_per_img": mean("total_ms", rows)}
    rs = [r for r in rows if r["has_gt"]]
    out["n_gt_images"] = len(rs)
    for kind in KINDS:
        if not rs:
            continue
        tp, fp, fn = (sum(r[f"{kind}_in50_{k}"] for r in rs) for k in ("tp", "fp", "fn"))
        n = sum(r[f"n_{kind}"] for r in rs)
        p, r_, f1 = prf_in50(tp, fp, fn, n)
        out.update({f"{kind}_gt": sum(r[f"{kind}_n_gt"] for r in rs), f"{kind}_hit": tp, f"{kind}_fake": fp,
                    f"{kind}_cands": n,
                    f"{kind}_recall": rnd(r_), f"{kind}_precision": rnd(p), f"{kind}_f1": rnd(f1),
                    f"{kind}_fake_per_img": round(fp / len(rs), 2),
                    f"{kind}_frag_per_hit": round((n - fp) / tp, 2) if tp else None,
                    f"{kind}_precision_hit": rnd(prf(tp, fp, fn)[0])})
    return out


def scopes(rows):
    """요약 묶음: 전체 · 그룹(RDD 흐림 · 그림자 · 정상) · 폴더(하위 폴더가 둘 이상일 때)."""
    out = [("전체", rows)]
    for g in sorted({t for r in rows for t in r["group"].split(";") if t}):
        out.append((f"그룹:{g}", [r for r in rows if g in r["group"].split(";")]))
    folders = sorted({r["folder"] for r in rows})
    if len(folders) > 1:
        out += [(f"폴더:{f}", [r for r in rows if r["folder"] == f]) for f in folders]
    return out


def main():
    p = argparse.ArgumentParser(description="최종 구성 (전처리 + 균열 · 포트홀 검출 + 도로 위 판정) 실행 · in50 채점")
    p.add_argument("--dataset", default="rdd_dev",
                   help="rdd_dev · rdd_tune · rdd_val · rdd_test · provided · captured 또는 사진 폴더 경로")
    p.add_argument("--gt", type=Path, help="정답 파일 (.json: annotations.json 형식 · .csv: image,x,y,w,h,type)")
    p.add_argument("--limit", type=int, help="처음 N장만")
    p.add_argument("--allow-rdd-test", action="store_true", help="테스트 세트는 설정을 고정한 뒤 최종 확인에만")
    p.add_argument("--out", type=Path, default=RESULT_DIR, help="결과 폴더 (기본 final-result/) — 그 아래 run_<시각>/")
    p.add_argument("--no-images", action="store_true", help="그림(결과 · 노면 마스크 · 전처리 영상)을 저장하지 않음 (CSV만)")
    p.add_argument("--note", default="")
    a = p.parse_args()
    if a.dataset in PROTECTED and not a.allow_rdd_test:
        raise SystemExit(f"error:{a.dataset}에는 테스트 세트가 들어 있음 — --allow-rdd-test로만 (최종 확인용)")
    if a.gt and not a.gt.exists():
        raise SystemExit(f"error:정답 파일 없음 {a.gt}")

    pre_cfg, det = load_config(PRE_CFG), load_detection_config(DET_CFG)
    check_runtime(det)
    pc = json.loads(POT_CFG.read_text(encoding="utf-8"))["pothole"]
    unknown = set(pc) - set(POTHOLE_DEFAULTS)
    if unknown:
        raise SystemExit(f"error:{POT_CFG.name} — 모르는 키 {sorted(unknown)}")
    pothole_cfg = {**POTHOLE_DEFAULTS, **pc}

    name, folder, images = list_images(a.dataset)
    folder = Path(folder)
    images = images[:a.limit] if a.limit else images
    gt_of, gt_source = find_gt(a, name, folder)
    vps = load_viewpoints(a.dataset)
    group = {}
    if SPLIT_CSV.exists():
        group = {r["image"]: r["group"] for r in csv.DictReader(open(SPLIT_CSV, encoding="utf-8"))}
    sift = cv2.SIFT_create()

    run_dir = new_run_dir(a.out)
    print(f"{name}: {len(images)}장 → {run_dir}", flush=True)
    print(f"정답: {gt_source or '없음 — 정확도 없이 검출 · 측정만'}", flush=True)
    rows, dets_all, sheet = [], {}, []
    for i, path in enumerate(images, 1):
        rel = path.relative_to(folder)
        key = "__".join(rel.with_suffix("").parts)   # 하위 폴더가 달라도 이름이 겹치지 않게 (폴더__이름)
        t0 = time.perf_counter()                     # 처리 시간: 읽기 ~ 검출 (원본 실행기와 같은 구간)
        image = imread(path)
        meta, img, mask, geo = prepare(path, pre_cfg, det, image)
        t1 = time.perf_counter()
        dets, applied, proc, tags = [], False, None, set()
        if img is not None:
            dets, applied, proc, tags = detect_image(img, mask, geo, det, pothole_cfg,
                                                     set(group.get(path.name, "").split(";")))
        t2 = time.perf_counter()
        dets_all[str(rel.as_posix())] = dets
        gt = gt_of(path)

        H, W = image.shape[:2]
        row = {"image": path.name, "folder": rel.parent.as_posix(), "dataset": name, "group": group.get(path.name, ""),
               "viewpoint": vps.get(path.name, ""), "width": W, "height": H, "status": meta["status"],
               "mask_check": meta["mask_check"], "unverified_mask": meta["unverified_mask"],
               "road_ratio": round(float(np.count_nonzero(meta["road_mask"])) / (H * W), 4)
               if meta["road_mask"] is not None else None,
               "sharpness": round(road_sharpness(img, mask, geo), 3) if img is not None else None,
               "blur_judge": next((t for t in ("blur_rt", "not_blur_rt") if t in tags), ""),
               "gamma": meta["gamma"], "corrections_applied": applied,
               "det_scale": round(geo["scale_x"], 4) if geo else None,
               "pre_ms": round((t1 - t0) * 1000, 1), "det_ms": round((t2 - t1) * 1000, 1),
               "total_ms": round((t2 - t0) * 1000, 1)}
        for kind in KINDS:
            row[f"n_{kind}"] = sum(d["type"] == kind for d in dets)
        row["n_all"] = len(dets)
        row["cand_box_area"] = round(sum(d["bbox"][2] * d["bbox"][3] for d in dets), 1)   # 원본 px
        row.update(measure(proc, mask, sift) if proc is not None else {"edges": None, "keypoints": None, "road_px_det": None})
        row["has_gt"], row["clean"] = gt is not None, gt is not None and len(gt) == 0
        if gt is not None:
            for kind, e in evaluate(dets, gt, None).items():      # 원본 좌표끼리 — 잘린 정답 없음
                row[f"{kind}_n_gt"], row[f"{kind}_n_cut"] = e["n_gt"], e["n_cut"]
                for c in CRITERIA:
                    row.update({f"{kind}_{c}_{k}": v for k, v in e[c].items()})
        rows.append(row)

        if not a.no_images:
            n_of = {k: row[f"n_{k}"] for k in KINDS}
            title = f"{rel.as_posix()}  crack {n_of['crack']} / pothole {n_of['pothole']}"
            if gt is not None:
                title += "  " + " ".join(f"{k} {row[f'{k}_in50_tp']}/{row[f'{k}_n_gt']}" for k in KINDS if row[f"{k}_n_gt"])
            if meta["status"] != "detected":
                title += f"  [{meta['status']}]"
            vis = draw_result(image, meta["road_mask"], [{"type": d["type"], "bbox_original": d["bbox"], "on_road": True}
                                                         for d in dets], title=title)
            if gt:                                   # 정답 박스 (흰색) — draw_result와 같은 축소 배율
                s = vis.shape[1] / W
                for g in gt:
                    cv2.rectangle(vis, (int(g[1] * s), int(g[2] * s)), (int(g[3] * s), int(g[4] * s)), GT_COLOR, 1)
            imwrite(run_dir / "images" / f"{key}.jpg", vis)
            if len(sheet) < SHEET_MAX:
                sheet.append(vis)
            if meta["road_mask"] is not None:
                road = meta["road_mask"] > 0
                imwrite(run_dir / "road_mask" / f"{key}.png", road.astype(np.uint8) * 255)
                view = image.copy()
                view[~road] = (view[~road] * 0.3).astype(np.uint8)          # 노면 밖을 어둡게
                imwrite(run_dir / "road_view" / f"{key}.jpg", view)
            if proc is not None:
                imwrite(run_dir / "preprocessed" / f"{key}.jpg", proc)
        if i % 50 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] {rel.as_posix()}", flush=True)

    with open(run_dir / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    json.dump(dets_all, open(run_dir / "detections.json", "w", encoding="utf-8"), ensure_ascii=False)
    if sheet:
        imwrite(run_dir / "review_sheet.jpg", review_sheet(sheet))
    summary = [{"scope": s, **summarize(rs)} for s, rs in scopes(rows)]
    with open(run_dir / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in summary for k in r)), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)
    json.dump({"dataset": name, "limit": a.limit, "gt": gt_source, "preprocessing_config": PRE_CFG.name,
               "preprocessing_hash": config_hash(pre_cfg), "detection_config": DET_CFG.name,
               "detect_cfg": det["detect_cfg"], "corrections": det["corrections"], "integration": det["integration"],
               "pothole": {k: v for k, v in pothole_cfg.items() if k.startswith("pothole")},
               "summary": summary, "note": a.note},
              open(run_dir / "run_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)

    s0 = summary[0]
    print(f"상태: { {s: sum(r['status'] == s for r in rows) for s in dict.fromkeys(r['status'] for r in rows)} }"
          f" · 검증 안 된 후보 마스크 {s0['n_unverified_mask']}장 · 아주 흐림 {s0['n_blur_rt']}장")
    print(f"측정 (장당 평균): 후보 {s0['cands_per_img']} · 후보 박스 면적 {s0['cand_box_area_per_img']} px · "
          f"에지 {s0['edges_per_img']} · 특징점 {s0['keypoints_per_img']} · 처리 시간 {s0['total_ms_per_img']} ms")
    if s0["n_gt_images"]:
        print(f"정확도 (in50, 정답 있는 사진 {s0['n_gt_images']}장):")
        for s in summary:
            if not s["n_gt_images"]:
                continue
            line = " | ".join(f"{k} {s.get(f'{k}_hit')}/{s.get(f'{k}_gt')} R {s.get(f'{k}_recall')} P {s.get(f'{k}_precision')}"
                              f" F1 {s.get(f'{k}_f1')} 가짜 {s.get(f'{k}_fake_per_img')}/장" for k in KINDS)
            print(f"  {s['scope']:<24} {line}")
    print(f"완료 → {run_dir}")


if __name__ == "__main__":
    main()
