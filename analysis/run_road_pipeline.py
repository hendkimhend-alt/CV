"""
B 구성(균열 최종, 10/10 확정)을 팀 공통 방법(EVALUATION.md)으로 잰다.

B = 팀원 전처리 v1.1 (자동 노면 마스크 · 조건부 감마) → 연결 (노면 외접 영역 자르기 · 긴 변 1024)
    → 검출기 p2bd → 도로 위 판정 (후보 박스의 50% 이상이 노면 마스크 안이면 남김)
실행 코드 · 설정은 CV-main/ 그대로 쓴다 (CV-main/configs/preprocessing.json · detection_p2bd.json).
채점은 맨 위 src/의 in50 (원본 좌표, ROI로 잘린 정답 없음) → outputs/pipeline/<run>/results.csv
(compare_runs.py가 읽는 형식) + results/runs.csv 자동 기록 (run_pipeline.py와 같은 칸).

실행 (저장소 맨 위 폴더에서, opencv-contrib 필요 · 563장 약 20분):
  python analysis/run_road_pipeline.py --note "B 재현"
  python analysis/run_road_pipeline.py --limit 20 --no-log                  # 빠른 확인
  python analysis/run_road_pipeline.py --det-config configs/새것.json --config-name 새것 --note "…"   # B 위에 하나만 바꿔 시험
  python analysis/compare_runs.py A=<p2bd 실행> B=<이 실행>
rdd_test는 --allow-rdd-test를 줄 때만 (EVALUATION.md 7절 — 쓰면 experiments.md에 기록).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "CV-main"
# CV-main/src와 맨 위 src/에 같은 이름의 모듈(paths · data · evaluate · preprocess · roi · detect)이 있다.
# 먼저 CV-main 쪽을 불러 실행 함수를 잡아 두고, 겹치는 이름을 지운 뒤 맨 위 src/를 불러 채점에 쓴다.
SHARED = ("paths", "data", "evaluate", "preprocess", "roi", "detect", "roadcue")
sys.path.insert(0, str(MAIN / "src"))
from preprocessing.config import config_hash, load_config  # noqa: E402
from road_detection.config import check_runtime, load_detection_config  # noqa: E402
from road_detection.pipeline import run_image  # noqa: E402
sys.path.remove(str(MAIN / "src"))
for name in SHARED:
    sys.modules.pop(name, None)
sys.path.insert(0, str(ROOT / "src"))
from data import list_images, load_gt, load_viewpoints  # noqa: E402
from evaluate import CRITERIA, KINDS, evaluate  # noqa: E402
from paths import OUTPUT_DIR, imread  # noqa: E402
from run_pipeline import git_version, log_run, new_run_dir  # noqa: E402

PRE_CFG = MAIN / "configs" / "preprocessing.json"
DET_CFG = MAIN / "configs" / "detection_p2bd.json"
SPLIT_CSV = ROOT / "labels" / "split_rdd.csv"
CONDITION, DETECTOR = "v1.1_road", "D1hv"      # runs.csv의 조건 · 검출기 칸


def main():
    p = argparse.ArgumentParser(description="B 구성 (팀원 전처리 v1.1 + 연결 + 도로 위 판정 + p2bd)을 in50으로 채점")
    p.add_argument("--dataset", default="rdd_dev", choices=("rdd_dev", "rdd_test"))
    p.add_argument("--limit", type=int)
    p.add_argument("--allow-rdd-test", action="store_true", help="rdd_test는 최종 확인에만 (EVALUATION.md 7절)")
    p.add_argument("--pre-config", type=Path, default=PRE_CFG, help="전처리 설정 (기본 CV-main/configs/preprocessing.json)")
    p.add_argument("--det-config", type=Path, default=DET_CFG, help="검출 설정 (기본 CV-main/configs/detection_p2bd.json)")
    p.add_argument("--config-name", default="b_road", help="runs.csv의 설정 칸 (설정 파일을 바꾸면 이름도 바꿀 것)")
    p.add_argument("--note", default="")
    p.add_argument("--no-log", action="store_true", help="results/runs.csv에 남기지 않음")
    a = p.parse_args()
    if a.dataset == "rdd_test" and not a.allow_rdd_test:
        raise SystemExit("error:rdd_test는 --allow-rdd-test로만 (최종 확인용 — 쓰면 experiments.md 테스트 사용 기록에 적기)")

    pre_path, det_path = a.pre_config.resolve(), a.det_config.resolve()
    pre_cfg, det = load_config(pre_path), load_detection_config(det_path)
    check_runtime(det)
    name, _, images = list_images(a.dataset)
    images = images[:a.limit] if a.limit else images
    gt_of, vps = load_gt(a.dataset, name), load_viewpoints(a.dataset)
    group = {r["image"]: r["group"] for r in csv.DictReader(open(SPLIT_CSV, encoding="utf-8"))}
    run_dir = new_run_dir(OUTPUT_DIR / "pipeline")
    print(f"{a.dataset}: {len(images)}장 · B 구성 ({pre_path.relative_to(ROOT)} + {det_path.relative_to(ROOT)}) → {run_dir}")

    rows, dets_all = [], {}
    for i, path in enumerate(images, 1):
        t0 = time.perf_counter()
        r = run_image(imread(path), pre_cfg, det, image_id=path.name)
        ms = (time.perf_counter() - t0) * 1000
        dets = [{"type": c["type"], "bbox": c["bbox_original"]} for c in r["candidates"] if c["on_road"]]
        dets_all[path.name] = dets
        gt = gt_of(path)
        row = {"image": path.name, "dataset": name, "group": group.get(path.name, ""),
               "viewpoint": vps.get(path.name, ""), "condition": CONDITION, "detector": DETECTOR,
               "status": r["status"], "unverified_mask": r["unverified_mask"],
               "gamma": bool(r["preprocess"].metadata.get("gamma", {}).get("applied")),
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
               "code": git_version(), "note": a.note}, open(run_dir / "run_config.json", "w"), ensure_ascii=False, indent=2)
    n_unv = sum(r["unverified_mask"] for r in rows)
    print(f"상태: { {s: sum(r['status'] == s for r in rows) for s in dict.fromkeys(r['status'] for r in rows)} }"
          f" · 검증 안 된 후보 마스크 {n_unv}장 · 감마 {sum(r['gamma'] for r in rows)}장")
    if not a.no_log:
        log_run(run_dir, rows, a.dataset, a.limit, a.config_name, a.note)
    print(f"완료 → {run_dir}")


if __name__ == "__main__":
    main()
