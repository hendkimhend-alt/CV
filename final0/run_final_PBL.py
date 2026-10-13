"""
PBL 데이터셋 실행기 (final0) — run_final.py와 같은 처리를 PBL 사진 폴더에 돌리고, 사진마다 그림 다섯 장을 저장한다.

처리는 run_final.py의 함수를 그대로 불러 쓴다 (설정 · 순서 · 파라미터 같음, run_final.py는 바꾸지 않음):
  run_final.prepare      자동 노면 마스크 → 노면 영역 자르기 · 긴 변 1024
  run_final.detect_image 보정(질감 줄이기 · CLAHE / 아주 흐리면 언샤프) → 균열 · 포트홀 검출 → 도로 위 판정 → 원본 좌표
  draw_result            run_final.py의 결과 그림과 같은 그리기 (노면 마스크 초록 · 균열 빨강 · 포트홀 파랑)

PBL 사진에는 정답이 없으므로 정확도(Recall · Precision · F1 · IoU)는 계산하지 않는다.

출력: final-result-PBL/run_<시각>/ (실행마다 새 폴더 — run_final.py의 final-result/와 따로)
  <사진 이름>_1_original.<원본 확장자>  원본 그대로 복사 (입력 파일은 건드리지 않음)
  <사진 이름>_2_road_mask.jpg           원본 위에 노면 마스크만 초록 칠 · 노란 테두리 (박스 없음)
  <사진 이름>_3_preprocessed.jpg        전처리된 영상 = 검출기에 실제로 들어간 영상
                                        (노면 영역 자르기 · 긴 변 1024 · 노면 마스크 안 보정) — run_final.py의 preprocessed/와 같은 것
  <사진 이름>_4_crack_bbox.jpg          원본 위에 균열 박스(빨강)만 (초록 칠 · 포트홀 없음)
  <사진 이름>_5_mask_bbox.jpg           원본 위에 초록 노면 칠 + 균열(빨강) · 포트홀(파랑) 박스 — run_final.py의 images/와 같은 그림
  전처리에 실패한 사진: 2 · 3번은 원본에 실패 상태를 적은 그림, 4 · 5번은 박스 없는 그림 (검출 없음)

실행 (final0 폴더에서, opencv-contrib-python 필요):
  python run_final_PBL.py
  python run_final_PBL.py --input <다른 사진 폴더>
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_final as F  # noqa: E402  (src/ 경로 설정 · 처리 함수 · 설정 파일 위치를 그대로 씀)
from data import IMAGE_EXTENSIONS  # noqa: E402
from paths import DATA_DIR, imread, imwrite  # noqa: E402
from pothole import POTHOLE_DEFAULTS  # noqa: E402
from preprocessing.config import load_config  # noqa: E402
from road_detection.config import check_runtime, load_detection_config  # noqa: E402
from road_detection.render import draw_result  # noqa: E402

PBL_DIR = DATA_DIR / "PBL module 1 - dataset"
RESULT_DIR = ROOT / "final-result-PBL"


def list_pbl(folder):
    """폴더(하위 폴더까지)의 사진 — run_final.py의 폴더 입력(data.list_images)과 같은 규칙 (확장자 · 숨김 경로 제외)."""
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                  and not any(part.startswith(".") for part in p.relative_to(folder).parts))


def output_names(images, folder):
    """사진마다 출력 이름 — 보통 파일 이름(확장자 제외), 하위 폴더에 같은 이름이 있으면 폴더__이름."""
    stems = [p.stem for p in images]
    return {p: (p.stem if stems.count(p.stem) == 1 else "__".join(p.relative_to(folder).with_suffix("").parts))
            for p in images}


def main():
    p = argparse.ArgumentParser(description="PBL 사진 폴더에 최종 구성을 돌려 사진마다 그림 다섯 장 저장 — 원본 · 노면 마스크 · 전처리 · 균열 박스 · 노면+박스 (정확도 없음)")
    p.add_argument("--input", type=Path, default=PBL_DIR, help=f"사진 폴더 (기본 {PBL_DIR})")
    p.add_argument("--out", type=Path, default=RESULT_DIR, help="결과 폴더 (기본 final-result-PBL/) — 그 아래 run_<시각>/")
    a = p.parse_args()
    if not a.input.is_dir():
        raise SystemExit(f"error:사진 폴더 없음 {a.input}")

    # 설정은 run_final.py와 같은 파일 · 같은 방법으로 읽는다
    pre_cfg, det = load_config(F.PRE_CFG), load_detection_config(F.DET_CFG)
    check_runtime(det)
    pc = json.loads(F.POT_CFG.read_text(encoding="utf-8"))["pothole"]
    unknown = set(pc) - set(POTHOLE_DEFAULTS)
    if unknown:
        raise SystemExit(f"error:{F.POT_CFG.name} — 모르는 키 {sorted(unknown)}")
    pothole_cfg = {**POTHOLE_DEFAULTS, **pc}

    images = list_pbl(a.input)
    if not images:
        raise SystemExit(f"error:사진 없음 {a.input}")
    names = output_names(images, a.input)
    run_dir = F.new_run_dir(a.out)
    print(f"PBL: {len(images)}장 → {run_dir}", flush=True)
    print("정답 없음 — 정확도(Recall · Precision · F1 · IoU)는 계산하지 않음", flush=True)

    failed = []
    for i, path in enumerate(images, 1):
        name = names[path]
        t0 = time.perf_counter()
        try:
            image = imread(path)
        except ValueError as exc:                    # 읽을 수 없는 파일은 건너뛰고 계속
            failed.append((path.name, str(exc)))
            print(f"  [{i}/{len(images)}] {path.name}  읽기 실패 — 건너뜀", flush=True)
            continue
        meta, img, mask, geo = F.prepare(path, pre_cfg, det, image)
        dets, proc, tags = [], None, set()
        if img is not None:
            dets, _, proc, tags = F.detect_image(img, mask, geo, det, pothole_cfg, set())
        ms = (time.perf_counter() - t0) * 1000

        n_crack = sum(d["type"] == "crack" for d in dets)
        n_pothole = sum(d["type"] == "pothole" for d in dets)
        title = f"{path.name}  crack {n_crack} / pothole {n_pothole}"     # run_final.py 결과 그림과 같은 제목 (정답 부분 없음)
        if meta["status"] != "detected":
            title += f"  [{meta['status']}]"
        boxes = [{"type": d["type"], "bbox_original": d["bbox"], "on_road": True} for d in dets]
        failed_note = f"  road surface processing failed [{meta['status']}]"

        # 1) 원본 — 파일 그대로 복사
        shutil.copy2(path, run_dir / f"{name}_1_original{path.suffix}")
        # 2) 노면 마스크 초록 칠 (박스 없음)
        imwrite(run_dir / f"{name}_2_road_mask.jpg",
                draw_result(image, meta["road_mask"], [], title=f"{path.name}  road mask"
                            + ("" if meta["road_mask"] is not None else failed_note)))
        # 3) 전처리된 영상 — 검출기 입력 (전처리 실패면 원본에 상태 표시)
        imwrite(run_dir / f"{name}_3_preprocessed.jpg",
                proc if proc is not None else draw_result(image, None, [], title=path.name + failed_note))
        # 4) 균열 박스만 (초록 칠 없음)
        imwrite(run_dir / f"{name}_4_crack_bbox.jpg",
                draw_result(image, None, [b for b in boxes if b["type"] == "crack"],
                            title=f"{path.name}  crack {n_crack}" + ("" if meta["status"] == "detected" else f"  [{meta['status']}]")))
        # 5) 초록 칠 + 균열 · 포트홀 박스 — run_final.py images/와 같은 그리기
        imwrite(run_dir / f"{name}_5_mask_bbox.jpg", draw_result(image, meta["road_mask"], boxes, title=title))

        blur = "아주 흐림(언샤프)" if "blur_rt" in tags else ("선명(질감 줄이기)" if "not_blur_rt" in tags else "-")
        mask_note = " · 검증 안 된 마스크" if meta["unverified_mask"] else ""
        print(f"  [{i}/{len(images)}] {path.name}  {meta['status']} · {blur}{mask_note} · 균열 {n_crack} · 포트홀 {n_pothole} · {ms:.0f} ms",
              flush=True)

    print(f"저장: 사진 {len(images) - len(failed)}장 × 5 = 그림 {5 * (len(images) - len(failed))}장"
          + (f" · 읽기 실패 {len(failed)}장" if failed else ""))
    for n, msg in failed:
        print(f"  읽기 실패: {n} — {msg}")
    print(f"완료 → {run_dir}")


if __name__ == "__main__":
    main()
