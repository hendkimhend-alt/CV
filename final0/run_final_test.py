"""
한 장 테스트 실행기 (final0) — data/test/에 넣은 사진에 run_final.py와 같은 처리를 하고,
원본 위에 검출 박스만 그린 그림 한 장을 저장한다.

처리는 run_final.py의 함수를 그대로 불러 쓴다 (설정 · 순서 · 파라미터 같음, run_final.py는 바꾸지 않음):
  run_final.prepare      자동 노면 마스크 → 노면 영역 자르기 · 긴 변 1024
  run_final.detect_image 보정 → 균열 · 포트홀 검출 → 도로 위 판정 → 원본 좌표
정답이 없으므로 정확도는 계산하지 않는다.

출력: final-result-test/run_<시각>/<사진 이름>_bbox.jpg (실행마다 새 폴더)
  원본 위에 균열(빨강) · 포트홀(파랑) 박스만 — 노면 마스크 초록 칠 없음
  전처리(노면 마스크)에 실패하면 박스 없이 원본에 실패 상태를 적어 저장한다

실행 (final0 폴더에서, opencv-contrib-python 필요):
  python run_final_test.py                    # data/test/ 의 사진 (한 장 넣는 용도, 여러 장이면 모두 처리)
  python run_final_test.py --input <사진 파일 또는 폴더>
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_final as F  # noqa: E402  (src/ 경로 설정 · 처리 함수 · 설정 파일 위치를 그대로 씀)
from run_final_PBL import list_pbl, output_names  # noqa: E402  (폴더의 사진 목록 · 출력 이름 규칙)
from data import IMAGE_EXTENSIONS  # noqa: E402
from paths import DATA_DIR, imread, imwrite  # noqa: E402
from pothole import POTHOLE_DEFAULTS  # noqa: E402
from preprocessing.config import load_config  # noqa: E402
from road_detection.config import check_runtime, load_detection_config  # noqa: E402
from road_detection.render import draw_result  # noqa: E402

TEST_DIR = DATA_DIR / "test"
RESULT_DIR = ROOT / "final-result-test"


def main():
    p = argparse.ArgumentParser(description="data/test/의 사진에 최종 구성을 돌려 원본 위 박스 그림만 저장 (정확도 없음)")
    p.add_argument("--input", type=Path, default=TEST_DIR, help=f"사진 파일 또는 폴더 (기본 {TEST_DIR})")
    p.add_argument("--out", type=Path, default=RESULT_DIR, help="결과 폴더 (기본 final-result-test/) — 그 아래 run_<시각>/")
    a = p.parse_args()
    if a.input.is_file():
        if a.input.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SystemExit(f"error:사진 파일이 아님 {a.input} (쓸 수 있는 확장자: {sorted(IMAGE_EXTENSIONS)})")
        images, base = [a.input], a.input.parent
    elif a.input.is_dir():
        images, base = list_pbl(a.input), a.input
    else:
        raise SystemExit(f"error:사진 파일 · 폴더 없음 {a.input}")
    if not images:
        raise SystemExit(f"error:사진 없음 {a.input} — 사진 한 장을 넣고 다시 실행하세요")
    if len(images) > 1:
        print(f"참고: 사진이 {len(images)}장 있어 모두 처리합니다 (한 장 테스트용 폴더)", flush=True)

    # 설정은 run_final.py와 같은 파일 · 같은 방법으로 읽는다
    pre_cfg, det = load_config(F.PRE_CFG), load_detection_config(F.DET_CFG)
    check_runtime(det)
    pc = json.loads(F.POT_CFG.read_text(encoding="utf-8"))["pothole"]
    unknown = set(pc) - set(POTHOLE_DEFAULTS)
    if unknown:
        raise SystemExit(f"error:{F.POT_CFG.name} — 모르는 키 {sorted(unknown)}")
    pothole_cfg = {**POTHOLE_DEFAULTS, **pc}

    names = output_names(images, base)
    run_dir = F.new_run_dir(a.out)
    for path in images:
        t0 = time.perf_counter()
        try:
            image = imread(path)
        except ValueError as exc:
            print(f"{path.name}  읽기 실패 — 건너뜀 ({exc})", flush=True)
            continue
        meta, img, mask, geo = F.prepare(path, pre_cfg, det, image)
        dets = []
        if img is not None:
            dets, _, _, _ = F.detect_image(img, mask, geo, det, pothole_cfg, set())
        ms = (time.perf_counter() - t0) * 1000
        n_crack = sum(d["type"] == "crack" for d in dets)
        n_pothole = sum(d["type"] == "pothole" for d in dets)
        title = f"{path.name}  crack {n_crack} / pothole {n_pothole}"
        if meta["status"] != "detected":
            title += f"  [{meta['status']}]"
        boxes = [{"type": d["type"], "bbox_original": d["bbox"], "on_road": True} for d in dets]
        out = run_dir / f"{names[path]}_bbox.jpg"
        imwrite(out, draw_result(image, None, boxes, title=title))       # 노면 마스크 없이 원본 위 박스만
        print(f"{path.name}  {meta['status']} · 균열 {n_crack} · 포트홀 {n_pothole} · {ms:.0f} ms → {out}", flush=True)
    print(f"완료 → {run_dir}")


if __name__ == "__main__":
    main()
