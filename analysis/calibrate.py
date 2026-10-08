"""
강한 기준 맞추기 — 찾기 단계를 바꿨을 때, 강한 흔적의 총수가 기준과 같아지는 Hessian 강한 기준값(line_hi_abs)을 찾는다.

왜: 찾기를 바꾸면(크기 평균 · 텐서 전파 · 가이드 필터 · 전처리 등) 점수의 크기 자체가 달라진다. 기준값을 그대로 두면
    "방법이 좋아서"가 아니라 "흔적을 더 많이 · 적게 뽑아서" Recall · Precision이 변한 것과 구분이 안 된다.
    그래서 비교하기 전에 **강한 흔적 총수를 같게** 맞춘다 (고정값 하나 — 사진마다 다른 기준은 쓰지 않음).
기준 흔적 수: 개발 세트 사진(ROI · 1024만, 보정 없음)에서 기본 Hessian 선 점수(σ 1 · 1.5 · 2 최댓값)의 중심선 중
    32.5 이상인 픽셀 수 (흔적 양 약 5.9% — E8 · E9에서 정한 값). 정답은 보지 않는다 (사진만 씀).
맞춘 값: 바꾼 설정으로 같은 사진의 중심선 점수를 내림차순으로 놓고, 기준 흔적 수 번째 값.

실행 (저장소 맨 위 폴더에서, 개발 563장 · 텐서 전파가 있으면 약 9분):
  python analysis/calibrate.py --config configs/p2bd.json                      # → 128.609375 (p2bd에 저장된 값)
  python analysis/calibrate.py --config configs/p2bd.json --set guided_filter='{"r":4,"eps":"var","eps_scale":2.0}'
  python analysis/calibrate.py --config configs/p2bd.json --set … --save configs/새이름.json   # 맞춘 값을 넣어 새 설정 저장
  python analysis/calibrate.py --config configs/p2bd.json --conditions gamma --save configs/감마.json  # 전처리 단계를 바꿀 때
  --limit N: 처음 N장만 (빠른 확인용 — 실제 값은 전체로)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import list_images  # noqa: E402
from detect import (DEFAULT_CFG, _to_gray, find_input, line_centerline, line_centerline_interp,  # noqa: E402
                    line_score, tensor_propagate)
from metrics import measure_quality  # noqa: E402
from paths import imread  # noqa: E402
from preprocess import classify_quality, geometry_preprocess, preprocess_condition, validate_config  # noqa: E402
from run_pipeline import load_config, parse_overrides  # noqa: E402

REF_HI = 32.5                      # 기준 흔적 = 기본 Hessian 점수의 중심선 ≥ 32.5
KEEP = 200_000                     # 사진마다 위쪽 점수만 보관 (기준 흔적 수 계산에 충분)


def candidate_center(gray, cfg):
    """바꾼 설정의 중심선 점수 지도 — line_trace()가 강한 기준과 비교하는 바로 그 값."""
    score, angle = line_score(find_input(gray, cfg), cfg["line_sigmas"], cfg.get("line_scale_combine", "max"))
    if cfg.get("tensor_propagate", 0) > 0:
        field, nangle = tensor_propagate(score, angle, cfg)
        score, angle = (score * field, angle) if cfg.get("tensor_gate", False) else (field, nangle)
    nms = line_centerline_interp if cfg.get("nms_interp", False) else line_centerline
    return nms(score, angle)


def main():
    p = argparse.ArgumentParser(description="강한 흔적 총수를 기준과 같게 하는 Hessian 강한 기준값 찾기")
    p.add_argument("--config", required=True, help="바꾼 설정의 바탕 설정 파일 (예: configs/p2bd.json)")
    p.add_argument("--set", nargs="+", metavar="키=값", help="설정 파일 위에 덮어쓸 detect 설정")
    p.add_argument("--conditions", help="전처리 조건 하나 (설정 파일의 conditions 대신, 예: gamma, flatten+gamma)")
    p.add_argument("--dataset", default="rdd_dev", help="기본 rdd_dev (값 고르기는 개발 세트에서만)")
    p.add_argument("--limit", type=int, help="처음 N장만 (빠른 확인용)")
    p.add_argument("--save", help="맞춘 값(line_hi_abs)을 넣은 새 설정 파일 경로")
    a = p.parse_args()

    base = load_config(a.config)
    over = {**base.get("set", {}), **parse_overrides(a.set)}
    cfg = {**DEFAULT_CFG, "crack_find": "line", **over}
    roi = base.get("roi", "bottom_half")
    conds = [a.conditions] if a.conditions else base.get("conditions", ["none"])
    if len(conds) != 1:
        raise SystemExit("error:설정의 conditions가 하나여야 함 (조건마다 따로 맞춤)")
    cond = conds[0]

    _, _, images = list_images(a.dataset)
    images = images[:a.limit] if a.limit else images
    cfg_a = validate_config({"roi": roi})
    n_ref, vals = 0, []
    for i, path in enumerate(images, 1):
        ref, _ = geometry_preprocess(imread(path), cfg_a, path.name)
        g0 = _to_gray(ref)
        s0, a0 = line_score(g0, DEFAULT_CFG["line_sigmas"])                       # 기준: 기본 점수 (보정 없음)
        n_ref += int((line_centerline(s0, a0) >= REF_HI).sum())
        fixed, _ = preprocess_condition(ref, cfg_a, cond, classify_quality(measure_quality(ref)), {})
        c = candidate_center(_to_gray(fixed), cfg)
        v = c[c > 0]
        vals.append(np.sort(v)[-KEEP:])
        if i == 1 or i % 50 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] {path.name}", flush=True)
    v = np.sort(np.concatenate(vals))[::-1]
    hi = float(v[n_ref - 1])
    print(f"\n{a.dataset} {len(images)}장 · ROI {roi} · 조건 {cond}")
    print(f"기준 흔적 수 (기본 Hessian ≥ {REF_HI}) = {n_ref:,}")
    print(f"맞춘 강한 기준 line_hi_abs = {hi!r}")
    if a.save:
        out = {**base, "conditions": [cond], "line_hi_abs": hi, "set": over}
        out["description"] = (base.get("description", "") + f" · 강한 기준 다시 맞춤 ({Path(a.config).stem} 바탕)").strip(" ·")
        Path(a.save).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"저장 → {a.save}")


if __name__ == "__main__":
    main()
