"""개발 A 전처리·자동 품질 분류: py m1/A.py [--limit N]."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .metrics import METRIC_NAMES, measure_quality
    from .preprocess import (CONDITIONS, GEOMETRY_COLUMNS, PARAMETER_COLUMNS,
                             classify_quality, geometry_preprocess, preprocess_condition, validate_config, validate_image,
                             validate_thresholds)
else:
    from metrics import METRIC_NAMES, measure_quality
    from preprocess import (CONDITIONS, GEOMETRY_COLUMNS, PARAMETER_COLUMNS,
                            classify_quality, geometry_preprocess, preprocess_condition, validate_config, validate_image,
                            validate_thresholds)

# 기능: 이미지 입력은 아래 한 줄을 수정한다. 출력은 기존처럼 m1의 새 실행 폴더다.
# 특징: 수동 그룹 파일 없이 보정 전 품질값으로 자동 분류하고 세 조건을 모두 실행한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "img" / "RDD2020_train" / "train" / "img" #!!!입력값!!!!
DEFAULT_OUTPUT = PROJECT_ROOT / "m1"                                     #!!!출력위치!!!
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROCESSING_ERRORS = (ValueError, OSError, UnicodeError, cv2.error, csv.Error)
PLOT_METRICS = ("gray_mean", "block_mean_std_4x4", "laplacian_variance", "noise_sigma", "saturation_ratio")
RESULT_COLUMNS = ["image_id", "input_path", "dataset", "dataset_kind", "condition", "tags", "reviewed",
                  "group_reason", "status", "reason", "preprocessing_ms", "output_path", "stages_json",
                  "metric_warnings", "group_source", "reference_laplacian_variance",
                  "reference_block_mean_std_4x4"] + GEOMETRY_COLUMNS + PARAMETER_COLUMNS + list(METRIC_NAMES)
SUMMARY_COLUMNS = ["dataset", "condition", "group", "attempted_n", "success_n", "skipped_n", "failed_n"]
for name in (*METRIC_NAMES, "preprocessing_ms"):
    SUMMARY_COLUMNS.extend((name + "_n", "mean_" + name))


# 기능: 처리 예외를 콘솔·CSV 공통의 error:원인 한 줄로 정리한다.
# 특징: 긴 OpenCV 빌드 경로와 중복 접두어를 제거하고 개별 이미지 실패는 배치에서 계속 처리한다.
def error_text(cause):
    reason = f"{cause.func}: {cause.err}" if isinstance(cause, cv2.error) else str(cause)
    reason = " ".join(reason.split())
    while reason.lower().startswith("error:"):
        reason = reason[6:].lstrip()
    return "error:" + (reason or "원인 정보 없음")


# 기능: 지정한 한 폴더와 하위 폴더에서 지원 확장자의 이미지를 찾는다.
# 특징: 숨김 경로를 제외하고 상대 경로 ID로 정렬한다. JSON 정답이나 검출기를 읽지 않는다.
def discover_images(root, dataset):
    if not root.is_dir():
        raise ValueError("error:입력 이미지 폴더를 찾을 수 없음")
    if not isinstance(dataset, str) or not dataset.strip() or dataset != dataset.strip() or any(c in dataset for c in ":/\\"):
        raise ValueError("error:데이터 이름이 비었거나 경로 구분 문자를 포함함")
    images = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not any(p.startswith(".") for p in relative.parts):
            images.append((f"{dataset}:{relative.as_posix()}", path, relative))
    return sorted(images, key=lambda item: item[0])


# 기능: 한글 경로의 이미지를 BGR로 해독한다.
# 특징: EXIF 자동 회전을 끄고 빈 파일·해독 실패를 오류로 남긴다. 입력 파일을 쓰지 않는다.
def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    if not data.size:
        raise ValueError("error:이미지 파일이 비어 있음")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("error:이미지 파일 해독 실패")
    validate_image(image)
    return image


# 기능: 보정 결과를 손실 없는 PNG로 저장한다.
# 특징: 원래 확장자 뒤에 .png를 붙여 이름 충돌을 피하고 기존 출력은 덮어쓰지 않는다.
def save_png(path, image):
    validate_image(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    okay, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not okay:
        raise OSError("error:PNG 인코딩 실패")
    payload = encoded.tobytes()
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            if handle.write(payload) != len(payload):
                raise OSError("error:PNG 쓰기 실패")
        if path.stat().st_size != len(payload):
            raise OSError("error:PNG 파일 크기 불일치")
    except OSError:
        # 기능: 이번 호출이 만든 불완전한 PNG만 정리하고 원래 저장 오류를 유지한다.
        # 특징: xb 열기에 실패한 기존 파일은 삭제하지 않는다. 정리 오류가 원래 원인을 가리지 않게 한다.
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


# 기능: CSV의 열 순서를 고정하고 UTF-8 BOM으로 저장한다.
# 특징: None은 빈칸이며 x 모드로 기존 결과를 덮어쓰지 않는다.
def write_csv(path, rows, columns):
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# 기능: 결과 행을 데이터·조건·품질 그룹별로 집계한다.
# 특징: 성공·유효 값만 평균에 넣고 지표별 개수를 기록한다. 중첩 태그를 합쳐 전체 수로 사용하지 않는다.
def summarize_results(rows):
    buckets = defaultdict(list)
    for row in rows:
        # CSV에서 읽은 "False"는 빈 문자열이 아니므로 단순 bool 검사로 검토 완료 처리하면 안 된다.
        reviewed = str(row.get("reviewed", False)).strip().lower() in {"true", "1"}
        if row.get("group_source") == "auto" or reviewed:
            labels = row["tags"].split(";") if row["tags"] else ["unclassified"]
        else:
            # 기능: 이전 수동 실행 결과를 다시 그릴 때도 누락·미검토 그룹을 유지한다.
            labels = ["missing" if row["group_reason"] == "group_missing" else "unreviewed"]
        for group in ("all", *labels):
            buckets[(row["dataset"], row["condition"], group)].append(row)
    summary = []
    for (dataset, condition, group), members in sorted(buckets.items()):
        successful = [r for r in members if r["status"] == "success"]
        item = dict(dataset=dataset, condition=condition, group=group, attempted_n=len(members),
                    success_n=len(successful), skipped_n=sum(r["status"] == "skipped" for r in members),
                    failed_n=sum(r["status"] == "failed" for r in members))
        for name in (*METRIC_NAMES, "preprocessing_ms"):
            values = [float(r[name]) for r in successful if r.get(name) not in (None, "")
                      and math.isfinite(float(r[name]))]
            item[name + "_n"] = len(values)
            item["mean_" + name] = float(np.mean(values)) if values else None
        summary.append(item)
    return summary


# 기능: 그래프 라이브러리와 한글 글꼴을 실행 시작 시 확인한다.
# 특징: 사진을 모두 처리한 뒤 의존성 오류가 드러나는 일을 피하고 같은 준비 결과를 재사용한다.
def prepare_plots():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        raise ValueError("error:기본 그래프 생성에 matplotlib이 필요함") from None
    # 기능: 설치된 한글 글꼴을 선택해 그래프의 한글이 깨지지 않도록 한다.
    # 특징: Windows의 맑은 고딕을 우선 사용하고 다른 환경에서는 설치된 한글 글꼴을 확인한다.
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    korean_font = next((name for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic")
                        if name in fonts), None)
    if korean_font is None:
        raise ValueError("error:한글 그래프용 글꼴이 필요함(맑은 고딕 또는 나눔고딕)")
    return plt, korean_font


# 기능: summary.csv의 조건·그룹 평균으로 기본 비교 그래프 5장을 만든다.
# 특징: 막대 높이와 위 숫자는 같은 평균 지표값이다. 사진 수는 범례에 표시한다.
#       세로축은 0부터 시작하며 포화 비율만 %로 변환한다. 미측정 값은 0으로 바꾸지 않는다.
def create_plots(summary, folder, plot_context=None):
    if not summary or len({row["dataset"] for row in summary}) != 1:
        raise ValueError("error:그래프에는 한 데이터의 집계 결과가 필요함")
    plt, korean_font = prepare_plots() if plot_context is None else plot_context
    folder.mkdir(exist_ok=True)
    group_labels = {"all": "전체 사진", "blur": "흐림", "local_illumination": "부분별 밝기 불균일",
                    "normal": "정상", "missing": "태그 없음", "unreviewed": "태그 검토 전",
                    "unclassified": "분류 불가 또는 일부 미측정"}
    condition_labels = ["P0_reference\nROI·크기 조절만", "P1\n감마·가우시안", "P1+\n감마·CLAHE·흐림만 선명화·가우시안"]
    descriptions = {
        "gray_mean": ("사진의 평균 밝기 비교", "평균 밝기 (0~255)",
                      "값이 클수록 밝습니다. 높다고 항상 좋은 것은 아니며, 너무 밝아지는지도 확인하세요."),
        "block_mean_std_4x4": ("사진 안의 밝기 불균일 비교", "16개 구역의 평균 밝기 편차",
                               "값이 클수록 구역마다 밝기 차이가 큽니다. 감소하면 밝기가 더 균일해졌다는 뜻입니다."),
        "laplacian_variance": ("경계의 선명도 비교", "라플라시안 분산",
                               "값이 클수록 경계 변화가 강합니다. 노이즈·노면 질감도 값을 높이므로 함께 확인하세요."),
        "noise_sigma": ("사진의 노이즈 추정값 비교", "노이즈 추정값 (σ)",
                        "값이 작을수록 노이즈 추정량이 적습니다. 가는 균열도 지워졌는지 사진과 함께 확인하세요."),
        "saturation_ratio": ("완전히 검거나 흰 픽셀의 비율 비교", "포화 픽셀 비율 (%)",
                             "밝기가 0 또는 255인 픽셀의 비율입니다. 값이 클수록 밝기 정보가 소실됐을 가능성이 큽니다."),
    }
    index = {(r["group"], r["condition"]): r for r in summary}
    groups = [g for g in ("all", "blur", "local_illumination", "normal", "missing", "unreviewed", "unclassified")
              if any(r["group"] == g for r in summary)]
    if not groups:
        raise ValueError("error:그래프에 지원하는 품질 그룹이 없음")
    x, width = np.arange(len(CONDITIONS)), 0.8 / len(groups)
    for metric in PLOT_METRICS:
        # 기능: 포화 비율은 그래프에서만 백분율로 바꿔 숫자와 축 단위를 일치시킨다.
        # 특징: CSV의 0~1 비율은 유지한다. 다른 지표는 원래 단위 그대로 표시한다.
        display_scale = 100.0 if metric.endswith("_ratio") else 1.0
        heights = []
        # rc_context를 사용해 이 그래프에서만 한글 글꼴을 적용한다.
        with plt.rc_context({"font.family": korean_font, "axes.unicode_minus": False}):
            figure, axis = plt.subplots(figsize=(12, 6))
        try:
            for group_index, group in enumerate(groups):
                items = [index.get((group, condition), {}) for condition in CONDITIONS]
                values = [r.get("mean_" + metric) for r in items]
                values = [value * display_scale if value is not None else None for value in values]
                counts = [r.get(metric + "_n", 0) for r in items]
                sample_label = f"{counts[0]}장" if len(set(counts)) == 1 else f"P0:{counts[0]}, P1:{counts[1]}, P1+:{counts[2]}장"
                positions = x - 0.4 + width * (group_index + 0.5)
                axis.bar(positions, [v if v is not None else np.nan for v in values], width,
                         label=f"{group_labels[group]} ({sample_label})")
                for position, value in zip(positions, values):
                    if value is not None:
                        heights.append(value)
                        decimals = 3 if metric == "noise_sigma" else 2
                        value_label = f"{value:.{decimals}f}" + ("%" if metric.endswith("_ratio") else "")
                        axis.annotate(value_label, (position, value), xytext=(0, 3),
                                      textcoords="offset points", ha="center", fontsize=7, fontfamily=korean_font)
            if not any(r[metric + "_n"] for r in summary):
                axis.text(0.5, 0.5, "측정 가능한 결과가 없습니다", transform=axis.transAxes, ha="center", fontfamily=korean_font)
            for condition_index, condition in enumerate(CONDITIONS):
                if not any(index.get((group, condition), {}).get(metric + "_n", 0) for group in groups):
                    axis.text(condition_index, 0.03, "미실행 또는 미측정", transform=axis.get_xaxis_transform(),
                              ha="center", fontsize=8, color="dimgray", fontfamily=korean_font)
            title, ylabel, explanation = descriptions[metric]
            axis.set_xticks(x, condition_labels, fontfamily=korean_font, fontsize=9)
            axis.set_xlim(-0.6, len(CONDITIONS) - 0.4)
            axis.set_title(title, fontfamily=korean_font)
            axis.set_ylabel(ylabel, fontfamily=korean_font)
            # 기능: 세로축을 0에서 시작해 숫자의 비율과 막대 높이의 비율이 같도록 한다.
            # 특징: 위쪽 여백을 확보해 평균값 라벨이 잘리지 않게 한다. 모든 값이 0이어도 축을 유지한다.
            axis.set_ylim(0, max(heights) * 1.18 if heights and max(heights) > 0 else 1)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(title="품질 그룹 · 괄호는 측정 사진 수",
                        title_fontproperties={"family": korean_font, "size": 8},
                        prop={"family": korean_font, "size": 8})
            figure.text(0.5, 0.055, explanation, ha="center", fontsize=9, fontfamily=korean_font)
            figure.text(0.5, 0.015, "막대 높이와 위 숫자는 평균 지표값 · 범례 괄호는 측정 사진 수 · 빈 막대는 미실행 또는 미측정(0이 아님)",
                        ha="center", fontsize=8, fontfamily=korean_font)
            # 기능: 실제 그림을 그릴 때도 한글 글꼴과 일반 빼기 기호를 적용한다.
            # 특징: 맑은 고딕에 없는 유니코드 마이너스 때문에 경고가 발생하지 않도록 한다.
            with plt.rc_context({"font.family": korean_font, "axes.unicode_minus": False}):
                figure.tight_layout(rect=(0, 0.1, 1, 1))
                target = folder / (metric + ".png")
                figure.savefig(target, dpi=140)
            if not target.stat().st_size:
                raise OSError("error:그래프 저장 실패")
        finally:
            plt.close(figure)


# 기능: 한 폴더의 이미지들을 세 조건으로 처리하고 이미지·CSV·설정·그래프를 저장한다.
# 특징: 기하·Gamma를 이미지 안에서 재사용하며 개별 읽기·보정·저장 실패 후 다음 이미지를 계속 처리한다.
def run_experiments(input_dir=None, output_dir=None, limit=None, cfg=None, dataset=None):
    input_dir = Path(DEFAULT_INPUT if input_dir is None else input_dir).resolve()
    output_dir = Path(DEFAULT_OUTPUT if output_dir is None else output_dir).resolve()
    normalized = validate_config(cfg)
    thresholds = validate_thresholds()
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("error:limit은 1 이상의 정수")
    if output_dir.is_relative_to(input_dir):
        raise ValueError("error:출력 폴더는 입력 폴더 안에 둘 수 없음")
    is_rdd = input_dir == (PROJECT_ROOT / "img/RDD2020_train/train/img").resolve()
    dataset = ("RDD2020" if is_rdd else input_dir.name or "IMAGES") if dataset is None else dataset
    images = discover_images(input_dir, dataset)
    if not images:
        raise ValueError("error:입력 폴더에 지원 이미지가 없음")
    image_ids = {image_id for image_id, _, _ in images}
    if set(normalized["roi_overrides"]) - image_ids:
        raise ValueError("error:ROI 설정에 입력 폴더에 없는 image_id가 있음")
    selected = images if limit is None else images[:limit]
    plot_context = prepare_plots()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / datetime.now(timezone(timedelta(hours=9))).strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir()
    metadata = dict(program="A.py", input_dir=str(input_dir), dataset=dataset,
                    dataset_kind="rdd" if is_rdd else "images", discovered_n=len(images), selected_n=len(selected),
                    group_mode="auto", group_thresholds=thresholds,
                    group_measurement="P0_reference before Gamma/CLAHE/Unsharp/Gaussian",
                    config=normalized,
                    versions=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__),
                    baseline="P0_reference = ROI + Resize; original source files are unchanged",
                    time_policy="ROI/Resize + correction; reused Gamma counted per condition; excludes I/O and metrics")
    (run_dir / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    rows, template = [], []
    print(f"입력 {len(selected)}장 / 출력: {run_dir}", flush=True)
    for index, (image_id, input_path, relative) in enumerate(selected, 1):
        tags, group_reason, reference_values = ("unclassified",), "auto_measurement_failed", {}
        geometry, reference, gamma_cache, geometry_ms = {}, None, {}, None
        input_error, stage = "", "read"
        try:
            original = read_image(input_path)
            stage, start = "geometry", time.perf_counter()
            reference, geometry = geometry_preprocess(original, normalized, image_id)
            geometry_ms = (time.perf_counter() - start) * 1000
            del original
            # 기능: 보정 전 기준 지표로 한 번만 분류하고 모든 조건에서 같은 그룹을 사용한다.
            # 특징: P0_reference 지표를 재사용한다. 자동 분류를 사람의 검토 완료로 표시하지 않는다.
            stage = "classification"
            reference_values = measure_quality(reference)
            tags = classify_quality(reference_values, thresholds)
            group_reason = "auto_partial_measurement" if "unclassified" in tags else "auto_thresholds"
        except PROCESSING_ERRORS as exc:
            input_error = error_text(exc)
        template.append(dict(image_id=image_id, tags=";".join(tags), reviewed=0))
        for condition in CONDITIONS:
            row = {name: None for name in RESULT_COLUMNS}
            row.update(image_id=image_id, input_path=str(input_path), dataset=dataset,
                       dataset_kind=metadata["dataset_kind"], condition=condition, tags=";".join(tags),
                       reviewed=False, group_reason=group_reason, group_source="auto",
                       reference_laplacian_variance=reference_values.get("laplacian_variance"),
                       reference_block_mean_std_4x4=reference_values.get("block_mean_std_4x4"),
                       status="failed", reason="",
                       output_path="", stages_json="[]", metric_warnings="", **geometry)
            if input_error:
                row["reason"] = input_error
            else:
                try:
                    stage = "preprocess"
                    shared_gamma_ms = gamma_cache.get("ms", 0.0) if condition != "P0_reference" else 0.0
                    start = time.perf_counter()
                    corrected, params = preprocess_condition(reference, normalized, condition, tags, gamma_cache)
                    row["preprocessing_ms"] = geometry_ms + shared_gamma_ms + (time.perf_counter() - start) * 1000
                    row.update({key: params[key] for key in PARAMETER_COLUMNS})
                    row["stages_json"] = json.dumps(params["stages"])
                    stage = "metrics"
                    values = reference_values if condition == "P0_reference" else measure_quality(corrected)
                    row["metric_warnings"] = ";".join(f"{k}:image_too_small" for k, v in values.items() if v is None)
                    stage = "save"
                    condition_folder = "P1_plus" if condition == "P1+" else condition
                    target = run_dir / condition_folder / relative.parent / (relative.name + ".png")
                    save_png(target, corrected)
                    row.update(values)
                    row.update(status="success", output_path=str(target))
                except PROCESSING_ERRORS as exc:
                    row["reason"] = error_text(exc)
            # 기능: 실패 단계는 reason과 별개 열로 남겨 수치가 없는 이유를 바로 찾게 한다.
            # 특징: 저장까지 성공한 경우에만 지표값을 행에 넣어 실패 영상을 평균에 섞지 않는다.
            row["error_stage"] = stage if row["status"] == "failed" else "group" if row["status"] == "skipped" else ""
            rows.append(row)
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"[{index}/{len(selected)}] {image_id}", flush=True)
    write_csv(run_dir / "results.csv", rows, RESULT_COLUMNS + ["error_stage"])
    write_csv(run_dir / "groups_template.csv", template, ["image_id", "tags", "reviewed"])
    summary = summarize_results(rows)
    write_csv(run_dir / "summary.csv", summary, SUMMARY_COLUMNS)
    create_plots(summary, run_dir / "plots", plot_context)
    print(f"완료: 성공 {sum(r['status'] == 'success' for r in rows)}, "
          f"건너뜀 {sum(r['status'] == 'skipped' for r in rows)}, 실패 {sum(r['status'] == 'failed' for r in rows)}", flush=True)
    return run_dir


# 기능: 간단한 실행 옵션을 읽어 1차 실험을 시작한다.
# 특징: 이미지 경로는 코드 상단에서 변경한다. 실행 중단 오류는 error:원인과 종료 코드 1로 표시한다.
def main(argv=None):
    parser = argparse.ArgumentParser(description="개발 A 자동 품질 분류·전처리 비교; 입력 폴더는 A.py 상단에서 변경")
    parser.add_argument("--limit", type=int, help="정렬된 처음 N장만 처리")
    # 기능: 명령줄 형식 오류도 다른 오류와 같은 한 줄 형식으로 표시한다.
    # 특징: --help는 argparse의 표준 안내를 그대로 사용한다.
    def argument_error(message):
        parser.exit(2, error_text(message) + "\n")
    parser.error = argument_error
    args = parser.parse_args(argv)
    try:
        run_experiments(limit=args.limit)
        return 0
    except PROCESSING_ERRORS as exc:
        print(error_text(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
