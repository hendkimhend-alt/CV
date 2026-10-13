# 전처리 파트 (`src/preprocessing/`) — final0

> 이 문서는 원본 저장소의 `CV-main/src/preprocessing/README.md`를 final0 폴더에 맞게 고친 판이다.
> 전처리 알고리즘(①~⑮)과 수치는 원본과 같고, 입력 · 출력 · 실행 방법을 final0 실행기(`run_final.py` 등) 기준으로 바꿨다.
> 경로는 모두 final0 폴더 기준이다.

도로 영상에서 **도로 영역(road_mask)을 자동으로 찾고 검증한 뒤, 도로 안에서만 밝기를 보정**해 검출기에 넘긴다.
고전 영상처리만 쓴다 (AI · 딥러닝 없음). 설정 v1.1 (`configs/preprocessing.json`, `preproc-v1.1-provisional`).

- 모든 수치는 `configs/preprocessing.json` 값이다 (원본 `CV-main/configs/preprocessing.json`과 같은 파일). 대부분 **잠정값**이며, 최적값이라는 근거는 없다.
- 검출기 입력 보정 · 연결 수치는 `configs/detection_final.json`의 `corrections` · `integration` 값이다.
- 길이(px) 단위는 별도 표시가 없으면 **작업 해상도(긴 변 1024)** 기준이다.
- final0에는 전처리만 따로 돌리는 실행기가 없다. `run_final.py`의 `prepare()`가 사진마다 `process_image()`를 부르고, `run_final_PBL.py` · `run_final_test.py`도 같은 `prepare()`를 쓴다.
- 전처리 단독 실행기와 수동 마스크 편집기는 원본 저장소에만 있다 (`CV-main/src/preprocessing/run_preprocess.py` · `mask_editor.py` · `edit_road_mask.py`).

문서 순서: **1. 전체 흐름 → 2. 입력 → 3. 처리 → 4. 검출기 입력 만들기 · 보정 → 5. 출력 → 6. 성능 → 7. 파일별 기능 · 실행 순서**

---

## 1. 전체 흐름

```text
[입력]  run_final.py main() ────────────────────────────────────────────────────────────
 A. 설정 읽기 · 검사        configs/preprocessing.json → load_config (주석 키 제거 · 기본 값 검사)
                            configs/detection_final.json → corrections · integration 검사
 B. 사진 목록               --dataset (rdd_dev · provided · 폴더 …) → rdd_test 보호 → --limit
 ── 사진마다 반복 ──
 C. 영상 읽기               paths.imread → BGR uint8 H×W×3

[처리]  run_final.prepare() → pipeline.process_image() ───────────────────────────────────
 ① 작업 영상 (긴 변 1024)  ② 특징 지도  ③ 격자 특징  ④ 시드 고르기  ⑤ Region Growing
 ⑥ 마스크 정리  ⑦ 원본 크기 복원  ⑧ 검증 PASS/FAIL → 최종 road_mask 결정
    (FAIL이면 on_fail=use_candidate에 따라 후보 마스크로 process_image() 한 번 더)
 ⑨ analysis_mask  ⑩ 초기 품질  ⑪ 조건부 Gamma  ⑫ 노이즈 재측정  ⑬ 조건부 Gaussian(꺼 둠)
 ⑭ 최종 품질  ⑮ 출력 규약 검사

[검출기 입력]  ──────────────────────────────────────────────────────────────────────────
 D. 연결 (detector_adapter)        노면 외접 영역 + 16px 자르기 · 긴 변 1024
 E. 보정 (detector_corrections)    노면 선명도 → blur_rt / not_blur_rt → 보정 (노면 마스크 안만)
 → 균열 detect() · 포트홀 texture_potholes() → 도로 위 판정(박스의 50% 이상) → 원본 좌표

[출력]  ──────────────────────────────────────────────────────────────────────────────
 F. final-result/run_<시각>/       road_mask/ · road_view/ · preprocessed/ · images/ · results.csv · summary.csv · run_config.json
```

---

## 2. 입력

### A. 설정 읽기 · 검사

**전처리 설정 (`config.py`)**
1. `configs/preprocessing.json`을 읽고 `_`로 시작하는 키(설명 주석)를 지운다.
2. 자주 틀리는 값만 검사한다: 큰 항목(roi, gamma 등)이 빠짐, Gamma · Gaussian 모드 이름,
   Gaussian을 켰는데 σ ≤ 0, 커널 크기가 짝수. 그 밖의 값은 검사하지 않는다.
3. 최종 설정의 SHA-256 해시(`config_hash`)를 `run_config.json`의 `preprocessing_hash`에 남긴다 (같은 설정인지 확인용).

`apply_override`(키.경로=값 덮어쓰기)는 코드에 남아 있지만 final0 실행기에는 이를 넘기는 인자가 없다. 값을 바꾸려면 `configs/preprocessing.json`을 고친다.

final0에서 실제로 쓰는 설정 항목:

| 항목 | final0에서 |
|---|---|
| `roi` · `validation` · `analysis_mask` · `quality` · `gamma` · `gaussian` | 그대로 쓴다 (3절) |
| `manual_correction.allow_external_mask` | 후보 마스크 재실행(⑧)에 쓴다. `false`면 후보 마스크도 받지 않는다 |
| `manual_correction`의 나머지 (`review_pass_results` · `mask_threshold` · `min_area_ratio`) | 쓰지 않는다 (원본의 편집기 · 마스크 파일 읽기용) |
| `output` | 검사에서 항목이 있는지만 본다. 저장 위치 · 저장 여부는 실행기가 정한다 (5절) |

**검출 설정 (`road_detection/config.py`)** — `configs/detection_final.json`에서 전처리와 관련된 부분:

| 블록 | 값 | 뜻 |
|---|---|---|
| `integration.input_mode` | `crop_to_road` | 검출기 입력을 road_mask 외접 영역만 잘라 만든다 |
| `integration.crop_margin` | 16 | 외접 사각형 여유 (원본 화소) |
| `integration.long_side` | 1024 | 검출기 입력 긴 변 |
| `integration.min_road_overlap` | 0.5 | 후보 박스 면적 중 노면 비율이 이 값 이상이면 "도로 위" |
| `integration.on_fail` | `use_candidate` | 자동 마스크 FAIL일 때 검증 안 된 후보 마스크로 계속 (`skip`이면 그 사진 건너뜀) |
| `corrections` | 4절 E | 검출기 입력 보정. 모르는 단계 이름은 실행 전에 오류 (`detector_corrections.validate`) |

### B. 사진 목록

| 실행기 | 입력 |
|---|---|
| `run_final.py` | `--dataset`: `rdd_dev` · `rdd_test`는 `labels/split_rdd.csv`, `rdd_tune` · `rdd_val`은 `labels/split_rdd_dev.csv`로 고르고, `provided` · `captured`는 `data/` 아래 폴더, 그 밖은 폴더 경로. 폴더는 하위까지 확장자 `.jpg .jpeg .png .bmp .tif .tiff .webp` (`.`으로 시작하는 숨김 경로 제외), 정렬. `rdd` · `rdd_test`는 `--allow-rdd-test` 없이는 거부. `--limit N` |
| `run_final_PBL.py` | `data/PBL module 1 - dataset/` (또는 `--input <폴더>`), 같은 확장자 · 숨김 규칙 |
| `run_final_test.py` | `data/test/` (또는 `--input <파일 또는 폴더>`) |

`run_final.py`의 사진 이름(key)은 입력 폴더 기준 상대 경로(확장자 제외)를 `__`로 이은 것이다 (예: `sub__Japan_000015`).

### C. 영상 읽기 (`paths.imread`)

- 한글 경로 대응을 위해 `np.fromfile` + `cv2.imdecode`로 읽는다 (`IMREAD_COLOR` → 회색조 · 알파 채널 영상도 BGR 3채널로 읽힘).
- 읽기 실패는 `ValueError`가 난다. `run_final_PBL.py` · `run_final_test.py`는 그 사진만 건너뛰고, `run_final.py`는 잡지 않으므로 실행이 멈춘다.
- `process_image()` 안에서 형식을 검사한다: 비어 있지 않은 `uint8` `H×W×3` 배열이어야 한다. 자동 변환은 하지 않는다 (어기면 그 사진은 error).

---

## 3. 처리 (`pipeline.process_image`)

final0에서는 `run_final.prepare()`가 `process_image(image, pre_cfg, image_id=<파일 이름>)`로 부른다. 편집기(`editor`) · `review` 인자는 넘기지 않는다.

### ① 작업 영상 — `roi.work_long_side = 1024`

원본을 긴 변 1024로 줄여 ②~⑥을 계산한다 (INTER_AREA). 1024보다 작은 영상은 그대로 쓴다.

### ② 특징 지도 — `roi.features`

| 지도 | 계산 | 수치 |
|---|---|---|
| `lab` | Gaussian 평활 후 CIE L\*a\*b\* (float, L\* 0~100) | `smooth_sigma 2.0` |
| `gradient` | L\*를 평활한 뒤 Sobel 3×3 크기 ÷ 8 (단위: L\*/px) | `gradient_sigma 1.0` |
| `texture` | 평활 전 L\*의 \|Laplacian 3×3\|를 Gaussian 국소 평균 | `texture_sigma 2.0` |

모든 단계에서 쓰는 색 거리:

```text
d = √((w_L·ΔL*)² + Δa*² + Δb*²),   w_L = lightness_weight = 0.35
```

색(a\*b\*) 위주로 보고 밝기는 0.35배만 반영한다. 그림자는 밝기를 크게 바꾸지만 색은 덜 바꾸기 때문이다.

### ③ 격자 특징 — `roi.grid`

작업 영상을 `cell_size 32` × 32 칸으로 나눈다. 칸마다 다음을 잰다.
- 평균 · 표준편차 Lab, 중심 위치(0~1)
- 질감 평균, 기울기 평균
- 경계 밀도: `gradient > edge_threshold 6.0`인 화소 비율
- 채도 √(a\*²+b\*²), 8-이웃 칸과의 평균 색 거리, 칸 채움 비율

가장자리에서 잘린 칸은 채움 비율이 `min_cell_fill 0.5` 미만이면 시드 후보에서 뺀다.

### ④ 시드 고르기 — `roi.seeds`

칸마다 6개 성분(0~1)의 **가중 기하평균**으로 점수를 매긴다. 기하평균이라서 한 성분이라도 0에 가까우면 전체 점수가 낮아진다.

```text
S = exp( Σ w_i · ln(max(s_i, 1e-6)) / Σ w_i )
```

| 성분 | 식 | 수치 (가중치) | 의도 |
|---|---|---|---|
| position | cy^p · (1 − h·\|2cx − 1\|) | p 0.5 · h 0.3 (1.0) | 아래쪽 · 가운데 |
| neutral_color | exp(−(chroma / s_c)²) | s_c 18 (1.0) | 무채색 (아스팔트 · 콘크리트) |
| homogeneity | exp(−edge_density / s_e) | s_e 0.2 (1.0) | 강한 경계가 적음 (차량 · 건물 제외) |
| texture | [t_lo, t_hi] 안이면 1, 아래면 (t/t_lo)², 위면 t_hi/t | [2, 45] (1.0) | 너무 매끈(하늘 · 보닛) · 너무 거침(수목) 제외 |
| brightness | L\* ∈ [L_lo, L_hi]면 1, 밖이면 margin 동안 선형 감소 | [10, 80] · margin 15 (1.0) | 아주 밝은 칸 제외, 그림자 노면은 유지 |
| neighbor_similarity | 8-이웃 평균 [exp(−d/s_n) · 이웃의 예비 점수] | s_n 6 (**1.5**) | 비슷한 도로다운 칸에 둘러싸임 |

선택 규칙 (무작위 없음): 점수 내림차순으로 보면서 아래 조건을 모두 만족하는 칸을 최대 `count_max 5`개 고른다.
1. 점수 ≥ `min_score 0.6`
2. 이미 고른 시드와의 칸 거리(Chebyshev) ≥ `min_distance_cells 3`
3. 첫 시드(최고 점수)와 일관됨: 채도 거리 ≤ 5.0 · 밝기 차 \|ΔL\*\| ≤ 25 · 질감 비 ≤ 2.0

시드가 `count_min 3`개 미만이면 경고만 남긴다. 0개면 ⑧에서 FAIL이다.
첫 시드가 도로가 아니면 이후 시드도 틀릴 수 있다. 이것은 알려진 한계이며, 검증으로 보완한다 (final0에는 수동 확인 단계가 없다).

### ⑤ Region Growing — `roi.region_growing`

시드 칸의 평균 Lab, 색 편차 σ_ab, 질감 중앙값을 기준으로 삼아, 화소 p가 아래 조건을 **모두** 만족하면 받아들인다.

| 조건 | 식 | 수치 |
|---|---|---|
| 색 | d(p) ≤ T = min(6.0 + 1.0·σ_ab, 9.0) | `color_threshold 6.0` · `seed_std_scale 1.0` · `max_color_threshold 9.0` |
| 밝기 차 상한 | \|ΔL\*\| ≤ 30 | `max_lightness_diff 30` (흰 차선 · 하늘 차단) |
| 국소 기울기 | gradient ≤ 5.0 | `max_gradient 5.0` (연석 · 차량 윤곽에서 멈춤) |
| 질감 비 | texture / 기준 ∈ [0.35, 2.5] | `texture_ratio_range` |
| 연결 | 시드에서 받아들인 화소만 거쳐 8-이웃으로 이어짐 | `connectivity 8` |

구현은 "조건을 만족하는 화소의 연결 요소 중 시드 칸을 포함하는 것"을 고르는 방식이며, 큐 BFS와 결과가 같다.
시드 칸 면적의 `min_seed_cell_coverage 0.3` 미만만 덮으면 그 시드는 자라지 못한 것으로 기록한다. 시드 영역들의 **합집합**이 결과다.

그림자: 색이 같다면 밝기 차이가 약 17~26(L\*) 이내일 때 받아들여진다 (색 거리 기준 6~9를 0.35로 나눈 값).
이보다 진한 그림자, 그리고 경계가 날카로워 기울기 상한에 막히는 그림자는 빠진다. 단, 그림자 안에 따로 시드가 있으면 포함된다.

### ⑥ 마스크 정리 — `roi.refine`

| 순서 | 연산 | 수치 |
|---|---|---|
| 1 | closing (타원) — 균열 · 틈으로 끊긴 노면 잇기 | `close_kernel 7` |
| 2 | opening (타원) — 가는 돌기 · 점 잡음 지우기 | `open_kernel 5` |
| 3 | 구멍 채우기 — 테두리에 닿지 않는 배경 중 영상 면적의 2% 이하만 | `fill_holes_max_area_ratio 0.02` |
| 4 | 작은 연결 요소 삭제 — 영상 면적의 1% 미만 | `min_component_area_ratio 0.01` |

가장 큰 요소 하나만 남기지 않는다 (중앙분리대로 나뉜 도로를 보존하기 위함). 노면 위 차량 같은 큰 구멍은 남겨 두고 검증에서 본다.

### ⑦ 원본 크기 복원

작업 해상도 마스크를 float로 원본 크기까지 선형 보간하고, `upsample_threshold 0.5` 이상을 도로(255)로 본다 → 자동 road_mask (`auto_mask`).

### ⑧ 검증과 최종 road_mask 결정 — `validation`

| 지표 | 정의 | 기준 |
|---|---|---|
| mask_area_ratio | 도로 화소 / 영상 화소 | ≥ 0.10 |
| largest_component_ratio | 가장 큰 8-연결 요소 / 도로 화소 | ≥ 0.80 |
| hole_ratio | 구멍 면적 / 구멍 채운 면적 | ≤ 0.15 |
| seed_consistency | 최종 마스크 안의 시드 수 / 시드 수 | ≥ 0.60 |

하나라도 못 넘으면 **FAIL**이다 (상태는 PASS/FAIL 둘뿐). 시드 0개나 빈 마스크도 FAIL이고, 정의되지 않은 지표도 FAIL로 처리한다(`undefined_metric_policy fail`). FAIL 사유는 모두 metadata에 기록한다.
이 PASS/FAIL이 `results.csv`의 `mask_check` 열이 된다 (첫 번째 `process_image()` 호출의 값).

`process_image()` 안에서 최종 road_mask는 다음 순서로 정한다 (`_choose_final_mask`).
1. `manual_mask` 인자가 있으면 (`allow_external_mask true`일 때) **검증 결과와 관계없이** 그것을 쓴다 → `final_mask.source = manual`
2. 없으면 **자동 마스크** — PASS일 때만 → `source = auto`
3. 둘 다 아니면(FAIL이고 `manual_mask` 없음) 전처리는 **`manual_required`로 여기서 멈춘다**.

**final0의 FAIL 처리 (`run_final.prepare`)** — `manual_required`이고 `integration.on_fail = use_candidate`이며 자동 후보 마스크가 비어 있지 않으면:
- `process_image(image, pre_cfg, image_id=…, manual_mask=<자동 후보 마스크>, manual_source="auto_candidate_unverified")`를 **한 번 더** 실행한다.
  ①~⑧(자동 마스크 · 검증)은 다시 계산되어 기록되고, 최종 마스크는 1번 규칙으로 후보 마스크가 된다
  (metadata: `manual_correction.source = external_file`, `path = auto_candidate_unverified`, `final_mask.source = manual`).
- `results.csv`의 `unverified_mask = True`로 표시한다.
- `on_fail = skip`이거나 후보 마스크가 비어 있으면 그 사진은 `skipped_manual_required`로 기록하고 검출하지 않는다.

FAIL 마스크를 영상 전체나 고정 사다리꼴로 몰래 바꾸지 않는다. 최종 마스크가 비어 있으면 error다.

### ⑨ analysis_mask — `analysis_mask`

road_mask **사본**을 `rect 5×5`로 1회 침식한다. 경계의 불확실한 화소를 품질 측정에서 빼기 위함이며, road_mask 자체는 그대로 둔다. 영상 테두리는 침식하지 않는다.
남은 화소가 `min_pixels 2000` 미만이거나 road_mask의 `min_ratio_of_road 0.3` 미만이면 측정이 불충분하다고 보고, 조건부 보정을 하지 않는다 (경고 `analysis_mask_too_small`).

### ⑩ 품질 측정 — `quality`

긴 변 `measurement_long_side 1024` 사본에서 analysis_mask 안 화소만 집계한다 (영상은 INTER_AREA/CUBIC, 마스크는 NEAREST로 크기 변경. 출력 영상에는 영향 없음).
회색조 = 0.299R + 0.587G + 0.114B (0~255).

| 지표 | 정의 |
|---|---|
| gray_mean · gray_std | 평균 · 모집단 표준편차 |
| dark_ratio · bright_ratio | 회색조 < 40 · > 215 화소 비율 |
| saturation_ratio | 0 또는 255 화소 비율 |
| block_mean_std_4x4 | 마스크 외접 영역을 4×4로 나눈 블록 평균들의 표준편차 (블록당 50화소 이상) |
| laplacian_variance | 4-이웃 Laplacian의 분산 (선명도) |
| noise_sigma | Immerkær 추정 σ = √(π/2) · Σ\|R\| / (6N), R = [[1,−2,1],[−2,4,−2],[1,−2,1]] 응답 |

노이즈 · 선명도는 필터 창이 도로 화소만 보도록, road_mask를 3×3 침식한 영역에서만 집계한다.
`noise_sigma`는 노면 질감과 균열 같은 실제 고주파 구조도 노이즈로 센다. 값이 크다고 해서 꼭 없애야 할 잡음이라는 뜻은 아니다.

이 품질 지표는 metadata에만 남고, final0의 `results.csv`에는 들어가지 않는다. `results.csv`의 `sharpness`는 4절 E의 별도 측정값이다.

### ⑪ 조건부 Gamma — `gamma`

```text
I_out = round(255 · (I_in / 255)^γ)     BGR 각 채널, 256칸 LUT, road_mask 안 화소만 (γ < 1 → 밝아짐)
```

1. **적용 조건** (`logic AND`): gray_mean < 80 **그리고** dark_ratio > 0.07
2. **γ 고르기** (`selection adaptive`): 후보 `[1.0, 0.9, 0.8, 0.7]` 중 1 미만인 값을 약한 것부터(0.9 → 0.8 → 0.7) 적용하고 품질을 다시 잰다.
   - 두 조건이 모두 풀리면(gray_mean ≥ 80, dark_ratio ≤ 0.07) 그 **첫 γ**를 쓴다.
   - 끝까지 안 풀리면 포화 방지를 통과한 가장 강한 γ를 쓴다.
3. **포화 방지**: 보정 후 saturation_ratio가 0.02를 넘고 보정 전보다 커졌으면 그 γ를 버린다. 그보다 강한 후보는 보지 않는다. 모든 후보가 걸리면 보정하지 않는다.

`selection fixed`로 바꾸면 `value 0.8` 하나만 쓴다 (v1.0 방식). 시도한 γ와 그 결과는 metadata `gamma.tried`에 남는다.
적용 여부(`gamma.applied`)는 `results.csv`의 `gamma` 열(True/False)로 나온다.

### ⑫ 노이즈 재측정

Gamma를 적용했으면 그 영상에서 품질을 다시 재고(`after_gamma`), 적용하지 않았으면 초기값을 그대로 쓴다. Gaussian 판단은 이 값으로 한다.

### ⑬ 조건부 Gaussian — `gaussian` (v1.1: `mode off`)

켜면(`conditional`) noise_sigma ≥ 0.6일 때 3×3, σ 0.8 Gaussian을 도로 안에만 적용한다.
마스크 인식 방식 out = G(I·M) / G(M)을 써서 경계에서 비도로 화소 값이 섞이지 않게 한다.
**꺼 둔 이유**: 노면 질감 때문에 사진의 56%에 적용됐고, 적용된 사진에서 균열 Recall이 0.117 떨어졌다 (95% 구간이 0을 포함하지 않음).

이 단계는 4절 E의 검출기 입력 보정 Gaussian(σ 0.8, 선명한 사진만)과 다른 단계다. 그쪽은 켜져 있다.

### ⑭ 최종 품질 · ⑮ 출력 규약 검사

최종 영상의 품질을 재고 초기값과의 차이를 계산한다. 그다음 아래 규약을 검사하며, 어기면 error다.
- 출력 영상 크기 = 입력 크기, road_mask 크기 = 입력 H×W
- **road_mask 밖 화소가 입력과 완전히 같음**

처리 중 어느 단계에서든 예외가 나면 그 사진만 `status = error`(실패한 단계 · 원인을 metadata에 기록)로 끝난다.
`run_final.py`는 이를 `skipped_error`로 기록하고 검출 없이 다음 사진으로 넘어간다.

`process_image()`의 결과 중 final0이 쓰는 것: `status` · `processed_image`(Gamma까지 반영된 원본 크기 영상) · `road_mask` · `auto_mask` · `metadata`의 `validation.status`와 `gamma.applied`.

---

## 4. 검출기 입력 만들기 · 보정

전처리가 `success`이면 `run_final.prepare()`가 D를, `run_final.detect_image()`가 E와 검출을 한다.

### D. 연결 — `detector_adapter.to_detector_input`

1. road_mask의 외접 사각형에 `crop_margin 16`px(원본 화소)을 더해 잘라 낸다 (영상 경계 안으로 제한).
2. 잘라 낸 영역의 긴 변을 `long_side 1024`로 맞춘다. 영상은 축소 INTER_AREA · 확대 INTER_CUBIC, 마스크는 NEAREST. 크기가 이미 같으면 복사만 한다.
3. 잘라 낸 영역 안의 비도로 화소는 검정으로 칠하지 않는다 (인공 경계가 생기지 않게).
4. 자른 위치 · 배율을 geometry(`crop_x` · `crop_y` · `scale_x` · `scale_y` 등)로 돌려준다. 검출 박스는 `to_original_bbox()`로 원본 좌표로 되돌린다.

입력은 `processed_image`(전처리 결과)와 최종 road_mask다. 이후 "검출기 마스크"는 이렇게 자르고 크기를 바꾼 road_mask를 말한다.

### E. 검출기 입력 보정 — `detector_corrections.py` (`configs/detection_final.json`의 `corrections`)

| 단계 | 설정 | 적용 대상 (`apply_to`) |
|---|---|---|
| `gaussian` | `ksize 3` · `sigma 0.8` · `blur_thr 2.76` | `not_blur_rt` (선명한 사진) |
| `line_close` | `length 7` · `angles 12` | `not_blur_rt` |
| `unsharp` | `amount 3.0` · `sigma 3.0` | `blur_rt` (아주 흐린 사진) |
| `clahe` | `clip 1.0` · `tile 8` | (지정 없음 = `all`) |
| `order` | `gaussian → line_close → unsharp → clahe` | |

**1) 노면 선명도 (`road_sharpness`)**
- 검출기 입력을 "원본 전체의 긴 변이 1024가 되는 배율"로 다시 맞춘다 (배율 = (1024 / 원본 긴 변) ÷ `scale_x`, 축소 INTER_AREA · 확대 INTER_LINEAR).
- 회색조로 바꾸고, 검출기 마스크(NEAREST로 크기 변경)를 5×5로 침식한 영역 안에서 Laplacian(`ksize=1`) 값의 분산을 잰다.
- 침식한 영역이 100화소 이하이면 영상 전체의 분산을 쓴다.

**2) 흐림 판단 (`judge_tags`)**
- `apply_to`가 `blur_rt` / `not_blur_rt`인 단계가 있으면, 단계 중 처음 찾은 `blur_thr`(2.76, 없으면 기본 50)를 기준으로
  선명도 < 2.76 → `blur_rt`, 아니면 `not_blur_rt` 태그를 붙인다.
- `run_final.py`는 `labels/split_rdd.csv`의 group(`blur` · `local_illumination` · `normal`)도 태그로 넘기지만, 현재 설정의 `apply_to`와 이름이 겹치지 않아 보정에는 영향이 없다. `run_final_PBL.py` · `run_final_test.py`는 빈 태그를 넘긴다.

**3) 보정 (`apply_corrections`)** — `order` 순서로, `apply_to`가 `all`이거나 태그에 있는 단계만 실행한다.

| 사진 | 실제 적용 순서 |
|---|---|
| 선명 (`not_blur_rt`, 선명도 ≥ 2.76) | Gaussian 3×3 σ0.8 → 막대 닫힘 7px · 12방향 → CLAHE 1.0 |
| 아주 흐림 (`blur_rt`, 선명도 < 2.76) | 언샤프 3.0 · σ3 → CLAHE 1.0 |

| 단계 | 계산 |
|---|---|
| gaussian | `cv2.GaussianBlur` 3×3 · σ 0.8 (BGR 3채널, `BORDER_REFLECT_101`) |
| line_close | 길이 7 선 모양 커널을 180°·i/12 (i = 0~11, 15° 간격) 방향마다 closing(`BORDER_REPLICATE`)하고 화소별 **최솟값**을 취한다 |
| unsharp | out = src + 3.0·(src − G_σ3(src)), 0~255로 자르고 반올림 (float 계산) |
| clahe | Lab으로 바꿔 L 채널에만 CLAHE (clip 1.0, 8×8 타일) → BGR |

- 각 필터는 잘라 낸 검출기 입력 전체에 계산한 뒤, 마지막에 **검출기 마스크 안 화소만** 보정 결과로 바꾸고 밖은 보정 전 값을 그대로 둔다.
  필터 창이 마스크 밖 화소를 볼 수 있다는 점에서 ⑬의 마스크 인식 Gaussian과 다르다.
- 한 단계라도 실행되면 `corrections_applied = True`다. 현재 설정은 CLAHE가 모든 사진에 걸리므로 검출까지 간 사진은 모두 True다.
- 보정된 영상에서 균열(`detect()`)과 포트홀(`texture_potholes()`)을 따로 찾고, 박스 면적의 50% 이상이 검출기 마스크 안이면 남긴 뒤 원본 좌표로 되돌린다.

---

## 5. 출력

final0은 `process_image()`의 metadata · overlay · processed_image.png 같은 전처리 전용 파일을 따로 저장하지 않는다 (`pipeline.save_result()` · `make_overlay()`는 코드에 남아 있지만 부르지 않는다).
전처리 결과는 아래 위치에 나온다.

### 상태 (`results.csv`의 `status`)

| 상태 | 뜻 |
|---|---|
| `detected` | 전처리 `success` → 연결 · 보정 · 검출까지 진행 |
| `skipped_manual_required` | 자동 마스크 FAIL이고 후보 마스크를 쓸 수 없음 (`on_fail skip` 또는 빈 후보 마스크) |
| `skipped_error` | 전처리 중 오류 (형식 · 처리 · 규약 검사 실패) |

### F. `run_final.py` — `final-result/run_<시각>/`

폴더 이름은 한국 시간이며, 같은 초에 겹치면 `_2`, `_3` …을 붙인다 (`--out`으로 상위 폴더 변경).

| 경로 | 내용 | 저장 조건 |
|---|---|---|
| `road_mask/<key>.png` | 최종 road_mask. 원본 크기, 도로 255 / 비도로 0 | 최종 road_mask가 있을 때 |
| `road_view/<key>.jpg` | 원본에서 노면 밖 화소를 0.3배로 어둡게 한 그림 | 최종 road_mask가 있을 때 |
| `preprocessed/<key>.jpg` | **검출기에 실제로 들어간 영상** = 전처리(Gamma) → 노면 영역 자르기 · 긴 변 1024 → E 보정. 원본 크기가 아님 | `detected`일 때 |
| `images/<key>.jpg` | 결과 그림: 노면(초록 반투명 · 노랑 외곽) · 균열(빨강) · 포트홀(파랑) · 정답(흰색, 있을 때) · 상태 문구. 긴 변 최대 1280 | 모든 사진 |

`--no-images`를 주면 위 그림을 하나도 저장하지 않는다 (CSV · JSON만).

**`results.csv` (사진당 한 줄, UTF-8 BOM) — 전처리 · 보정 관련 열**

| 열 | 뜻 |
|---|---|
| `status` | 위 상태 표 |
| `mask_check` | 첫 `process_image()`의 자동 마스크 검증 `PASS` / `FAIL` (검증 전에 오류가 나면 빈 값) |
| `unverified_mask` | 검증 안 된 후보 마스크로 진행했으면 True (⑧) |
| `road_ratio` | 최종 road_mask 화소 수 ÷ 원본 화소 수 (소수 4자리). road_mask가 없으면 빈 값 |
| `sharpness` | 노면 선명도 (E-1, 보정 **전** 검출기 입력, 소수 3자리). 검출까지 못 가면 빈 값 |
| `blur_judge` | `blur_rt` / `not_blur_rt` (E-2). 검출까지 못 가면 빈 값 |
| `gamma` | ⑪ Gamma 적용 여부 (True/False) |
| `corrections_applied` | E 보정 적용 여부 |
| `det_scale` | 검출기 입력 배율 (`scale_x`, 소수 4자리) |
| `pre_ms` | 읽기 + 전처리 + 연결(D) 시간 |
| `det_ms` | 보정(E) + 검출 + 도로 위 판정 시간 |
| `edges` · `keypoints` · `road_px_det` | 보정 **후** 검출기 입력의 검출기 마스크 안 Canny(100, 200) 에지 수 · SIFT 특징점 수 · 노면 화소 수 |

그 밖의 열(이미지 · 폴더 · 그룹 · 검출 수 · 정확도)은 `final0/README.md`를 본다.

**`summary.csv`** (전체 · 그룹 · 폴더별 한 줄) 중 전처리 관련: `n_detected` · `n_unverified_mask` · `n_blur_rt` · `road_ratio_mean`(검출까지 간 사진 평균) · `pre_ms_per_img`.

**`run_config.json`**: `preprocessing_config`(파일 이름) · `preprocessing_hash`(A의 설정 해시) · `corrections` · `integration`.

화면에는 끝에 상태별 장수, 검증 안 된 후보 마스크 장수, 아주 흐림(`blur_rt`) 장수를 출력한다.

### `run_final_PBL.py` — `final-result-PBL/run_<시각>/`

사진마다 다섯 장 중 전처리 관련 그림:

| 파일 | 내용 |
|---|---|
| `<이름>_2_road_mask.jpg` | 원본 위에 노면 마스크만 초록 칠 · 노란 테두리 (박스 없음). 마스크가 없으면 실패 상태 문구 |
| `<이름>_3_preprocessed.jpg` | 검출기에 실제로 들어간 영상 (`run_final.py`의 `preprocessed/`와 같은 것). 전처리 실패면 원본에 실패 상태를 적은 그림 |

CSV는 저장하지 않는다. 화면 로그에 사진마다 상태 · 흐림 판단 · 검증 안 된 마스크 여부를 출력한다.

### `run_final_test.py` — `final-result-test/run_<시각>/`

원본 위에 검출 박스만 그린 `<이름>_bbox.jpg` 한 장만 저장한다. 노면 마스크 · 전처리 영상은 저장하지 않는다.

---

## 6. 측정된 성능 (원본 저장소 기록, 2026-10-10, RDD2020 dev)

> 아래 값은 원본 저장소에서 전처리 단독 실행기와 당시 검출기(D1v, 검출기 입력 보정 없음)로 잰 것이다.
> final0의 최종 구성(검출기 D1hv + 4절 E 보정)으로 다시 잰 값이 아니다. rdd_test(241장)로는 재지 않았다.

### 도로 마스크 정확도 (사람이 그린 정답 마스크 36장)

| 방법 | 평균 IoU [95%] | IoU ≥ 0.5인 사진 |
|---|---|---:|
| 영상 전체 | 0.398 | - |
| 고정 사다리꼴 ROI (이전 방식) | 0.430 [0.38, 0.49] | 31% |
| **자동 Road Mask** | **0.585 [0.50, 0.67]** | **72%** |

- 오차: 실제 도로의 약 27%를 놓치고(Recall 0.726), 마스크의 약 25%가 도로가 아니다(Precision 0.745). 영상 면적 기준으로는 놓침 10.8%, 잘못 포함 5.3%.
- 출처별 IoU: China_Drone 0.85 · China_MotorBike 0.68 · Japan 0.56 · Czech 0.54 · United_States 0.54 · India 0.49 · Norway 0.43.
- 주된 오류: 그림자 · 차선 · 도색 너머 노면을 놓치고, 그늘진 식생 · 흙 · 차량 보닛을 도로로 잡는다.
- 36장 중 17장은 일부러 고른 어려운 사례다. 무작위 19장만 보면 IoU 0.597이다.

### 검증 (PASS/FAIL)

- dev 563장 중 PASS 483장 (86%) · FAIL 80장 (14%).
- 정답 마스크 36장 기준 PASS의 83%가 IoU ≥ 0.5였다. 그런데 **FAIL의 61%도 IoU ≥ 0.5**였다. 즉 FAIL 판정이 쓸 만한 마스크를 많이 버린다.
  그래서 final0도 FAIL이면 후보 마스크로 검출한다 (`on_fail use_candidate`, ⑧).

### 손상 덮기 (dev 563장, 정답 균열 박스 744개)

- 정답 균열 박스의 **88%**(654개)가 road_mask 위에 있다. Norway만 66%로 낮다.

### Gamma (dev, v1.1)

- 적용 112장 (20%). 고른 γ: 0.9 2장 · 0.8 22장 · 0.7 88장.
- 도로 밝기 중앙값 54.5 → 84.1. 두 조건이 모두 풀린 비율 **55%** (v1.0의 γ 0.8 고정은 21%). 포화 한도를 새로 넘긴 경우는 없다.

### 검출 성능에 대한 기여 (dev 563장, 검출기 D1v, 균열 in50, 같은 사진끼리 비교)

| 전처리 | Precision | Recall | F1 | 가짜/장 |
|---|---:|---:|---:|---:|
| 전처리 없음 (영상 전체) | 0.062 | 0.276 | 0.101 | 5.49 |
| 고정 사다리꼴 ROI | 0.082 | 0.333 | 0.132 | 4.93 |
| v1.0 (γ 0.8 고정 · Gaussian 조건부 · FAIL 건너뜀) | 0.187 | 0.273 | 0.222 | 1.57 |
| **v1.1 (final0과 같은 전처리 설정)** | 0.179 | **0.371** | 0.241 | 2.26 |
| 참고: 자동 마스크만 (Gamma 끔) | 0.194 | 0.363 | 0.253 | 1.99 |

- 전처리 없음 → v1.1: F1 +0.140 · Recall +0.095 · 가짜 −3.24/장 (모두 95% 구간이 0을 넘음).
- v1.0 → v1.1: Recall +0.098 (tune 394장 +0.100 · val 169장 +0.095), Precision은 차이 없음, 가짜 +0.69/장.
- 적응형 Gamma 자체는 검출 성능을 높이지 않는다 (끈 경우보다 F1 −0.012, Precision −0.016).
- 사람이 그린 정답 마스크를 넣어도 맞힌 균열 수가 같았다 (36장에서 24개 → 24개). ROI를 더 다듬어도 검출 이득은 작다.
- v1.1 변경은 dev 진단으로 정했으므로 완전히 독립된 확인은 아니다.

final0 최종 구성(전처리 + 검출기 입력 보정 + 검출기 D1hv)의 균열 성능은 `configs/detection_final.json`의 `description`에 적힌 값(개발 563장 R 0.587 · P 0.355 · F1 0.442 · 가짜 6.4개/장)을 참고한다.

### 처리 시간 (원본 개발 PC)

- 전처리 한 장 중앙값 약 200 ms (600×600 RDD 기준). 큰 영상(Norway 3650×2044 등)은 약 1~1.2 s.
- 검출은 약 90 ms/장 (긴 변 1024, 당시 검출기).
- final0 실행 시간은 `results.csv`의 `pre_ms` · `det_ms` · `total_ms`로 확인한다.

---

## 7. 파일별 기능 · 실행 순서

### 7-1. 각 py 파일이 하는 일

실행 순서대로 나열했다. "단계"는 1절 흐름도의 기호(A~F, ①~⑮)다.

| 순서 | 파일 | 단계 | 하는 일 | 주요 함수 | 누가 부르나 |
|---:|---|---|---|---|---|
| 0 | `run_final.py` (final0 맨 위) | A~F | **실행 시작점.** 설정 · 사진 목록, 사진마다 `prepare()`(전처리 + 연결) → `detect_image()`(보정 + 검출), 결과 저장 | `main` · `prepare` · `detect_image` · `new_run_dir` | 사용자 (명령줄) · `run_final_PBL.py` · `run_final_test.py` |
| 1 | `config.py` | A | 설정 JSON 읽기, `_` 주석 키 제거, 기본 값 검사, 설정 해시 | `load_config` · `check_config` · `config_hash` (`apply_override`는 final0에서 안 씀) | run_final · run_final_PBL · run_final_test · pipeline |
| 2 | `image_io.py` | ① · ⑧ · D | 영상 · 마스크 형식 검사, 0/255 이진화 | `validate_image` · `validate_mask` · `to_binary_mask` (`read_image` · `read_mask` · `write_image`는 final0 실행기에서 안 씀 — 읽기 · 쓰기는 `src/paths.py`의 `imread` · `imwrite`) | pipeline · detector_adapter |
| 3 | `pipeline.py` | ①~⑮ | **한 장 처리의 중심.** 아래 4~13번 파일을 순서대로 부르고 metadata를 채움 | `process_image` · `_choose_final_mask` · `_apply_gamma_step` (`save_result` · `make_overlay`는 final0에서 안 씀) | run_final.prepare |
| 4 | `road_mask.py` | ①~⑦ | 자동 Road Mask 전체 묶음. 작업 영상 축소 → 5~8번 호출 → 원본 크기 복원 | `extract_road_mask` · `work_image` · `upsample_mask` | pipeline |
| 5 | `grid_features.py` | ② ③ | Lab · gradient · texture 지도, 32px 격자 특징, 색 거리 식 | `compute_feature_maps` · `compute_grid_features` · `color_distance` | road_mask · seeds |
| 6 | `seeds.py` | ④ | 칸마다 6개 성분 점수 → 가중 기하평균 → 시드 3~5개 선택 | `score_cells` · `select_seeds` | road_mask |
| 7 | `region_growing.py` | ⑤ | 시드 기준값(Lab · σ_ab · 질감)으로 조건 만족 화소를 연결해 넓힘 | `grow_regions` · `acceptance_mask` | road_mask |
| 8 | `mask_refine.py` | ⑥ | closing · opening · 구멍 채우기 · 작은 조각 제거 | `refine_mask` · `fill_holes` | road_mask · mask_validation |
| 9 | `mask_validation.py` | ⑧ | 면적 · 최대 요소 · 구멍 · 시드 일관성 측정 → PASS/FAIL | `compute_mask_metrics` · `validate_road_mask` | pipeline |
| 10 | `analysis_mask.py` | ⑨ | road_mask 사본을 5×5 침식, 측정 충분 여부 판단 | `make_analysis_mask` | pipeline |
| 11 | `quality.py` | ⑩ ⑫ ⑭ | 긴 변 1024 사본에서 밝기 · 어두움 · 포화 · 선명도 · 노이즈 지표 8개 측정 | `measure_in_masks` · `measure_quality` | pipeline (Gamma 후보마다 다시 호출) |
| 12 | `gamma.py` | ⑪ | 조건 판단 → γ 0.9 / 0.8 / 0.7 차례로 시도 → 포화 방지 | `decide_gamma` · `choose_gamma` · `apply_gamma` · `guard_triggered` | pipeline |
| 13 | `gaussian.py` | ⑬ | 노이즈 조건 판단 · 마스크 인식 Gaussian (현재 `off`라 판단만 하고 적용 안 함) | `decide_gaussian` · `masked_gaussian` | pipeline |
| 14 | `detector_adapter.py` | D | 검출기 입력 만들기(도로 영역 자르기 · 긴 변 1024), 박스를 원본 좌표로 되돌림 | `to_detector_input` · `road_bbox` · `to_original_bbox` | run_final.prepare · run_final.detect_image |
| 15 | `detector_corrections.py` | E | 보정 설정 검사, 노면 선명도 · 흐림 판단, 노면 마스크 안 보정 (`src/detect.py`의 `_line_kernel` 사용) | `validate` · `road_sharpness` · `judge_tags` · `apply_corrections` | `road_detection/config.py`(validate) · run_final.detect_image · run_final.main(`road_sharpness`) |
| — | `__init__.py` | — | 패키지 표시 (한 줄) | — | — |

### 7-2. 호출 순서 — `run_final.py`

```text
run_final.py  main()
 ├─ 1 config.load_config(configs/preprocessing.json)                A. 설정 읽기 · 검사
 │    road_detection.config.load_detection_config(detection_final.json)
 │      └─ detector_corrections.validate()                          corrections 단계 이름 검사
 ├─ 2 data.list_images() · --limit · rdd_test 보호                   B. 사진 목록
 ├─ 3 new_run_dir() → final-result/run_<시각>/
 │
 │  ── 사진마다 반복 ──
 ├─ 4 paths.imread()                                                C. 영상 읽기
 ├─ 5 run_final.prepare()
 │    ├─ pipeline.process_image()                                   ①~⑮ 한 장 처리
 │    │    ├─ image_io.validate_image()                             입력 형식 검사
 │    │    ├─ road_mask.extract_road_mask()                         ①~⑦ 자동 Road Mask
 │    │    │    ├─ road_mask.work_image()                           ① 긴 변 1024로 축소
 │    │    │    ├─ grid_features.compute_feature_maps()             ② Lab · gradient · texture
 │    │    │    ├─ grid_features.compute_grid_features()            ③ 32px 격자 특징
 │    │    │    ├─ seeds.score_cells() → seeds.select_seeds()       ④ 시드 점수 · 선택
 │    │    │    ├─ region_growing.grow_regions()                    ⑤ 시드별 영역 확장
 │    │    │    ├─ mask_refine.refine_mask()                        ⑥ 모폴로지 정리
 │    │    │    └─ road_mask.upsample_mask()                        ⑦ 원본 크기 복원
 │    │    ├─ mask_validation.compute_mask_metrics()
 │    │    │   → mask_validation.validate_road_mask()               ⑧ PASS / FAIL
 │    │    ├─ pipeline._choose_final_mask()                         ⑧ 최종 마스크: manual_mask 인자 → 자동(PASS)
 │    │    │    ※ 최종 마스크가 없으면 여기서 manual_required로 끝
 │    │    ├─ analysis_mask.make_analysis_mask()                    ⑨ 5×5 침식
 │    │    ├─ quality.measure_in_masks()                            ⑩ 초기 품질
 │    │    ├─ pipeline._apply_gamma_step()                          ⑪ ⑫
 │    │    │    ├─ gamma.decide_gamma()                             조건 판단 (gray_mean < 80 AND dark_ratio > 0.07)
 │    │    │    ├─ gamma.choose_gamma()                             γ 0.9 → 0.8 → 0.7 시도
 │    │    │    │    └─ 후보마다 gamma.apply_gamma() → quality.measure_in_masks() → gamma.guard_triggered()
 │    │    │    └─ gamma.guard_triggered()                          최종 γ 포화 재확인 (걸리면 원본으로)
 │    │    ├─ gaussian.decide_gaussian() → (gaussian.masked_gaussian())   ⑬ 현재 off
 │    │    ├─ quality.measure_in_masks()                            ⑭ 최종 품질
 │    │    └─ 크기 · 도로 밖 화소 불변 검사                          ⑮ 출력 규약
 │    ├─ (manual_required · on_fail=use_candidate · 후보 마스크 있음)
 │    │    → pipeline.process_image(manual_mask=후보 마스크,
 │    │                             manual_source="auto_candidate_unverified")   ⑧ 한 번 더, unverified_mask=True
 │    └─ detector_adapter.to_detector_input()                       D. 노면 영역 자르기(+16px) · 긴 변 1024
 ├─ 6 run_final.detect_image()                                      (전처리 success일 때만)
 │    ├─ detector_corrections.judge_tags()                          E-1 · E-2 선명도 → blur_rt / not_blur_rt
 │    │    └─ detector_corrections.road_sharpness()
 │    ├─ detector_corrections.apply_corrections()                   E-3 gaussian → line_close → unsharp → clahe (노면 마스크 안)
 │    ├─ detect() · texture_potholes()                              균열 · 포트홀 후보
 │    ├─ road_detection.pipeline.road_overlap() ≥ 0.5               도로 위 판정
 │    └─ detector_adapter.to_original_bbox()                        원본 좌표로 되돌림
 ├─ 7 detector_corrections.road_sharpness() → results.csv의 sharpness
 ├─ 8 images/ · road_mask/ · road_view/ · preprocessed/ 저장          F. 사진별 저장 (--no-images면 생략)
 │  ── 반복 끝 ──
 │
 └─ 9 results.csv · detections.json · review_sheet.jpg · summary.csv · run_config.json   F. 실행 단위 저장
```

`run_final_PBL.py` · `run_final_test.py`는 위 4~6을 같은 함수(`run_final.prepare` · `run_final.detect_image`)로 부르고, 저장만 다르다 (5절).

### 7-3. 실행 명령

```powershell
# final0 폴더에서 (opencv-contrib-python 필요)
python run_final.py --limit 20               # rdd_dev 처음 20장 → final-result/run_<시각>/
python run_final.py --dataset provided       # 제공 13장
python run_final.py --dataset <사진 폴더>     # 아무 사진 폴더 (하위 폴더까지)
python run_final.py --no-images              # 그림 없이 CSV · JSON만
python run_final_PBL.py                      # data/PBL module 1 - dataset/ → final-result-PBL/run_<시각>/
python run_final_test.py                     # data/test/ → final-result-test/run_<시각>/
```

전처리 수치를 바꾸려면 `configs/preprocessing.json`을, 보정 · 연결 수치를 바꾸려면 `configs/detection_final.json`의 `corrections` · `integration`을 고친다.
