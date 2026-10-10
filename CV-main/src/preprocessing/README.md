# 전처리 파트 (`src/preprocessing/`)

도로 영상에서 **도로 영역(road_mask)을 자동으로 찾고 검증한 뒤, 도로 안에서만 밝기를 보정**해 검출기에 넘긴다.
고전 영상처리만 쓴다 (AI · 딥러닝 없음). 설정 v1.1 (`configs/preprocessing.json`, `preproc-v1.1-provisional`).

- 모든 수치는 `configs/preprocessing.json` 값이다. 대부분 **잠정값**이며, 최적값이라는 근거는 없다.
- 길이(px) 단위는 별도 표시가 없으면 **작업 해상도(긴 변 1024)** 기준이다.
- 성능 수치는 2026-10-10 RDD2020 dev(563장)에서 잰 값이다. rdd_test(241장)로는 재지 않았다.

문서 순서: **1. 전체 흐름 → 2. 입력 → 3. 처리 → 4. 출력 → 5. 성능 → 6. 파일별 기능 · 실행 순서**

---

## 1. 전체 흐름

```text
[입력]  ─────────────────────────────────────────────────────────────────────────────
 A. 실행 인자 해석          --input / --dataset · --images · --limit · --set · --manual-mask(-dir) · --edit-failed
 B. 설정 읽기 · 검사        configs/preprocessing.json + --set 덮어쓰기 → 기본 값 검사
 C. 입력 목록 만들기        파일 · 폴더 · 데이터셋 → rdd_test 보호 → 이름 거르기 → 개수 제한 → image_id 부여
 D. 결과 폴더 만들기        outputs/preprocessing/run_<시각>/ + run_config.json 기록
 ── 사진마다 반복 ──
 E. 영상 읽기 · 형식 검사   BGR uint8 H×W×3 (실패하면 그 사진만 error로 기록하고 다음 사진)
 F. 수동 마스크 찾기        있으면 읽고 검사 (못 쓰면 경고 후 자동 마스크로 진행)

[처리]  process_image() ──────────────────────────────────────────────────────────────
 ① 작업 영상 (긴 변 1024)  ② 특징 지도  ③ 격자 특징  ④ 시드 고르기  ⑤ Region Growing
 ⑥ 마스크 정리  ⑦ 원본 크기 복원  ⑧ 검증 PASS/FAIL → 최종 road_mask 결정 (수동 · 편집기 · 자동)
 ⑨ analysis_mask  ⑩ 초기 품질  ⑪ 조건부 Gamma  ⑫ 노이즈 재측정  ⑬ 조건부 Gaussian(꺼 둠)
 ⑭ 최종 품질  ⑮ 출력 규약 검사

[출력]  ──────────────────────────────────────────────────────────────────────────────
 G. 사진별 저장             images/<image_id>/ processed_image · road_mask · analysis_mask · metadata · overlay
 H. 실행 단위 저장          summary.csv · run_summary.json · review_sheet.jpg · 화면 로그
 I. (검출 실행 시)          detector_adapter → 도로 영역 자르기 · 긴 변 1024 → detect() → 원본 좌표 · 도로 위 판정
```

---

## 2. 입력

### A. 실행 인자 (`run_preprocess.py`)

| 인자 | 뜻 |
|---|---|
| `--input <파일 또는 폴더>` | 사진 한 장 또는 폴더 (하위 폴더까지). `--dataset`과 둘 중 하나는 필수 |
| `--dataset <이름>` | `rdd_dev` · `rdd_tune` · `rdd_val` · `provided` · `captured` |
| `--images 이름 …` | 입력 중 이 파일 이름만 처리 (못 찾은 이름은 경고) |
| `--limit N` | 정렬된 처음 N장만 (1 이상) |
| `--config <경로>` | 설정 파일 (기본 `configs/preprocessing.json`) |
| `--set 키.경로=값 …` | 설정 덮어쓰기 (예: `gamma.mode=off`). 값은 JSON으로 해석, 실패하면 문자열 |
| `--output <폴더>` | 결과 폴더. 없거나 비어 있어야 함 (기존 결과를 덮어쓰지 않음) |
| `--manual-mask <PNG>` | 한 장 입력일 때 쓸 수동 road_mask |
| `--manual-mask-dir <폴더>` | 사진별 수동 마스크 폴더 |
| `--edit-failed` | 자동 마스크가 FAIL이면 다각형 편집기를 연다 (GUI 필요) |
| `--review` | PASS 결과도 편집기로 확인 · 수정 (`manual_correction.review_pass_results`와 같음) |
| `--allow-rdd-test` | rdd_test 보호 해제 — 최종 설정이 확정된 뒤 최종 확인에만 |

### B. 설정 읽기 · 검사 (`config.py`)

1. JSON을 읽고 `_`로 시작하는 키(설명 주석)를 지운다.
2. `--set` 덮어쓰기를 차례로 적용한다 (없는 키를 덮어쓰려 하면 오류).
3. 자주 틀리는 값만 검사한다: 큰 항목(roi, gamma 등)이 빠짐, Gamma · Gaussian 모드 이름,
   Gaussian을 켰는데 σ ≤ 0, 커널 크기가 짝수. 그 밖의 값은 검사하지 않는다.
4. 최종 설정의 SHA-256 해시를 만들어 결과에 남긴다 (같은 설정인지 확인용).

### C. 입력 목록 만들기 (`collect_images`)

| 순서 | 처리 |
|---|---|
| 1 | `--dataset`: RDD는 `labels/split_rdd.csv`(dev/test) · `labels/split_rdd_dev.csv`(tune/val)로 고르고, `provided`/`captured`는 `data/` 아래 폴더. `rdd` · `rdd_test`는 `--allow-rdd-test` 없이는 거부 |
| 1′ | `--input`: 파일이면 그 한 장. 폴더면 하위까지 확장자 `.jpg .jpeg .png .bmp .tif .tiff .webp` 파일 (`.`으로 시작하는 숨김 경로 제외), 정렬 |
| 2 | `--images` 이름 거르기 |
| 3 | **rdd_test 보호**: `split_rdd.csv`의 test 이미지(241장)는 폴더 입력이어도 건너뛰고 몇 장 건너뛰었는지 알린다 |
| 4 | `--limit` |
| 5 | `image_id` = 입력 폴더 기준 상대 경로(확장자 제외)를 `__`로 이은 것 (예: `sub__Japan_000015`) |

입력이 0장이면 오류로 멈춘다. `--manual-mask`는 한 장 입력일 때만 쓸 수 있다.

### D. 결과 폴더 (`new_run_dir`)

기본 `outputs/preprocessing/run_YYYYmmdd_HHMMSS/` (한국 시간, 같은 초에 겹치면 `_2`, `_3` …).
처리를 시작하기 전에 `run_config.json`을 먼저 쓴다 (중간에 멈춰도 어떤 설정으로 돌렸는지 남도록).

### E. 영상 읽기 (`image_io.read_image`)

- 한글 경로 대응을 위해 `np.fromfile` + `cv2.imdecode`로 읽는다 (`IMREAD_COLOR` → 회색조 · 알파 채널 영상도 BGR 3채널로 읽힘).
- 없는 파일 · 빈 파일 · 해독 실패는 `ImageLoadError`가 난다. 그 사진은 `status = error`, `stage = read`로 기록하고 **배치는 계속**된다.
- 형식 검사: 비어 있지 않은 `uint8` `H×W×3` 배열이어야 한다. 자동 변환은 하지 않는다.

### F. 수동 마스크 입력 (`mask_editor`)

- `--manual-mask-dir`에서 `<이름>.png` → `<이름>_road_mask.png` → `<이름>/road_mask.png` 순서로 찾는다.
- 회색조로 읽어 **128 이상 = 도로**(255), 나머지 0으로 만든다. **원본과 크기가 같아야** 하고, 도로 면적이 영상의 **0.1% 이상**이어야 한다.
- 못 쓰는 마스크는 경고(`external_mask_unusable`)를 남기고, 그 사진은 자동 마스크로 처리한다.
- 수동 마스크가 있으면 **자동 검증 결과(PASS/FAIL)와 관계없이** 그것을 최종 road_mask로 쓴다. 자동 마스크와 검증 결과는 기록용으로 계속 계산한다.

---

## 3. 처리 (`pipeline.process_image`)

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
첫 시드가 도로가 아니면 이후 시드도 틀릴 수 있다. 이것은 알려진 한계이며, 검증과 수동 확인으로 보완한다.

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

작업 해상도 마스크를 float로 원본 크기까지 선형 보간하고, `upsample_threshold 0.5` 이상을 도로(255)로 본다 → 자동 road_mask.

### ⑧ 검증과 최종 road_mask 결정 — `validation`

| 지표 | 정의 | 기준 |
|---|---|---|
| mask_area_ratio | 도로 화소 / 영상 화소 | ≥ 0.10 |
| largest_component_ratio | 가장 큰 8-연결 요소 / 도로 화소 | ≥ 0.80 |
| hole_ratio | 구멍 면적 / 구멍 채운 면적 | ≤ 0.15 |
| seed_consistency | 최종 마스크 안의 시드 수 / 시드 수 | ≥ 0.60 |

하나라도 못 넘으면 **FAIL**이다 (상태는 PASS/FAIL 둘뿐). 시드 0개나 빈 마스크도 FAIL이고, 정의되지 않은 지표도 FAIL로 처리한다(`undefined_metric_policy fail`). FAIL 사유는 모두 기록한다.

최종 road_mask는 다음 우선순위로 정한다.
1. **수동 마스크 파일** (F에서 읽은 것) → `source = manual`
2. **편집기** (`--edit-failed`이고 FAIL일 때, 또는 `--review`) — 자동 마스크에서 시작해 다각형으로 고친다.
   a 더하기(기본) · d 빼기 · r 바꾸기 · e 가장 큰 외곽선 불러오기 · Esc 다시 그리기 · Enter 확정 · q 취소 → `source = manual`
3. **자동 마스크** — PASS일 때만 → `source = auto`
4. 셋 다 해당하지 않으면(FAIL이고 수동 마스크 없음): 전처리는 **`manual_required`로 여기서 멈춘다** (후보 마스크만 저장).
   검출 실행기(`run_road_detection.py`)는 기본으로 이 후보 마스크를 수동 마스크 자리에 넣어 다시 처리하고 `unverified_mask = True`로 표시한다 (`--on-fail skip`이면 건너뜀).

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

### ⑫ 노이즈 재측정

Gamma를 적용했으면 그 영상에서 품질을 다시 재고(`after_gamma`), 적용하지 않았으면 초기값을 그대로 쓴다. Gaussian 판단은 이 값으로 한다.

### ⑬ 조건부 Gaussian — `gaussian` (v1.1: `mode off`)

켜면(`conditional`) noise_sigma ≥ 0.6일 때 3×3, σ 0.8 Gaussian을 도로 안에만 적용한다.
마스크 인식 방식 out = G(I·M) / G(M)을 써서 경계에서 비도로 화소 값이 섞이지 않게 한다.
**꺼 둔 이유**: 노면 질감 때문에 사진의 56%에 적용됐고, 적용된 사진에서 균열 Recall이 0.117 떨어졌다 (95% 구간이 0을 포함하지 않음).

### ⑭ 최종 품질 · ⑮ 출력 규약 검사

최종 영상의 품질을 재고 초기값과의 차이를 계산한다. 그다음 아래 규약을 검사하며, 어기면 error다.
- 출력 영상 크기 = 입력 크기, road_mask 크기 = 입력 H×W
- **road_mask 밖 화소가 입력과 완전히 같음**

처리 중 어느 단계에서든 예외가 나면 그 사진만 `status = error`(실패한 단계 · 원인 기록)로 끝내고, 배치는 다음 사진으로 넘어간다.

---

## 4. 출력

### 상태 (`metadata.status`)

| 상태 | 뜻 | 저장되는 영상 |
|---|---|---|
| `success` | 최종 road_mask와 보정 영상이 만들어짐 | processed_image · road_mask · analysis_mask · overlay |
| `manual_required` | 자동 마스크가 FAIL이고 수동 마스크가 없음 → 보정 단계로 가지 않음 | road_mask_candidate · overlay |
| `error` | 읽기 · 처리 · 규약 검사 실패 | (있으면) road_mask_candidate · overlay |

### G. 사진별 파일 — `run_<시각>/images/<image_id>/`

| 파일 | 내용 · 규약 |
|---|---|
| `processed_image.png` | 보정된 영상. 원본과 같은 크기 · 좌표 · BGR uint8. 도로 밖 화소는 입력과 동일 |
| `road_mask.png` | 최종 도로 마스크. 원본 크기 uint8, 도로 255 / 비도로 0 |
| `analysis_mask.png` | 품질 측정용 침식 마스크 (road_mask의 부분집합) · `output.save_analysis_mask` |
| `road_mask_candidate.png` | FAIL일 때의 자동 후보 마스크 (수동 수정의 시작점) |
| `overlay.jpg` | 검토 그림 (긴 변 최대 1024): 도로(초록 반투명 · 노랑 외곽) · analysis_mask 외곽(파랑) · 시드(빨강, 못 자란 시드는 자홍) · 상태 문구 · `output.save_debug` |
| `metadata.json` | 아래 구조 · `output.save_metadata` |

### `metadata.json` 구조

| 키 | 내용 |
|---|---|
| `image_id` · `input_path` · `image_size` | 입력 정보 |
| `config_version` · `config_sha256` | 사용한 설정 버전과 해시 |
| `status` | success / manual_required / error |
| `roi` | 방법 이름, 작업 배율 · 크기, 격자 크기, 시드 목록(행 · 열 · 원본 좌표 x,y · 점수 · 6개 성분 · 성장 여부 · 덮은 비율 · 색 임계값), 단계별 시간 |
| `validation` | PASS/FAIL, 지표 값(4개 + 연결 요소 수 · 시드 수 · 도로 화소 수), 지표별 검사 결과, FAIL 사유 |
| `manual_correction` | 수동 수정 필요 여부 · 적용 여부 · 출처(external_file / interactive_editor) · 파일 경로 |
| `final_mask` | 최종 마스크 출처(auto / manual) · 도로 화소 수 · 면적 비율 |
| `analysis_mask` | 커널 · 남은 화소 수 · road_mask 대비 비율 · 측정 충분 여부 |
| `quality` | 측정 조건(배율 · 회색조 식 · 임계값), `initial` · `after_gamma` · `final` 지표 8개, 최종 − 초기 차이 |
| `gamma` | 모드 · 선택 방식, 조건별 값과 판정, 적용 여부, 쓴 γ(`value_used`, 안 쓰면 1.0), 포화 방지 작동 여부, 시도 기록(`tried`), 사유 |
| `noise_remeasurement` | Gaussian 판단에 쓴 noise_sigma · laplacian_variance |
| `gaussian` | 모드 · 적용 여부 · σ · 커널 · 판단 사유 (`mode_off` 등) |
| `timing_ms` | road_mask · validation · analysis_mask · quality_initial · gamma · noise_remeasurement · gaussian · quality_final · total |
| `outputs` | 저장한 파일 이름 |
| `warnings` · `errors` | 경고 목록 · 오류(단계 · 종류 · 메시지) |
| `versions` | Python · OpenCV · NumPy 버전 |

실수는 소수 6자리로 반올림하고, NaN은 저장하지 않는다.

### H. 실행 단위 파일 — `run_<시각>/`

| 파일 | 내용 |
|---|---|
| `run_config.json` | 실행 프로그램 · 인자 · 설정 파일 경로 · **실제 적용된 설정 전체** · 해시 · 사진 수 · 시작 시각 · 라이브러리 버전 (처리 전에 기록) |
| `summary.csv` | 사진당 한 줄 (UTF-8 BOM, 엑셀에서 바로 열림). 열: image_id · input_path · status · auto_validation · fail_reasons · final_mask_source · manual_required · manual_applied · 검증 지표 4개 · n_seeds · analysis_pixels · analysis_sufficient · gamma_applied · gamma_value · gamma_reason · gaussian_applied · gaussian_sigma · gaussian_kernel · gaussian_reason · initial_품질 8개 · final_품질 8개 · time_ms · warnings · errors |
| `run_summary.json` | 사진 수 · success 수 · 자동 PASS/FAIL 수 · 수동 수정 필요 · 적용 수 · error 수 · Gamma/Gaussian 적용 수 · 처리 시간 통계(평균 · 중앙값 · 최소 · 최대) · FAIL 사유별 개수 · 전체 소요 시간 |
| `review_sheet.jpg` | 처음 48장의 overlay를 4열로 모은 그림 · `output.review_sheet` |

화면 로그 (`output.log_level`이 `quiet`가 아니면): 사진마다 `[번호/전체] image_id: 상태 (PASS/FAIL · 마스크 출처 · γ · σ 또는 FAIL 사유) 시간`, 끝에 합계를 출력한다.
종료 코드는 error가 0장이면 0, 있으면 1이다.

### I. 검출로 넘길 때 (`detector_adapter.py` → `src/road_detection/`)

`run_road_detection.py`는 사진마다 위 전처리를 실행한 뒤 이어서 다음을 한다.
1. road_mask의 외접 사각형에 `crop_margin 16`px을 더해 잘라 내고, 긴 변 1024로 맞춘다 (영상 INTER_AREA/CUBIC, 마스크 NEAREST).
   잘라 낸 영역 안의 비도로 화소는 검정으로 칠하지 않는다 (인공 경계가 생기지 않게).
2. `detect()` 실행 → 후보 박스를 `to_original_bbox()`로 원본 좌표로 되돌린다.
3. 박스 면적의 50% 이상이 도로 위면 "도로 위 후보"로 판정한다.
4. 위 전처리 파일에 더해 `result.jpg`(도로 초록 · 균열 빨강 · 포트홀 파랑 · 도로 밖 회색) · `detections.json`을 사진별로,
   `summary.csv` · `detections.csv` · `run_summary.json` · `review_sheet.jpg`를 실행 단위로 `outputs/road_detection/run_<시각>/`에 저장한다.

---

## 5. 측정된 성능 (2026-10-10, RDD2020 dev)

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
  그래서 검출 실행기는 FAIL이어도 후보 마스크로 검출한다.

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
| **v1.1 (현재)** | 0.179 | **0.371** | 0.241 | 2.26 |
| 참고: 자동 마스크만 (Gamma 끔) | 0.194 | 0.363 | 0.253 | 1.99 |

- 전처리 없음 → v1.1: F1 +0.140 · Recall +0.095 · 가짜 −3.24/장 (모두 95% 구간이 0을 넘음).
- v1.0 → v1.1: Recall +0.098 (tune 394장 +0.100 · val 169장 +0.095), Precision은 차이 없음, 가짜 +0.69/장.
- 적응형 Gamma 자체는 검출 성능을 높이지 않는다 (끈 경우보다 F1 −0.012, Precision −0.016).
- 사람이 그린 정답 마스크를 넣어도 맞힌 균열 수가 같았다 (36장에서 24개 → 24개). ROI를 더 다듬어도 검출 이득은 작다.
- v1.1 변경은 dev 진단으로 정했으므로 완전히 독립된 확인은 아니다. 최종 검출기 p2bd(opencv-contrib 필요)로는 아직 재지 않았다.

### 처리 시간 (이 개발 PC)

- 전처리 한 장 중앙값 약 200 ms (600×600 RDD 기준). 큰 영상(Norway 3650×2044 등)은 약 1~1.2 s.
- 검출은 약 90 ms/장 (긴 변 1024).

---

## 6. 파일별 기능 · 실행 순서

### 6-1. 각 py 파일이 하는 일

실행 순서대로 나열했다. "단계"는 1절 흐름도의 기호(A~I, ①~⑮)다.

| 순서 | 파일 | 단계 | 하는 일 | 주요 함수 | 누가 부르나 |
|---:|---|---|---|---|---|
| 0 | `run_preprocess.py` | A~D · H | **전처리 실행 시작점.** 인자 해석, 입력 목록, 결과 폴더, 사진마다 반복, summary · review_sheet 저장 | `main` · `parse_args` · `collect_images` · `new_run_dir` · `summary_row` · `run_summary` · `write_review_sheet` | 사용자 (명령줄) |
| 1 | `config.py` | B | 설정 JSON 읽기, `_` 주석 키 제거, `--set` 덮어쓰기, 기본 값 검사, 설정 해시 | `load_config` · `apply_override` · `check_config` · `config_hash` | run_preprocess · edit_road_mask · pipeline |
| 2 | `image_io.py` | E · G | 영상 · 마스크 읽기/쓰기(한글 경로), 형식 검사, 0/255 이진화 | `read_image` · `read_mask` · `write_image` · `validate_image` · `validate_mask` · `to_binary_mask` | 거의 모든 파일 |
| 3 | `mask_editor.py` | F · ⑧ | 수동 마스크 파일 찾기 · 읽기, 다각형 편집기 창(matplotlib) | `find_external_mask` · `load_external_mask` · `edit_mask_interactive` | run_preprocess · pipeline(편집기 인자로) · edit_road_mask |
| 4 | `pipeline.py` | ①~⑮ · G | **한 장 처리의 중심.** 아래 5~14번 파일을 순서대로 부르고 metadata를 채운 뒤 저장 | `process_image` · `_choose_final_mask` · `_apply_gamma_step` · `save_result` · `make_overlay` | run_preprocess · road_detection |
| 5 | `road_mask.py` | ①~⑦ | 자동 Road Mask 전체 묶음. 작업 영상 축소 → 6~9번 호출 → 원본 크기 복원 | `extract_road_mask` · `work_image` · `upsample_mask` | pipeline · edit_road_mask |
| 6 | `grid_features.py` | ② ③ | Lab · gradient · texture 지도, 32px 격자 특징, 색 거리 식 | `compute_feature_maps` · `compute_grid_features` · `color_distance` | road_mask · seeds |
| 7 | `seeds.py` | ④ | 칸마다 6개 성분 점수 → 가중 기하평균 → 시드 3~5개 선택 | `score_cells` · `select_seeds` | road_mask |
| 8 | `region_growing.py` | ⑤ | 시드 기준값(Lab · σ_ab · 질감)으로 조건 만족 화소를 연결해 넓힘 | `grow_regions` · `acceptance_mask` | road_mask |
| 9 | `mask_refine.py` | ⑥ | closing · opening · 구멍 채우기 · 작은 조각 제거 | `refine_mask` · `fill_holes` | road_mask · mask_validation |
| 10 | `mask_validation.py` | ⑧ | 면적 · 최대 요소 · 구멍 · 시드 일관성 측정 → PASS/FAIL | `compute_mask_metrics` · `validate_road_mask` | pipeline |
| 11 | `analysis_mask.py` | ⑨ | road_mask 사본을 5×5 침식, 측정 충분 여부 판단 | `make_analysis_mask` | pipeline |
| 12 | `quality.py` | ⑩ ⑫ ⑭ | 긴 변 1024 사본에서 밝기 · 어두움 · 포화 · 선명도 · 노이즈 지표 8개 측정 | `measure_in_masks` · `measure_quality` | pipeline (Gamma 후보마다 다시 호출) |
| 13 | `gamma.py` | ⑪ | 조건 판단 → γ 0.9 / 0.8 / 0.7 차례로 시도 → 포화 방지 | `decide_gamma` · `choose_gamma` · `apply_gamma` · `guard_triggered` | pipeline |
| 14 | `gaussian.py` | ⑬ | 노이즈 조건 판단 · 마스크 인식 Gaussian (현재 `off`라 판단만 하고 적용 안 함) | `decide_gaussian` · `masked_gaussian` | pipeline |
| — | `detector_adapter.py` | I | 검출기 입력 만들기(도로 영역 자르기 · 긴 변 1024), 박스를 원본 좌표로 되돌림 | `to_detector_input` · `to_original_bbox` | `road_detection/pipeline.py` (검출 실행 때만) |
| — | `edit_road_mask.py` | — | 수동 마스크만 따로 만드는 실행기 | `main` | 사용자 (명령줄) |
| — | `__init__.py` | — | 패키지 표시 (한 줄) | — | — |

### 6-2. 호출 순서 — `run_preprocess.py` (전처리만)

```text
run_preprocess.py  main()
 ├─ 1 run_preprocess.parse_args()                               A. 인자 해석
 ├─ 2 config.load_config()                                      B. 설정 읽기 · --set · 검사
 ├─ 3 run_preprocess.collect_images()                           C. 입력 목록 · rdd_test 보호
 ├─ 4 run_preprocess.new_run_dir() → run_config.json 기록       D. 결과 폴더
 │
 │  ── 사진마다 반복 ──
 ├─ 5 image_io.read_image()                                     E. 영상 읽기 (실패 → error 기록, 다음 사진)
 ├─ 6 mask_editor.find_external_mask() · load_external_mask()   F. 수동 마스크 (옵션이 있을 때만)
 ├─ 7 pipeline.process_image()                                  ①~⑮ 한 장 처리
 │    ├─ image_io.validate_image()                              입력 형식 검사
 │    ├─ road_mask.extract_road_mask()                          ①~⑦ 자동 Road Mask
 │    │    ├─ road_mask.work_image()                            ① 긴 변 1024로 축소
 │    │    ├─ grid_features.compute_feature_maps()              ② Lab · gradient · texture
 │    │    ├─ grid_features.compute_grid_features()             ③ 32px 격자 특징
 │    │    ├─ seeds.score_cells() → seeds.select_seeds()        ④ 시드 점수 · 선택
 │    │    ├─ region_growing.grow_regions()                     ⑤ 시드별 영역 확장
 │    │    ├─ mask_refine.refine_mask()                         ⑥ 모폴로지 정리
 │    │    └─ road_mask.upsample_mask()                         ⑦ 원본 크기 복원
 │    ├─ mask_validation.compute_mask_metrics()
 │    │   → mask_validation.validate_road_mask()                ⑧ PASS / FAIL
 │    ├─ pipeline._choose_final_mask()                          ⑧ 최종 마스크: 수동 파일 → 편집기 → 자동(PASS)
 │    │    └─ mask_editor.edit_mask_interactive()               (--edit-failed / --review일 때만)
 │    │    ※ 최종 마스크가 없으면 여기서 manual_required로 끝
 │    ├─ analysis_mask.make_analysis_mask()                     ⑨ 5×5 침식
 │    ├─ quality.measure_in_masks()                             ⑩ 초기 품질
 │    ├─ pipeline._apply_gamma_step()                           ⑪ ⑫
 │    │    ├─ gamma.decide_gamma()                              조건 판단 (gray_mean < 80 AND dark_ratio > 0.07)
 │    │    ├─ gamma.choose_gamma()                              γ 0.9 → 0.8 → 0.7 시도
 │    │    │    └─ 후보마다 gamma.apply_gamma() → quality.measure_in_masks() → gamma.guard_triggered()
 │    │    │       (적용 → 재측정 → 포화 방지. 포화되면 그보다 강한 γ는 보지 않음)
 │    │    └─ gamma.guard_triggered()                           최종 γ 포화 재확인 (걸리면 원본으로)
 │    ├─ gaussian.decide_gaussian() → (gaussian.masked_gaussian())  ⑬ 현재 off
 │    ├─ quality.measure_in_masks()                             ⑭ 최종 품질
 │    └─ (pipeline 안에서) 크기 · 도로 밖 화소 불변 검사         ⑮ 출력 규약
 ├─ 8 pipeline.save_result()                                    G. 사진별 저장
 │    ├─ image_io.write_image()                                 processed_image · road_mask · analysis_mask
 │    └─ pipeline.make_overlay()                                overlay.jpg
 │  ── 반복 끝 ──
 │
 └─ 9 summary.csv · run_summary() → run_summary.json · write_review_sheet()   H. 실행 단위 저장
```

### 6-3. 다른 실행기에서의 호출 순서

**`edit_road_mask.py` — 수동 마스크만 만들 때**

```text
edit_road_mask.py  main()
 ├─ 1 config.load_config()
 ├─ 2 image_io.read_image()
 ├─ 3 시작 마스크: image_io.read_mask() (--candidate가 있을 때)
 │                 또는 road_mask.extract_road_mask() (없을 때, 위 ①~⑦과 같음)
 ├─ 4 mask_editor.edit_mask_interactive()        다각형으로 수정 (GUI)
 └─ 5 image_io.write_image()                     → outputs/preprocessing/manual_masks/<이름>.png
```

**`src/run_road_detection.py` — 전처리 + 검출 (전처리 폴더 밖, 이 폴더의 함수를 가져다 씀)**

```text
run_road_detection.py
 ├─ 1~6 config · run_preprocess.collect_images/new_run_dir · image_io · mask_editor   (6-2의 1~6과 같음)
 ├─ 7 road_detection/pipeline.run_image()
 │    ├─ pipeline.process_image()                            (6-2의 7과 같음)
 │    │    ※ FAIL(manual_required)이고 on_fail=use_candidate면
 │    │      후보 마스크를 수동 마스크 자리에 넣어 process_image()를 한 번 더 실행
 │    ├─ detector_adapter.to_detector_input()                I-1 도로 영역 자르기 · 긴 변 1024
 │    ├─ detect()  (검출 알고리즘, 수정하지 않음)             I-2 균열 · 포트홀 후보
 │    ├─ 박스마다 road_overlap ≥ 0.5 → on_road 판정           I-3
 │    └─ detector_adapter.to_original_bbox()                 I-4 원본 좌표로 되돌림
 └─ 8 pipeline.save_result() + result.jpg · detections.json/csv 저장
```

### 6-4. 실행 명령

```powershell
# 전처리만 (저장소 맨 위 폴더)
py src/preprocessing/run_preprocess.py --dataset rdd_dev --limit 20
py src/preprocessing/run_preprocess.py --input <사진 또는 폴더> --set gamma.mode=off
# 수동 마스크 만들기 → 다시 처리
py src/preprocessing/edit_road_mask.py --image <사진>
py src/preprocessing/run_preprocess.py --input <사진> --manual-mask-dir outputs/preprocessing/manual_masks
# 전처리 → 검출 전체
py src/run_road_detection.py --dataset rdd_dev --limit 20
```
