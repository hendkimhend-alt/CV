# 성능 비교 방법 (팀 공통) — final0 판

> **final0 안내** — 원본 저장소 `EVALUATION.md`를 final0에 맞게 고친 것이다. 재는 정의 · 판정 규칙(2 · 4 · 7절)은 원본과 같고,
> 돌리는 방법 · 결과가 남는 곳(1 · 3 · 5절)만 final0 기준으로 바꿨다. final0은 **최종 구성 하나만** 돌리고 채점한다 (`run_final.py`).
> 실험 기록은 원본 `results/experiments.md`를 복사한 [`experiments.md`](experiments.md) (지나간 기록, 경로 · 명령은 원본 저장소 기준).

> 누가 무엇을 바꾸든 **이 문서의 방법 하나로** 재고 비교한다. 그래야 "무엇을 바꿔서 얼마나 좋아졌나"를 서로 같은 숫자로 말할 수 있다.
> 이 방법과 다르게 잰 숫자는 비교 표에 넣지 않는다.

---

## 1. 고정하는 것 — 바꾸지 않는다

| 항목 | 고정값 | 이유 |
|---|---|---|
| 코드 | final0 **`run_final.py`** + `src/evaluate.py` (원본 `src/evaluate.py`를 고치지 않고 복사) | 측정 코드가 하나여야 숫자가 같은 뜻 |
| 구성 | 균열 · 포트홀 최종 구성 (10/12) — `configs/preprocessing.json` · `configs/detection_final.json` · `configs/pothole_final.json` | 원본 `configs/final.json`과 같은 처리 순서 · 같은 설정 ([`README.md`](../README.md)) |
| 데이터 | **`rdd_dev` 563장** (RDD 개발 세트, `labels/split_rdd.csv`) | 실험 · 비교는 여기서만. `rdd_test`는 아래 7절 |
| 측정 | **in50** 판정 · 균열 | 아래 2절 |
| 지표 | **Recall · Precision · F1** + 가짜/장 · 조각/맞힘 (10/10 고정) | 아래 2절 |
| 좌표 | 후보 · 정답 모두 **원본 좌표** (`evaluate(dets, gt, None)`) | 정답을 잘라 내지 않으므로 모든 정답이 분모에 들어감 |

- final0에는 실험 옵션(`--config` · `--set` · `--extra` · `--calibrate` · `--degrade` · `--cache` · `--det-set`)이 없다. 설정을 바꿔 비교하는 실험은 원본 저장소에서 한다 (3절)
- 원본 비교 기준: 균열 최종 구성 F (`t_c10_post`) · 최종 (균열 + 포트홀, `final`) — 숫자는 6절

## 2. 무엇을 재나 — 정의

| 지표 | 뜻 | 계산 |
|---|---|---|
| **Recall** | 진짜 손상 중 찾은 비율 (정답 쪽에서 셈) | 맞힌 정답 수 ÷ 전체 정답 수 (ROI 밖으로 잘린 정답도 분모에 넣어 **놓침**으로 셈) |
| **Precision** | 내놓은 후보 중 정답 위에 있는 비율 (후보 쪽에서 셈) | 정답 박스 안 후보 수 ÷ 전체 후보 수 = (후보 − 가짜) ÷ 후보 |
| **F1** | 둘의 조화평균 | 2 · P · R ÷ (P + R) |
| **가짜/장** | 사진 한 장당 가짜 수 | 가짜 수 ÷ 사진 수 (final0: 정답 있는 사진 수) |
| **조각/맞힘** | 맞힌 정답 하나에 들어간 후보 수 | 정답 박스 안 후보 수 ÷ 맞힌 정답 수 |

- **맞힘 (in50)**: 후보 박스 면적의 **절반 이상이 같은 종류 정답 박스 안**. 정답 하나에 맞힘은 최대 1개 (겹침이 큰 쌍부터). 같은 정답 안의 추가 조각은 맞힘도 가짜도 아님
- **가짜**: 어느 정답 박스 안에도 들어가지 않은 후보
- **Precision을 후보 쪽에서 세는 이유 (10/10 고정)**: 이전 계산(맞힌 정답 ÷ (맞힌 정답 + 가짜))은 맞힘을 **정답 수**로, 가짜를 **조각 수**로 세서 단위가 달랐다. 진짜 균열 하나가 조각 6개로 나오면 1개, 가짜가 조각 4개로 나오면 4개로 세어져 P가 0.07 ~ 0.10으로 지나치게 낮았다. 이제 P는 후보 단위로, R은 정답 단위로 각각 센다 (균열 논문의 픽셀 P · R — 검출 쪽 · 정답 쪽에서 각각 셈 — 을 박스로 옮긴 것). 코드: `src/evaluate.py`의 `prf_in50`
- **허점과 보완**: 정답 안에서 잘게 쪼갤수록 P가 오를 수 있다 → **가짜/장 · 조각/맞힘을 항상 같이 보고**한다. P가 올랐어도 가짜/장이 늘었으면 "가짜가 줄었다"고 쓰지 않는다 (예: `baseline` → `p2bd`는 P 0.108 → 0.286이지만 가짜/장은 5.2 → 9.3, 조각/맞힘 1.7 → 4.9)
- 이전 계산 값은 `summary.csv`의 `crack_precision_hit` · `pothole_precision_hit`로 계속 남긴다 (1차 계획서 목표와 잇기 위해)
- 사진마다 센 뒤 **전체를 더해서** 비율을 낸다 (사진별 평균 아님)
- 균열과 포트홀은 따로 잰다. 판정은 균열 기준 (포트홀은 따로 표시)
- `results.csv`에는 사진별로 iou50 · iou30 판정도 남지만 (`src/evaluate.py`의 `CRITERIA`), 요약 · 판정은 in50만 쓴다

**목표** — Recall 우선 + Precision 하한

| | Recall | Precision |
|---|---|---|
| 지키는 선 (기준선 수준) | ≥ 0.278 | ≥ 0.108 |
| 달성 목표 | 0.50 | (이전 계산 0.10) |
| 도전 목표 | 0.70 | (이전 계산 0.30) |

- 지키는 선 = 기준선(`baseline`)의 값 — 계산을 바꿔도 규칙은 같아서 새 계산 값으로 다시 잰 것
- Precision 달성 · 도전 목표는 1차 계획서에서 **이전 계산으로 정한 숫자**다. 새 계산으로 숫자를 다시 정하지 않는다 (결과를 보고 목표를 고르면 목표가 의미 없어짐) → `precision_hit`으로 확인

## 3. 절차 — final0에서 재기

**① 돌리기** (final0 폴더에서)
```bash
python run_final.py --note "…"                       # rdd_dev 563장 (기본) → final-result/run_<시각>/
python run_final.py --limit 20                       # 빠른 확인 (판단은 563장 전체로만)
python run_final.py --dataset rdd_tune               # 튜닝 394장 (labels/split_rdd_dev.csv)
python run_final.py --dataset rdd_val                # 검증 169장
python run_final.py --dataset provided               # 제공 13장 (초안 정답)
python run_final.py --dataset <폴더> --gt <정답 파일>  # 아무 사진 폴더 + 정답 파일 (.json · .csv)
```
- 그 밖의 옵션: `--out <폴더>` (기본 `final-result/`) · `--no-images` (그림 없이 CSV만)
- RDD는 **데이터 이름**(`rdd_dev` · `rdd_tune` · `rdd_val`)으로 돌린다. RDD 사진 폴더를 경로로 주면 분할을 거치지 않아 **테스트 사진까지 섞이고**, 정답도 RDD ann에서 찾지 않는다 (7절)

**② 정답을 찾는 순서** (`run_final.py`의 `find_gt()`) — 정답이 없으면 정확도 칸은 비고 검출 · 측정값만 나온다

| 순서 | 정답 | 언제 |
|---|---|---|
| 1 | `--gt <파일>` | 주면 항상 이것 |
| 2 | `data/RDD2020_train/train/ann/<사진 이름>.json` | 데이터 이름이 `rdd`로 시작할 때 |
| 3 | `labels/<데이터 이름>.csv` | 폴더로 돌리면 데이터 이름 = 폴더 이름 |
| 4 | 사진 폴더 안의 `annotations.json` · `annotations.csv` | |
| 5 | `data/road_damage_annotated/annotations.json` | `provided`일 때 |

- 정답 파일 형식 · 종류 구분은 [`README.md`](../README.md). 정답 파일에 없는 사진은 "손상 없음"으로 채점한다
- `run_final_PBL.py` · `run_final_test.py`는 정확도를 계산하지 않는다 — 비교에 쓰지 않는다

**③ 숫자 읽기** — 화면에 범위별 `맞힘/정답 · R · P · F1 · 가짜/장`이 나오고, 같은 값이 `summary.csv`에 남는다 (5절)

**④ 설정을 바꿔 비교하려면 — 원본 저장소에서**

final0에는 아래 도구가 없다. 판정(4절)에 필요한 ✱는 원본에서만 낼 수 있다.
- 강한 흔적 총수 맞추기 (`analysis/calibrate.py`) — 찾기에 영향을 주는 것(전처리 · 찾기 · 잇기)을 바꾸면 Hessian 점수의 크기가 달라지므로, **강한 흔적 총수를 기준과 같게** 맞춘 강한 기준값으로 비교한다 (사진만 보고 정답은 보지 않음). final0 `configs/detection_final.json`의 강한 기준 132.17도 이렇게 맞춘 값이다
- 같은 사진끼리 짝지은 95% 범위 · ✱ (`analysis/compare_runs.py`, `--split`으로 튜닝 · 검증 나눠 보기)
- 실행마다 자동 기록 (`results/runs.csv`)

final0의 `configs/*.json`을 고쳐 돌리면 숫자는 나오지만, 위 맞추기 · 짝 비교 없이는 "개선"이라고 쓰지 않는다. `results.csv`에 사진별 in50 `tp · fp · fn`과 후보 수가 있어 같은 사진끼리 짝지을 재료는 남는다.

**절차 원칙** (원본과 같음)
- 돌리기 **전에** [`experiments.md`](experiments.md)에 한 줄: 무엇을 바꾸나 · 왜(가설) · **결정 규칙** (4절 표에서 고름)
- 하나만 바꾼다 — 데이터 · 나머지 설정은 그대로
- 새 기능은 **설정으로 켜고 기본은 꺼 둔다** (기존 결과가 바뀌지 않게)
- 판정하고 기록 — 결과 · 판정 · run 이름을 [`experiments.md`](experiments.md)에

## 4. 판정 — 개선인가

**Precision 하한 P_min** = 지키는 선 **0.108** (기준선 수준, 2절). Precision 차이는 가짜/장과 함께 본다 — P ✱ 상승이어도 가짜/장이 늘었으면 개선이라고 쓰지 않는다
- ✱ = 같은 사진끼리 짝지은 차이의 95% 범위가 0을 포함하지 않음 = 우연으로 보기 어려운 차이 (원본 `analysis/compare_runs.py`)
- 10/10 전 판정(P_min 0.066 · 0.10, 이전 계산)은 그때 규칙으로 내린 것이라 그대로 둔다. 새 계산으로 다시 재도 순서는 같다 (`p2bd` < `gf3`, A < B)

| 결과 (기준 대비, 563장) | 판정 |
|---|---|
| Recall ✱ 상승 + Precision ≥ P_min | ✅ 개선 |
| Recall 차이 없음 + Precision ✱ 상승 | ✅ 개선 |
| Recall ✱ 상승 + Precision ✱ 상승 | ✅ 개선 (예: S1 → P2b+D) |
| Recall ✱ 하락 | ❌ (Recall 우선) |
| 둘 다 ✱ 없음 | ❌ 차이 없음 (숫자가 조금 달라도 "개선"이라고 쓰지 않음) |
| Precision < P_min | ❌ |

- 규칙은 **돌리기 전에** 적은 것을 쓴다. 결과를 보고 규칙을 바꾸지 않는다
- **예외는 팀 결정으로만, 기록과 함께**: 예) G2(가이드 필터)는 Recall ✱ 하락으로 규칙상 ❌였지만, 처음으로 달성 목표(R 0.50 · P 0.10)를 둘 다 넘어 사용자 결정으로 채택 — `experiments.md`에 "규칙상 ❌ · 결정으로 채택"이라고 남김
- **여러 값 중 하나를 고를 때**: 튜닝 394장(`rdd_tune`)에서 위 표로 후보를 고르고, 후보 중 **가짜/장이 가장 적은 값 — 조각/맞힘이 기준보다 늘지 않는 것 중** (10/10부터 · P 최고로 고르면 정답 안 조각이 늘어난 값이 뽑힐 수 있음: 막대 닫힘 15px 사례) → 검증 169장(`rdd_val`)에서 "Recall이 기준보다 떨어지지 않고 Precision ≥ P_min"이면 ✅. 같은 사진으로 고르고 확인하면 점수가 부풀려진다
- 참고: 10/6 이전 초반 실험(E0 ~ E3)은 F1으로 판정했다 — 그 뒤 목표가 Recall 우선으로 바뀌면서 이 표로 통일

## 5. 기록 — 어디에 무엇이 남나

실행마다 `final-result/run_<시각>/`에 새로 남는다 (앞 결과를 덮어쓰지 않음). final0은 `results/runs.csv`에 기록하지 않는다 — 그 기록은 원본 저장소에만 있다.

| 파일 | 누가 | 무엇 |
|---|---|---|
| `results.csv` | 자동 | 사진별 — 아래 표 |
| `summary.csv` | 자동 | 범위(`scope`)별 정확도와 측정값 — 아래 표 |
| `detections.json` | 자동 | 사진별 후보 박스 (원본 좌표) |
| `run_config.json` | 자동 | 데이터 · `--limit` · 정답 출처 · 쓴 설정(전처리 해시 포함) · `summary.csv`와 같은 요약 · `--note` |
| `images/` · `road_mask/` · `road_view/` · `preprocessed/` · `review_sheet.jpg` | 자동 (`--no-images`면 없음) | 결과 그림 · 노면 마스크 · 검출기 입력 영상 |
| [`experiments.md`](experiments.md) | **사람** (실험마다 한 줄) | 바꾼 것 · 왜 · 결정 규칙 · 결과 · 판정 · run 이름 |

**`summary.csv`의 범위** (`scopes()`): `전체` · `그룹:blur` · `그룹:local_illumination` · `그룹:normal` (사진 이름이 `labels/split_rdd.csv`에 있을 때 — 흐림 · 국소 조도 · 정상) · `폴더:<하위 폴더>` (하위 폴더가 둘 이상일 때)

**`summary.csv`의 칸** (`summarize()`)

| 칸 | 뜻 |
|---|---|
| `scope` | 범위 |
| `n_images` · `n_detected` · `n_gt_images` | 사진 수 · 검출까지 간 사진 수 · 정답 있는 사진 수 (정확도는 정답 있는 사진만) |
| `n_unverified_mask` · `n_blur_rt` | 검증 안 된 후보 마스크로 계속한 사진 수 · 아주 흐림 판정 수 |
| `{kind}_gt` · `{kind}_hit` · `{kind}_fake` · `{kind}_cands` | 정답 수 · 맞힘 · 가짜 · 후보 수 (`kind` = `crack` · `pothole`, in50) |
| `{kind}_recall` · `{kind}_precision` · `{kind}_f1` | `prf_in50` (2절) |
| `{kind}_fake_per_img` · `{kind}_frag_per_hit` | 가짜/장 (가짜 ÷ 정답 있는 사진 수) · 조각/맞힘 ((후보 − 가짜) ÷ 맞힘) |
| `{kind}_precision_hit` | 이전 계산 P (맞힘 ÷ (맞힘 + 가짜)) |
| `cands_per_img` · `cand_box_area_per_img` · `edges_per_img` · `keypoints_per_img` · `road_ratio_mean` · `pre_ms_per_img` · `det_ms_per_img` · `total_ms_per_img` | 측정값 평균 (판정에는 안 씀) |

**`results.csv`의 칸** (사진 한 장)

| 칸 | 뜻 |
|---|---|
| `image` · `folder` · `dataset` · `group` · `viewpoint` · `width` · `height` | 사진 정보 (`group` = `split_rdd.csv`, `viewpoint` = `labels/viewpoint_rdd.csv` — 결과를 나눠 보는 분석용) |
| `status` · `mask_check` · `unverified_mask` · `road_ratio` · `sharpness` · `blur_judge` · `gamma` · `corrections_applied` · `det_scale` | 처리 상태와 판정 |
| `n_crack` · `n_pothole` · `n_all` · `cand_box_area` | 후보 수 · 후보 박스 면적 합 (원본 px) |
| `edges` · `keypoints` · `road_px_det` · `pre_ms` · `det_ms` · `total_ms` | 측정값 · 처리 시간 |
| `has_gt` · `clean` | 정답이 있나 · 정답이 있고 손상 0개인가 |
| `{kind}_n_gt` · `{kind}_n_cut` | 정답 수 · 잘린 정답 수 (원본 좌표로 채점하므로 0) |
| `{kind}_{iou50 · iou30 · in50}_{tp · fp · fn}` | 기준별 맞힘 · 가짜 · 놓침 |

- `final-result/`는 올리지 않는다. 판정에 쓴 실행은 run 이름을 [`experiments.md`](experiments.md)에 적는다

## 6. 지금 기준 숫자 (원본 저장소 기록)

| 설정 | 내용 | 개발 R · P · F1 | 개발 가짜/장 · 조각/맞힘 | 테스트 R · P · F1 |
|---|---|---|---|---|
| `baseline` | 기준선 — 수업 기술만 (감마 + 가우시안 + Canny) | 0.278 · 0.108 · 0.156 | 5.2 · 1.7 | 0.255 · 0.111 · 0.155 |
| `old_main` | 이전 주 검출기 (Black-hat + 양쪽 확인 + 잇기) | 0.360 · 0.166 · 0.227 | 5.5 · 2.3 | 0.328 · 0.165 · 0.220 |
| `p2bd` | 검출기 최종 (ROI auto · 보정 없음 · Hessian 찾기 · 텐서 전파 · 노면 단서 · 깊이) | 0.571 · 0.286 · 0.381 | 9.3 · 4.9 | 0.482 · 0.261 · 0.339 |
| `gf3` | `p2bd` + 가이드 필터 (전처리 제안 — opencv-contrib 필요) | 0.555 · 0.334 · 0.417 | 6.3 · 4.3 | 0.482 · 0.310 · 0.377 |
| `b_road` (B) | 노면 마스크 구성 (10/10) = 팀원 전처리 v1.1 + 연결 + 도로 위 판정 + `p2bd` | 0.555 · 0.330 · 0.414 | 7.3 · 4.9 | — |
| **`t_c10_post` (F)** | **균열 최종 구성 (10/11)** = B + 질감 줄이기(가우시안 σ0.8 → 막대 닫힘 7px) + CLAHE 1.0 (노면 마스크 안에만) | **0.587 · 0.355 · 0.442** | **6.4 · 4.6** | (아직 안 씀 — 동결 뒤 한 번) |

10/10 지표 고정 때 저장된 결과로 다시 채점한 값 (검출은 다시 안 돌림). 이전 계산 P는 원본 `runs.csv`의 `crack_precision_hit`.

- **최종 (균열 + 포트홀) = final0이 돌리는 구성**: 10/12에 흐림 대응(노면 선명도 < 2.76인 사진만 언샤프 3.0 · σ3 → CLAHE)과 포트홀 2단계를 더했다. 깨끗한 개발 563장의 균열 결과는 F 그대로 (아주 흐림 판단 4장). 포트홀 찾기를 주변 노면 고리 비로 바꾼 뒤 (`run_20261012_151714`): 균열 그대로 · 포트홀 R 0.415 · P 0.044 · 가짜 2.02개/장
- final0은 합성 사진 8장으로 원본 경로와 같은 박스가 나오는 것만 확인했다. **RDD 데이터로는 아직 다시 재지 않았다** ([`README.md`](../README.md)). 다시 재면 위 최종 숫자와 같은지 먼저 본다
- 포트홀 1차 방식은 모든 설정에서 개발 R ≈ 0.01 · 가짜 약 4.7개/장

## 7. 테스트 세트 (`rdd_test` 241장)

- 설정을 고르는 데 쓰지 않는다. 다 정한 뒤 **확인**에만
- final0에서는 `rdd_test`와 `rdd`(RDD 전체 804장 = 개발 563 + 테스트 241)가 `--allow-rdd-test` 없이는 멈춘다 (`run_final.py`의 `PROTECTED`)
```bash
python run_final.py --dataset rdd_test --allow-rdd-test --note "…"   # 설정을 고정한 뒤 한 번만
```
- 이 막음은 데이터 **이름**에만 걸린다. RDD 사진 폴더(`data/RDD2020_train/train/img`)를 경로로 주면 막히지 않고 테스트 사진이 섞이므로, RDD는 반드시 이름으로 돌린다
- 쓸 때마다 [`experiments.md`](experiments.md)의 "테스트 세트 사용 기록"에 순서 · 설정 · 결과를 적고, 보고서에 몇 번째 사용인지 밝힌다
- 지금까지 2번 사용 (G2, GF3) + 보고용 단계별 기여 표 1번 (원본 기록 기준)
- 10/10 지표 고정 때 저장된 테스트 결과를 새 계산으로 **다시 채점만** 했다 (새로 돌리지 않음, 설정 고르기에 안 씀 — 사용 횟수에 넣지 않음)
