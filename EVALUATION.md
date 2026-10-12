# 성능 비교 방법 (팀 공통)

> 누가 무엇을 바꾸든 **이 문서의 방법 하나로** 재고 비교한다. 그래야 "무엇을 바꿔서 얼마나 좋아졌나"를 서로 같은 숫자로 말할 수 있다.
> 전처리 · 검출 · ROI 어느 쪽을 바꿔도 같다. 이 방법과 다르게 잰 숫자는 비교 표에 넣지 않는다.

---

## 1. 고정하는 것 — 바꾸지 않는다

| 항목 | 고정값 | 이유 |
|---|---|---|
| 코드 | 저장소 **맨 위 `src/`** (`src/run_pipeline.py`) | 측정 코드가 하나여야 숫자가 같은 뜻 |
| 데이터 | **`rdd_dev` 563장** (RDD 개발 세트) | 실험 · 비교는 여기서만. `rdd_test`는 아래 7절 |
| 측정 | **in50** 판정 (`src/evaluate.py`) · 균열 | 아래 2절 |
| 지표 | **Recall · Precision · F1** + 가짜/장 · 조각/맞힘 (10/10 고정) | 아래 2절 |
| 비교 기준 | **균열 최종 구성 F** (10/11 — B + 질감 줄이기(가우시안 σ0.8 → 막대 닫힘 7px) + CLAHE 1.0, `analysis/run_road_pipeline.py`) · 검출기만 바꿀 때는 `configs/p2bd.json` (6절 표) | 무엇을 바꿨든 이것보다 나아졌는지를 본다 |
| 비교 방법 | 같은 사진끼리 짝지은 95% 범위 (`analysis/compare_runs.py`) | 우연인 차이를 걸러냄 |
| 흔적 양 | 찾기를 바꾸면 **강한 흔적 총수를 기준과 같게** 맞춘 강한 기준값으로 (`analysis/calibrate.py`) | "더 많이 뽑아서" 오른 것과 "방법이 좋아서" 오른 것을 가르려고 |

> **B 구성 재기**: `python analysis/run_road_pipeline.py --note "…"` — `CV-main/`의 실행 코드 · 설정(`CV-main/configs/preprocessing.json` · `detection_p2bd.json`)으로 돌리고 **이 문서의 in50으로 채점**해 `runs.csv`에 남긴다 (설정 칸 `b_road`). B 위에 하나만 바꿀 때는 `--pre-config` · `--det-config`로 바꾼 설정 파일을 주고 `--config-name`을 새 이름으로.
> **최종 구성 F 재기** (10/11, 설정 칸 `t_c10_post`):
> `python analysis/run_road_pipeline.py --final-config configs/final_crack.json --config-name f_final --note "…"`
> — 보정 순서와 맞춘 강한 기준(132.172, B의 강한 흔적 총수에 맞춘 값)이 `configs/final_crack.json`에 들어 있다 (캐시 없이 처음부터 돌려도 됨, `--cache`를 주면 빨라짐). 순서 = 노면 마스크 → 감마(어두운 사진만) → 노면 영역 자르기 · 긴 변 1024 → [노면 안에만] 가우시안 → 막대 닫힘 → CLAHE → Hessian 선 점수 → 텐서 전파 → 중심선 → 이중 임계값 → 버리기 (길이 · 양쪽 확인 · 노면 단서 · 깊이) → 도로 위 판정
> **흐림 대응 (10/12)**: `configs/final_crack.json` · `configs/final.json`의 보정에 추가 — 노면 선명도 < 2.76 (아주 흐림)인 사진만 질감 처리를 건너뛰고 언샤프 3.0 · σ3 → CLAHE. 보정 코드는 `CV-main/src/preprocessing/detector_corrections.py` 하나 (이 실행기와 `CV-main/src/run_road_detection.py --detection-config configs/detection_final.json`이 같은 함수 — 개발 20장 균열 박스 모두 같음 확인). 깨끗한 개발 563장 결과는 그대로 (아주 흐림 판단 4장) · 효과는 섞인 흐림 평가셋으로 잰다 (`--degrade '{"kind": "mixed", "frac": 0.25, "ks": [2, 3.5]}' --det-set line_hi_abs=132.17221069335938`, 아주 흐린 76장 R 0.047 → 0.299 · `results/experiments.md` 8단계)
> **최종 (균열 + 포트홀)**: `python analysis/run_road_pipeline.py --final-config configs/final.json --config-name final --note "…"` — 같은 전처리 영상에서 균열은 위 검출기, 포트홀은 포트홀 2단계(`src/pothole.py`, `docs/pothole.md`)로 따로 찾고 도로 위 판정은 공통 (10/12, `run_20261012_140359`: 균열 R 0.587 · P 0.355 그대로 · 포트홀 R 0.340 · P 0.035 · 가짜 1.94개/장) → 포트홀 찾기를 주변 노면 고리 비로 바꾼 뒤 (`pothole_find=ring`, `run_20261012_151714`): 균열 그대로 · 포트홀 R 0.415 · P 0.044 · 가짜 2.02개/장
> ⚠️ `CV-main/`의 자체 평가(`src/run_final.py` 등)는 "겹치면 적중" + ROI 밖 정답 제외라 **이 문서의 숫자와 비교할 수 없다.**
> `CV-main/`의 전처리를 비교하려면, 그 전처리를 맨 위 `src/`의 설정(`--conditions` · `--roi` 등)으로 넣어 같은 방법으로 잰다.

## 2. 무엇을 재나 — 정의

| 지표 | 뜻 | 계산 |
|---|---|---|
| **Recall** | 진짜 손상 중 찾은 비율 (정답 쪽에서 셈) | 맞힌 정답 수 ÷ 전체 정답 수 (ROI 밖으로 잘린 정답도 분모에 넣어 **놓침**으로 셈) |
| **Precision** | 내놓은 후보 중 정답 위에 있는 비율 (후보 쪽에서 셈) | 정답 박스 안 후보 수 ÷ 전체 후보 수 = (후보 − 가짜) ÷ 후보 |
| **F1** | 둘의 조화평균 | 2 · P · R ÷ (P + R) |
| **가짜/장** | 사진 한 장당 가짜 수 | 가짜 수 ÷ 사진 수 |
| **조각/맞힘** | 맞힌 정답 하나에 들어간 후보 수 | 정답 박스 안 후보 수 ÷ 맞힌 정답 수 |

- **맞힘 (in50)**: 후보 박스 면적의 **절반 이상이 같은 종류 정답 박스 안**. 정답 하나에 맞힘은 최대 1개 (겹침이 큰 쌍부터). 같은 정답 안의 추가 조각은 맞힘도 가짜도 아님
- **가짜**: 어느 정답 박스 안에도 들어가지 않은 후보
- **Precision을 후보 쪽에서 세는 이유 (10/10 고정)**: 이전 계산(맞힌 정답 ÷ (맞힌 정답 + 가짜))은 맞힘을 **정답 수**로, 가짜를 **조각 수**로 세서 단위가 달랐다. 진짜 균열 하나가 조각 6개로 나오면 1개, 가짜가 조각 4개로 나오면 4개로 세어져 P가 0.07 ~ 0.10으로 지나치게 낮았다. 이제 P는 후보 단위로, R은 정답 단위로 각각 센다 (균열 논문의 픽셀 P · R — 검출 쪽 · 정답 쪽에서 각각 셈 — 을 박스로 옮긴 것). 코드: `src/evaluate.py`의 `prf_in50`
- **허점과 보완**: 정답 안에서 잘게 쪼갤수록 P가 오를 수 있다 → **가짜/장 · 조각/맞힘을 항상 같이 보고**한다. P가 올랐어도 가짜/장이 늘었으면 "가짜가 줄었다"고 쓰지 않는다 (예: `baseline` → `p2bd`는 P 0.108 → 0.286이지만 가짜/장은 5.2 → 9.3, 조각/맞힘 1.7 → 4.9)
- 이전 계산 값은 `runs.csv`의 `precision_hit` · `summary.csv`의 `in50_precision_hit`로 계속 남긴다 (1차 계획서 목표와 잇기 위해)
- 사진마다 센 뒤 **전체를 더해서** 비율을 낸다 (사진별 평균 아님)
- 균열과 포트홀은 따로 잰다. 판정은 균열 기준 (포트홀은 따로 표시)

**목표** — Recall 우선 + Precision 하한

| | Recall | Precision |
|---|---|---|
| 지키는 선 (기준선 수준) | ≥ 0.278 | ≥ 0.108 |
| 달성 목표 | 0.50 | (이전 계산 0.10) |
| 도전 목표 | 0.70 | (이전 계산 0.30) |

- 지키는 선 = 기준선(`baseline`)의 값 — 계산을 바꿔도 규칙은 같아서 새 계산 값으로 다시 잰 것
- Precision 달성 · 도전 목표는 1차 계획서에서 **이전 계산으로 정한 숫자**다. 새 계산으로 숫자를 다시 정하지 않는다 (결과를 보고 목표를 고르면 목표가 의미 없어짐) → `precision_hit`으로 확인

## 3. 절차 — 여섯 단계

**① 기준 결과 준비** — 이미 있으면 그 실행을 쓴다 (`results/runs.csv`에서 `config = p2bd`, `dataset = rdd_dev`인 줄)
```bash
python src/run_pipeline.py --dataset rdd_dev --config configs/p2bd.json --no-keypoints --bootstrap 0 --note "기준"
```

**② 돌리기 전에 기록** — [`results/experiments.md`](results/experiments.md)에 한 줄: 무엇을 바꾸나 · 왜(가설) · **결정 규칙** (4절 표에서 고름)

**③ 찾기에 영향을 주는 것을 바꿨으면 → 강한 기준부터 맞추기**

전처리 · 찾기 · 잇기(가이드 필터, 크기 평균, 텐서 전파, 감마 같은 보정 등)를 바꾸면 Hessian 점수의 크기 자체가 달라진다. 기준값을 그대로 두면 흔적이 더 많이(적게) 뽑혀서 Recall · Precision이 움직인 것인지, 방법이 좋아서인지 가를 수 없다. 그래서 **강한 흔적 총수가 기준(기본 Hessian ≥ 32.5, 흔적 양 약 5.9%)과 같아지는 값**으로 맞춘 뒤 비교한다. 사진만 보고 정답은 보지 않는다.
```bash
python analysis/calibrate.py --config configs/p2bd.json --set guided_filter='{"r":4,"eps":"var","eps_scale":2.0}' --save configs/새이름.json
# → "맞춘 강한 기준 line_hi_abs = …" 출력, 그 값을 넣은 설정 파일 저장 (개발 563장, 텐서 전파가 있으면 약 9분)
python analysis/calibrate.py --config configs/p2bd.json --conditions gamma --save configs/감마.json   # 전처리 단계를 바꿀 때
```
- 버리기 단계만 바꾼 경우(길이 · 노면 단서 · 깊이 등)는 맞출 필요 없음 — 흔적은 그대로
- 그동안 맞춘 값: 32.5 (기본) → 24.87 (크기 평균) → 128.61 (텐서 전파 = `p2bd`) → 3.003 (가이드 필터 = `gf3`)

**④ 하나만 바꿔서 돌리기** — 데이터 · 나머지 설정은 그대로
```bash
# 검출 설정을 바꿀 때
python src/run_pipeline.py --dataset rdd_dev --config configs/p2bd.json --set crack_min_length=60 --no-keypoints --bootstrap 0 --note "길이 45 → 60"
# 전처리 단계를 바꿀 때 (flatten · gamma · clahe · unsharp · gaussian 중 골라 +로) — ③에서 맞춘 설정으로
python src/run_pipeline.py --dataset rdd_dev --config configs/감마.json --no-keypoints --bootstrap 0 --note "감마 추가"
# ROI를 바꿀 때
python src/run_pipeline.py --dataset rdd_dev --config configs/p2bd.json --roi bottom_half --no-keypoints --bootstrap 0 --note "ROI 하단 50%"
```
- 새 기능을 코드에 넣을 때는 **설정으로 켜고 기본은 꺼 둔다** (기존 결과가 바뀌지 않게)
- 빠른 확인은 `--limit 20`, 판단은 **563장 전체**로만

**⑤ 짝지어 비교**
```bash
python analysis/compare_runs.py 기준=<①의 run 이름> 새것=<③의 run 이름>
```
```
기준   R 0.571  P 0.286  F1 0.381  맞힘  425  후보/장  13.0  가짜/장   9.3
새것   R 0.555  P 0.334  F1 0.417  맞힘  413  후보/장   9.5  가짜/장   6.3  ΔR -0.016 (-0.049~+0.016)   ΔP +0.048 (+0.026~+0.073)✱
```
괄호 = 차이의 95% 범위 · **✱ = 범위가 0을 포함하지 않음 = 우연으로 보기 어려운 차이**. 그룹별(흐림 · 국소 조도 · 정상)도 같이 나온다
- 여러 값(예: 길이 45 · 60 · 80) 중 하나를 고를 때는 `--split`: 같은 결과를 **튜닝 394장 · 검증 169장**으로 나눠 보여 준다 → 튜닝에서 고르고 검증에서 확인 (4절)
```bash
python analysis/compare_runs.py 기준=run_A 후보=run_B --split
```

**⑥ 판정하고 기록** — 4절 표대로 판정 → `results/experiments.md`에 결과 · 판정 · run 이름. 채택하면 `configs/<새 이름>.json`으로 저장하고 6절 표를 고친다

## 4. 판정 — 개선인가

**Precision 하한 P_min** = 지키는 선 **0.108** (기준선 수준, 2절). Precision 차이는 가짜/장과 함께 본다 — P ✱ 상승이어도 가짜/장이 늘었으면 개선이라고 쓰지 않는다
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
- **여러 값 중 하나를 고를 때**: 튜닝 394장에서 위 표로 후보를 고르고, 후보 중 **가짜/장이 가장 적은 값 — 조각/맞힘이 기준보다 늘지 않는 것 중** (10/10부터 · P 최고로 고르면 정답 안 조각이 늘어난 값이 뽑힐 수 있음: 막대 닫힘 15px 사례) → 검증 169장에서 "Recall이 기준보다 떨어지지 않고 Precision ≥ P_min"이면 ✅ (`compare_runs.py --split`). 같은 사진으로 고르고 확인하면 점수가 부풀려진다
- 참고: 10/6 이전 초반 실험(E0 ~ E3)은 F1으로 판정했다 — 그 뒤 목표가 Recall 우선으로 바뀌면서 이 표로 통일

## 5. 기록 — 어디에 무엇이 남나

| 파일 | 누가 | 무엇 |
|---|---|---|
| [`results/runs.csv`](results/runs.csv) | **자동** (정답 있는 실행마다) | 시각 · run 이름 · 설정 이름 · 데이터 · 정답 수 · 맞힘 · 가짜 · 후보 수 · R · P · F1 · 가짜/장 · 조각/맞힘 · 이전 계산 P (균열 · 포트홀) · 코드 버전 · `--note` |
| [`results/experiments.md`](results/experiments.md) | **사람** (실험마다 한 줄) | 바꾼 것 · 왜 · 결정 규칙 · 결과 · 판정 · run 이름 |
| [`configs/`](configs/) | 사람 | 채택한 설정 파일 |

- 올릴 것: `results/runs.csv` · `results/experiments.md` · 새 `configs/*.json` (`outputs/`는 올리지 않음)
- 기록하지 않을 시험 실행은 `--no-log`

## 6. 지금 기준 숫자 (비교 대상)

| 설정 | 내용 | 개발 R · P · F1 | 개발 가짜/장 · 조각/맞힘 | 테스트 R · P · F1 |
|---|---|---|---|---|
| `baseline` | 기준선 — 수업 기술만 (감마 + 가우시안 + Canny) | 0.278 · 0.108 · 0.156 | 5.2 · 1.7 | 0.255 · 0.111 · 0.155 |
| `old_main` | 이전 주 검출기 (Black-hat + 양쪽 확인 + 잇기) | 0.360 · 0.166 · 0.227 | 5.5 · 2.3 | 0.328 · 0.165 · 0.220 |
| **`p2bd`** | **지금 기준 = 검출기 최종** (ROI auto · 보정 없음 · Hessian 찾기 · 텐서 전파 · 노면 단서 · 깊이) | **0.571 · 0.286 · 0.381** | **9.3 · 4.9** | **0.482 · 0.261 · 0.339** |
| `gf3` | `p2bd` + 가이드 필터 (전처리 제안 — opencv-contrib 필요) | 0.555 · 0.334 · 0.417 | 6.3 · 4.3 | 0.482 · 0.310 · 0.377 |
| `b_road` (B) | 노면 마스크 구성 (10/10) = 팀원 전처리 v1.1 + 연결 + 도로 위 판정 + `p2bd` (`analysis/run_road_pipeline.py`) | 0.555 · 0.330 · 0.414 | 7.3 · 4.9 | — |
| **`t_c10_post` (F)** | **균열 최종 구성 (10/11)** = B + 질감 줄이기(가우시안 σ0.8 → 막대 닫힘 7px) + CLAHE 1.0 (노면 마스크 안에만) | **0.587 · 0.355 · 0.442** | **6.4 · 4.6** | (아직 안 씀 — 동결 뒤 한 번) |

10/10 지표 고정 때 저장된 결과로 다시 채점한 값 (검출은 다시 안 돌림). 이전 계산 P는 `runs.csv`의 `crack_precision_hit`.

포트홀은 모든 설정에서 개발 R ≈ 0.01 · 가짜 약 4.7개/장 (1차 방식 그대로).

## 7. 테스트 세트 (`rdd_test` 241장)

- 설정을 고르는 데 쓰지 않는다. 다 정한 뒤 **확인**에만
- 쓸 때마다 `results/experiments.md`의 "테스트 세트 사용 기록"에 순서 · 설정 · 결과를 적고, 보고서에 몇 번째 사용인지 밝힌다
- 지금까지 2번 사용 (G2, GF3) + 보고용 단계별 기여 표 1번
- 10/10 지표 고정 때 저장된 테스트 결과를 새 계산으로 **다시 채점만** 했다 (새로 돌리지 않음, 설정 고르기에 안 씀 — 사용 횟수에 넣지 않음)
