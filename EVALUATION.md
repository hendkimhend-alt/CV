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
| 지표 | **Recall · Precision · 가짜/장** | 아래 2절 |
| 비교 기준 | **지금 최종 설정** `configs/p2bd.json` (6절 표) | 무엇을 바꿨든 이것보다 나아졌는지를 본다 |
| 비교 방법 | 같은 사진끼리 짝지은 95% 범위 (`analysis/compare_runs.py`) | 우연인 차이를 걸러냄 |
| 흔적 양 | 찾기를 바꾸면 **강한 흔적 총수를 기준과 같게** 맞춘 강한 기준값으로 (`analysis/calibrate.py`) | "더 많이 뽑아서" 오른 것과 "방법이 좋아서" 오른 것을 가르려고 |

> ⚠️ `CV-main/`의 평가는 "겹치면 적중" + ROI 밖 정답 제외라 **이 문서의 숫자와 비교할 수 없다.**
> `CV-main/`의 전처리를 비교하려면, 그 전처리를 맨 위 `src/`의 설정(`--conditions` · `--roi` 등)으로 넣어 같은 방법으로 잰다.

## 2. 무엇을 재나 — 정의

| 지표 | 뜻 | 계산 |
|---|---|---|
| **Recall** | 진짜 손상 중 찾은 비율 | 맞힌 정답 수 ÷ 전체 정답 수 (ROI 밖으로 잘린 정답도 분모에 넣어 **놓침**으로 셈) |
| **Precision** | 내놓은 후보 중 진짜 비율 | 맞힌 후보 수 ÷ (맞힌 후보 + 가짜) |
| **가짜/장** | 사진 한 장당 가짜 수 | 가짜 수 ÷ 사진 수 |

- **맞힘 (in50)**: 후보 박스 면적의 **절반 이상이 같은 종류 정답 박스 안**. 정답 하나에 맞힘은 최대 1개 (겹침이 큰 쌍부터). 같은 정답 안의 추가 조각은 맞힘도 가짜도 아님
- **가짜**: 어느 정답과도 짝이 안 된 후보
- 사진마다 센 뒤 **전체를 더해서** 비율을 낸다 (사진별 평균 아님)
- 균열과 포트홀은 따로 잰다. 판정은 균열 기준 (포트홀은 따로 표시)

**목표** — Recall 우선 + Precision 하한

| | Recall | Precision |
|---|---|---|
| 지키는 선 (기준선 수준) | ≥ 0.278 | ≥ 0.066 |
| 달성 목표 | 0.50 | 0.10 |
| 도전 목표 | 0.70 | 0.30 |

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
기준   R 0.571  P 0.075  F1 0.133  맞힘 425  가짜/장  9.3
새것   R 0.555  P 0.104  F1 0.176  맞힘 413  가짜/장  6.3   ΔR -0.016 (-0.049~+0.016)   ΔP +0.029 (+0.019~+0.039)✱
```
괄호 = 차이의 95% 범위 · **✱ = 범위가 0을 포함하지 않음 = 우연으로 보기 어려운 차이**. 그룹별(흐림 · 국소 조도 · 정상)도 같이 나온다
- 여러 값(예: 길이 45 · 60 · 80) 중 하나를 고를 때는 `--split`: 같은 결과를 **튜닝 394장 · 검증 169장**으로 나눠 보여 준다 → 튜닝에서 고르고 검증에서 확인 (4절)
```bash
python analysis/compare_runs.py 기준=run_A 후보=run_B --split
```

**⑥ 판정하고 기록** — 4절 표대로 판정 → `results/experiments.md`에 결과 · 판정 · run 이름. 채택하면 `configs/<새 이름>.json`으로 저장하고 6절 표를 고친다

## 4. 판정 — 개선인가

**Precision 하한 P_min** = 기준 설정의 Precision이 이미 넘은 목표선. 지금 기준 `p2bd`는 P 0.075 → **P_min = 0.066** (지키는 선). 기준이 `gf3`처럼 0.10 이상이면 **P_min = 0.10** (달성 목표를 지켜야 함 — 10/8 튜닝 · 검증 실험에서 쓴 규칙)

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
- **여러 값 중 하나를 고를 때**: 튜닝 394장에서 위 표로 후보를 고르고 → 검증 169장에서 "Recall이 기준보다 떨어지지 않고 Precision ≥ P_min"이면 ✅ (`compare_runs.py --split`). 같은 사진으로 고르고 확인하면 점수가 부풀려진다
- 참고: 10/6 이전 초반 실험(E0 ~ E3)은 F1으로 판정했다 — 그 뒤 목표가 Recall 우선으로 바뀌면서 이 표로 통일

## 5. 기록 — 어디에 무엇이 남나

| 파일 | 누가 | 무엇 |
|---|---|---|
| [`results/runs.csv`](results/runs.csv) | **자동** (정답 있는 실행마다) | 시각 · run 이름 · 설정 이름 · 데이터 · 정답 수 · 맞힘 · 가짜 · R · P · F1 · 가짜/장 (균열 · 포트홀) · 코드 버전 · `--note` |
| [`results/experiments.md`](results/experiments.md) | **사람** (실험마다 한 줄) | 바꾼 것 · 왜 · 결정 규칙 · 결과 · 판정 · run 이름 |
| [`configs/`](configs/) | 사람 | 채택한 설정 파일 |

- 올릴 것: `results/runs.csv` · `results/experiments.md` · 새 `configs/*.json` (`outputs/`는 올리지 않음)
- 기록하지 않을 시험 실행은 `--no-log`

## 6. 지금 기준 숫자 (비교 대상)

| 설정 | 내용 | 개발 R · P · 가짜/장 | 테스트 R · P |
|---|---|---|---|
| `baseline` | 기준선 — 수업 기술만 (감마 + 가우시안 + Canny) | 0.278 · 0.066 · 5.2 | 0.255 · 0.066 |
| `old_main` | 이전 주 검출기 (Black-hat + 양쪽 확인 + 잇기) | 0.360 · 0.080 · 5.5 | 0.328 · 0.091 |
| **`p2bd`** | **지금 기준 = 검출기 최종** (ROI auto · 보정 없음 · Hessian 찾기 · 텐서 전파 · 노면 단서 · 깊이) | **0.571 · 0.075 · 9.3** | **0.482 · 0.077** |
| `gf3` | `p2bd` + 가이드 필터 (전처리 제안 — opencv-contrib 필요) | 0.555 · 0.104 · 6.3 | 0.482 · 0.110 |

포트홀은 모든 설정에서 개발 R ≈ 0.01 · 가짜 약 4.7개/장 (1차 방식 그대로).

## 7. 테스트 세트 (`rdd_test` 241장)

- 설정을 고르는 데 쓰지 않는다. 다 정한 뒤 **확인**에만
- 쓸 때마다 `results/experiments.md`의 "테스트 세트 사용 기록"에 순서 · 설정 · 결과를 적고, 보고서에 몇 번째 사용인지 밝힌다
- 지금까지 2번 사용 (G2, GF3) + 보고용 단계별 기여 표 1번
