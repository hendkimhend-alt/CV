> **이 저장소에는 두 버전이 함께 있습니다** (main 머지 2026-10-08, 두 README를 모두 그대로 남김)
> - **`CV-main/`** — FINAL 버전 (고정 사다리꼴 ROI + FINAL 전처리 + D1 검출기). 설명은 바로 아래 첫 번째 README이며, 그 안의 경로(`src/...`)는 **`CV-main/` 기준**입니다.
> - **맨 위 `src/`** — sj/dev 버전 (ROI auto · 보정 단계 켜고 끄기 · 조명 펴기 · 새 검출기). 설명은 아래 두 번째 README입니다.
> - 📏 **성능을 재고 비교하는 팀 공통 방법 → [`EVALUATION.md`](EVALUATION.md)** (무엇을 바꾸든 이 방법으로 재고, 결과는 `results/`에 남김)
> - `CV-main/` 버전의 명령(`py src/run_final.py` 등)은 **`CV-main/` 폴더 안에서** 실행해야 합니다. 저장소 맨 위에서 치면 맨 위 `src/`가 실행됩니다.

---

# CV — 도로 균열·포트홀 후보 검출

고정 사다리꼴 ROI 안의 도로 이미지에서 균열·포트홀 후보를 검출하고 이미지와 CSV로 저장한다.
모든 실행 파일의 전처리는 **FINAL 설정**을 사용한다. 기본 검출기는 D1이다.

## 1. 현재 처리 흐름

```text
입력 이미지(BGR)
    → 사다리꼴 ROI 마스크 생성
    → ROI 외접 영역 자르기·긴 변 1024 리사이즈
    → 선택적 MSR
    → ROI 내부 품질 측정
    → 선택적 Gamma
    → 선택적 Gaussian
    → 최종 품질 측정
    → D1 균열·포트홀 후보 검출
    → 검출 이미지·CSV·실행 설정 저장
```

현재 설정의 `msr.enabled`, `gamma.enabled`, `gaussian.enabled`는 모두 `false`다.
따라서 기본 실행은 **사다리꼴 ROI·리사이즈·품질 측정 후 검출**하며, 이 세 보정은 적용하지 않는다.
보정이 생략되면 동일한 영상의 품질 측정값을 재사용한다.

기존 P0/P1/P1+ 설정과 실행 로직은 주석으로 보존했다. 실행 옵션으로는 FINAL만 허용한다.
MSR·Gamma·Gaussian·CLAHE·Unsharp 연산 함수는 남아 있으며, CLAHE·Unsharp는 현재 FINAL 흐름에 연결하지 않는다.
검출 로직과 검출 파라미터는 FINAL 통합 과정에서 변경하지 않았다.

## 2. 파일별 역할

| 파일·폴더 | 역할 |
|---|---|
| [src/final_preprocessing_config.json](src/final_preprocessing_config.json) | 공통 FINAL 설정. ROI·리사이즈·MSR/Gamma/Gaussian 활성화 여부와 후보 값을 저장 |
| [src/adaptive_preprocess.py](src/adaptive_preprocess.py) | FINAL 처리 순서, 적용 판단, 품질값·마스크·처리 시간 반환. GT와 검출기에 접근하지 않음 |
| [src/preprocess.py](src/preprocess.py) | 공통 기하·보정 연산과 공개 `preprocess()` 인터페이스. 공개 함수는 FINAL 실행으로 연결 |
| [src/roi.py](src/roi.py) | 사다리꼴 마스크, 크기 변환된 마스크, 유효 내부 영역, 박스·마스크 교집합 계산 |
| [src/metrics.py](src/metrics.py) | ROI 내부 품질 지표 9개 측정 |
| [src/detect.py](src/detect.py) | D0/D1 검출과 후보의 형태 계산·종류 분류·후처리 |
| [src/data.py](src/data.py) | 입력 목록, RDD dev/test 분할, 정답 박스 로딩 |
| [src/evaluate.py](src/evaluate.py) | GT 좌표 변환과 종류별 hit/miss·Precision·Recall 계산 |
| [src/visualize.py](src/visualize.py) | 검출 후보 박스 그리기 |
| [src/paths.py](src/paths.py) | 데이터·출력 경로와 한글 경로 이미지 입출력 |
| [src/run_pipeline.py](src/run_pipeline.py) | FINAL 전처리 → 검출 → GT 평가 → 결과·집계 저장 |
| [src/run_final.py](src/run_final.py) | GT 없이 FINAL 전처리 → D1 검출 → 이미지별·후보별 결과 저장 |
| [src/run_detect.py](src/run_detect.py) | `run_pipeline.py`의 공통 실행기로 연결. 별도 P0 실행은 주석으로 보존 |
| [src/A.py](src/A.py) | 검출 없이 FINAL 전처리 이미지·품질 CSV·그룹별 그래프 저장 |
| `labels/` | RDD 분할 CSV와 데이터별 GT CSV |
| `tests/`, `backups/` | 기능 검증과 비교용 원본 자료. 백업 5개 파일은 테스트에서 사용 |

## 3. 고정 사다리꼴 ROI

활성 좌표는 `src/final_preprocessing_config.json`의 `geometry.roi` 한 곳에서 정의한다.
`preprocess.py`의 기하 기본 설정도 같은 JSON에서 읽는다. 주석에 남아 있는 과거 설정은 실행하지 않는다.

```json
{
  "type": "trapezoid",
  "top_y_ratio": 0.60,
  "top_left_x_ratio": 0.075,
  "top_right_x_ratio": 0.725,
  "bottom_left_x_ratio": 0.05,
  "bottom_right_x_ratio": 0.95,
  "bottom_y_ratio": 1.00
}
```

꼭짓점 순서는 왼쪽 위 `(0.075, 0.60)` → 오른쪽 위 `(0.725, 0.60)` → 오른쪽 아래 `(0.95, 1.00)` → 왼쪽 아래 `(0.05, 1.00)`이다.
비율에 원본 너비−1·높이−1을 곱해 반올림하고, `cv2.fillConvexPoly`로 마스크를 만든다.
외접 직사각형을 자른 뒤 종횡비를 유지하며 긴 변을 1024로 맞춘다. 축소는 AREA, 확대는 CUBIC, 마스크는 NEAREST를 사용한다.
사다리꼴 밖은 검정으로 채우고 검출·품질 측정에서 제외한다. 원근 변환이나 사다리꼴 워핑은 하지 않는다.

## 4. 실행 환경과 입력

Windows / Python 3.14.7 환경에서 OpenCV 4.10.0, NumPy 2.4.4, Matplotlib 3.10.8로 실행을 검증했다.
의존성은 [requirements.txt](requirements.txt)에 고정돼 있다. OpenCV 창을 사용하지 않으므로 `opencv-python-headless`를 설치한다.
같은 환경에 다른 OpenCV 배포판을 함께 설치하지 않는다.

프로젝트 폴더에서 가상 환경을 준비한다.

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

이하 명령의 `py` 대신 `.\.venv\Scripts\python`을 사용하면 해당 가상 환경에서 실행한다.
`A.py`의 한글 그래프에는 맑은 고딕·나눔고딕 등 지원되는 한글 글꼴이 필요하다.

```text
data/
├── README.md
├── provided/                  제공 이미지
├── captured/                  직접 촬영 이미지
└── RDD2020_train/train/
    ├── img/                   RDD 원본 이미지
    └── ann/                   <이미지 파일명>.json 정답

labels/
├── split_rdd.csv              RDD dev/test 분할
├── viewpoint_rdd.csv          촬영 관점 라벨
└── <데이터 이름>.csv          제공·촬영 이미지 GT가 있으면 배치
```

입력 배치는 [data/README.md](data/README.md), 분할 설명은 [labels/README.md](labels/README.md)를 참고한다.
RDD 분할은 dev 563장·test 241장이다. 현재 개발 실행 예시는 `rdd_dev`만 사용한다.
`viewpoint_rdd.csv`는 보존된 라벨 자료이며 현재 실행기는 이를 읽지 않는다.

RDD GT는 Supervisely JSON에서 균열 3종을 `crack`, 포트홀을 `pothole`로 읽고 기타 손상은 제외한다.
제공·촬영 이미지 GT CSV는 `image,x,y,w,h,type` 형식이다.
정답 파일이 없으면 평가 실행도 검출은 수행하지만 해당 이미지의 GT 평가값은 생성하지 않는다.
`run_final.py`와 `A.py`는 GT 파일을 읽지 않는다. 기본 RDD 입력에는 이미지와 분할 CSV가 필요하다.

데이터·출력을 다른 위치에 두려면 실행 전에 환경변수를 지정한다. 데이터 내부 구조는 위와 같아야 한다.

```powershell
$env:CV_DATA_DIR = "D:\CV_data"
$env:CV_OUTPUT_DIR = "D:\CV_outputs"
```

## 5. 실행 명령

**FINAL 전처리·검출·GT 평가: 먼저 개발 이미지 5장으로 확인**

```powershell
py src/run_pipeline.py --dataset rdd_dev --limit 5 --save-images 5
```

**개발 세트 전체 평가와 검출 이미지 전체 저장**

```powershell
py src/run_pipeline.py --dataset rdd_dev --save-images -1
```

**GT 없이 FINAL/D1 검출 및 후보별 박스 저장**

```powershell
py src/run_final.py --dataset rdd_dev --save-images -1
```

**전처리 이미지·품질 CSV·그래프만 저장**

```powershell
py src/A.py --limit 5
py src/A.py --input "data/captured"
```

`A.py`는 기본적으로 `rdd_dev`를 처리하며, 검출 박스가 없는 전처리 이미지를 저장한다.
`run_detect.py`는 `run_pipeline.py`와 같은 옵션·출력을 사용한다.
평가·검출 실행기의 기본 입력은 `provided`, 기본 검출기는 D1이다.

| 옵션 | 적용 범위·동작 |
|---|---|
| `--dataset provided / captured / rdd_dev` | 평가·검출 실행기의 입력 선택 |
| `--limit N` | 처음 N장만 처리 |
| `--save-images N` | 평가·검출 이미지 저장 수. 기본 5, 0은 저장 생략, −1은 전부 저장. CSV는 처리한 모든 이미지에 대해 저장 |
| `--config 경로.json` | 모든 실행 파일에서 FINAL 형식의 설정 파일 지정 |
| `--detectors D0 D1` | `run_pipeline.py`·`run_detect.py`에서 기존 두 검출기 비교 |
| `--no-keypoints` | `run_pipeline.py`·`run_detect.py`에서 SIFT 특징점 측정 생략 |
| `--conditions FINAL` | 평가 실행기의 활성 조건. P0/P1/P1+는 허용하지 않음 |
| `--output 새폴더` | `run_final.py`의 출력 위치 지정. 기존 내용이 있는 폴더는 거부 |

`run_final.py`의 CLI는 `provided`·`captured`·`rdd_dev`만 허용한다.
평가 실행기는 `rdd`·`rdd_test`도 지원하므로 개발 실험에서는 `rdd_dev`를 지정한다.

## 6. 생성되는 출력

| 실행 파일 | 출력 위치 | 저장 파일 |
|---|---|---|
| `run_pipeline.py`, `run_detect.py` | `outputs/pipeline/run_<시각>/` | `results.csv`, `summary.csv`, `run_config.json`, `images/FINAL_<검출기>/*.png` |
| `run_final.py` | `outputs/final_runtime/<시각>/` | `results.csv`, `detections.csv`, `config.json`, `images/*.png` |
| `A.py` | `outputs/A/run_<시각>/` | `results.csv`, `summary.csv`, `groups_template.csv`, `run_config.json`, `FINAL/` 이미지, `plots/` 그래프 5장 |

### 평가 실행의 CSV

- `results.csv`: 이미지×검출기당 한 행. FINAL 적용 여부·값, 품질 5개, 에지·SIFT 수, 종류별 후보 수·면적, 전처리·검출 시간, GT가 있으면 GT 수·적중 수·정답과 겹친 후보 수를 저장한다.
- `summary.csv`: 검출기·품질 그룹별 평균과 종류별 Precision·Recall을 저장한다. GT 평가값은 그룹의 개수를 합산해 계산한다.
- `run_config.json`: 실제 전처리·검출 설정, 입력 정보, 라이브러리 버전을 저장한다.

### GT 없는 FINAL 실행의 CSV

- `results.csv`: 이미지당 한 행. 품질 9개, MSR/Gamma/Gaussian 적용 여부와 값, 전체 후보 수, 전처리·검출·합산 시간을 저장한다.
- `detections.csv`: 검출 후보당 한 행. 이미지 이름, 종류, `bbox_xywh`를 저장한다. 박스는 **ROI 자르기·리사이즈 후 이미지 좌표**다.
- `config.json`: 실제 FINAL 설정과 검출 설정·검출 코드 해시 등을 저장한다.

`run_final.py`는 GT 정확도와 집계 `summary.csv`를 생성하지 않는다. GT 비교는 평가 실행기를 사용한다.
전처리·검출 시간에는 이미지 읽기·출력 저장이 포함되지 않는다. FINAL 전처리 시간에는 내부 품질 측정 시간이 포함된다.

검출 이미지는 균열 후보를 빨간색, 포트홀 후보를 파란색으로 표시한다.
평가 이미지에는 ROI에 남은 GT를 초록색으로 함께 그린다. 검출 후보가 실제 손상으로 확정됐다는 뜻은 아니다.
출력은 실행별 폴더에 저장되며, 과거 결과 폴더는 새 실행으로 자동 갱신되지 않는다.

## 7. 품질값과 GT 평가 해석

품질 지표는 `gray_mean`, `gray_std`, `block_mean_std_4x4`, `saturation_ratio`, `dark_ratio`,
`bright_ratio`, `hsv_v_mean`, `laplacian_variance`, `noise_sigma`다.
사다리꼴 밖의 검정 패딩과 필요한 필터 경계 응답을 제외해 계산한다.
ROI 내부에 남은 비도로 영역도 측정 대상이므로 품질값이 도로 분할 정확도를 의미하지는 않는다.
노면 질감·실제 균열도 선명도와 노이즈 추정값에 영향을 준다.

자동 품질 그룹은 선택적 MSR 이후·Gamma 이전 품질값으로 분류한다.
Laplacian 분산이 50 미만이면 `blur`, 4×4 블록 밝기 편차가 44 초과이면 `local_illumination`이며 두 태그가 함께 붙을 수 있다.

GT 비교는 같은 종류 후보 박스와 GT 박스가 **양의 면적으로 겹치면 적중**하는 기존 방식이다.
일대일 IoU 매칭이나 mAP 평가는 하지 않는다.
Precision은 정답과 겹친 후보 수÷전체 후보 수, Recall은 적중한 GT 수÷ROI에 남은 GT 수다.
ROI 밖으로 완전히 제외된 GT는 이 Recall의 분모에 들어가지 않는다.
따라서 원본 GT 전체를 분모로 계산한 다른 실험 결과와 동일한 수치로 비교하면 안 된다.

## 8. 테스트와 GitHub 공유

```powershell
py -m unittest discover -s tests
```

FINAL 통합 후 테스트 39개가 통과했고, 각 실행 경로를 실제 `rdd_dev` 이미지 5장으로 확인했다.
CSV·이미지·품질 그래프 저장과 실행 경로 간 FINAL 품질값·D1 후보 수 일치를 검증했다.
이 검증은 실행 동작 확인이며 전체 데이터셋의 성능 개선을 의미하지 않는다.
원본 데이터 없이도 단위 테스트를 실행할 수 있다. `backups/`는 테스트에서 읽으므로 함께 보존한다.

| GitHub에 포함 | GitHub에서 제외 |
|---|---|
| `src/` 전체와 `final_preprocessing_config.json` | `data/`의 원본 이미지·RDD GT JSON |
| `labels/`, `tests/`, 테스트용 `backups/` | `outputs/`의 생성 이미지·CSV·JSON |
| `README.md`, `data/README.md`, `requirements.txt` | `.venv/`, 캐시, 개인 IDE 설정 |
| `.gitignore`, `.gitattributes` | |

**로컬 원본 데이터는 삭제하지 않고 업로드 대상에서 제외한다.**
현재 `.gitignore`는 `data/README.md`를 제외한 데이터와 `outputs/` 등을 Git 추적에서 제외한다.
GitHub 웹에서 파일을 직접 업로드할 때는 `.gitignore`가 자동 필터링하지 않으므로 제외할 파일을 직접 구분해야 한다.
코드만 받은 팀원은 실제 검출 실행에 필요한 입력 데이터를 각자 준비해야 한다.
출력 폴더가 없어도 실행할 수 있으며 프로그램이 새 결과를 생성한다.
`analysis/`는 현재 실행에 필요하지 않다. 자동 후보 탐색·과거 실험 분석 기능은 포함하지 않지만 전처리·품질 측정·검출·CSV 집계는 유지한다.


---

# CV — PBL 모듈 1: 저품질 도로 영상의 손상(균열·포트홀) 후보 추출

## 실행 · 측정 · 기록 가이드 (맨 위 src/)

누가 돌려도 **같은 방법으로 Recall · Precision을 재고, 결과가 자동으로 쌓이게** 하는 명령 모음입니다. 비교 · 판정 규칙의 기준 문서는 [`EVALUATION.md`](EVALUATION.md)입니다.

### 1. 준비 (한 번)

```bash
pip install -r requirements.txt          # Python 3.12 · OpenCV 4.13
# gf3(가이드 필터)만 cv2.ximgproc가 필요 → opencv-python 대신 opencv-contrib-python (같은 버전)을 설치
```
RDD 이미지 · 정답을 `data/RDD2020_train/train/{img,ann}/`에 넣습니다 ([`data/README.md`](data/README.md)). 모든 명령은 **저장소 맨 위 폴더**에서 실행합니다.

### 2. 저장된 설정으로 돌리기 (`--config`)

설정은 [`configs/`](configs/)에 파일로 있습니다. 같은 파일을 쓰면 같은 결과가 나옵니다.

| 설정 | 내용 | 개발 563장 R · P · 가짜/장 | 테스트 241장 R · P |
|---|---|---|---|
| `baseline` | 기준선 — 수업 기술만 (감마 + 가우시안 + Canny) | 0.278 · 0.066 · 5.2 | 0.255 · 0.066 |
| `old_main` | 이전 주 검출기 (Black-hat + 양쪽 확인 + 잇기) | 0.360 · 0.080 · 5.5 | 0.328 · 0.091 |
| **`p2bd`** | **검출기 최종** — Hessian 찾기 · 텐서 전파 · 노면 단서 · 깊이 | **0.571 · 0.075 · 9.3** | **0.482 · 0.077** |
| `gf3` | `p2bd` + 가이드 필터 (전처리 담당 제안) | 0.555 · 0.104 · 6.3 | 0.482 · 0.110 |

```bash
# 빠른 확인 (20장, 1분 안팎)
python src/run_pipeline.py --dataset rdd_dev --config configs/p2bd.json --limit 20 --no-keypoints

# 개발 세트 전체 (약 9분 — 텐서 전파 때문. baseline · old_main은 약 2분)
python src/run_pipeline.py --dataset rdd_dev --config configs/p2bd.json --no-keypoints --bootstrap 0 --note "무엇을 바꿨는지 한 줄"

# 설정 일부만 바꿔서 시험: 명령줄 옵션이 설정 파일보다 우선
python src/run_pipeline.py --dataset rdd_dev --config configs/p2bd.json --set crack_min_length=60 --note "길이 45 → 60"
```
- `--config` 파일의 키 = 명령줄 옵션 이름(`-` 대신 `_`) · `set` = `--set`과 같은 검출 설정 사전. 모르는 키는 실행 전에 오류
- 설정을 새로 만들면 `configs/<이름>.json`으로 저장해 같이 올립니다 (설정 이름이 기록에 남음)
- 결과 폴더: `outputs/pipeline/run_<시각>/` — `summary.csv`(묶음별 지표) · `results.csv`(사진별) · `run_config.json`(실제 설정 · 코드 버전) · `images/`

### 3. 측정 방법 — Recall · Precision (in50)

`src/evaluate.py` · 균열과 포트홀을 따로 잽니다.

| | 정의 |
|---|---|
| **맞힘 (in50)** | 후보 박스 면적의 **절반 이상이 같은 종류 정답 박스 안**이면 그 정답을 맞힘. 정답 하나에 맞힘은 최대 1개 (겹침이 큰 쌍부터 짝짓기) |
| 같은 정답 안의 추가 조각 | 맞힘도 가짜도 아님 (균열을 조각으로 잡는 건 봐주되, 잘게 쪼개 점수를 올리는 건 막음) |
| **가짜** | 어느 정답과도 짝이 안 된 후보 |
| **Recall** | 맞힌 정답 수 ÷ 전체 정답 수 — **ROI 밖으로 잘린 정답도 분모에 넣어 놓침으로 셈** (ROI를 좁혀 Recall을 부풀리지 못하게) |
| **Precision** | 맞힌 후보 수 ÷ (맞힌 후보 + 가짜) |
| **가짜/장** | 가짜 수 ÷ 정답이 있는 사진 수 |

- 합산 방식: 사진마다 맞힘 · 가짜 · 놓침을 센 뒤 **전체를 더해서** 비율을 냅니다 (사진별 평균 아님)
- `summary.csv`의 `crack_in50_recall` · `crack_in50_precision` · `crack_in50_fppi`(= 가짜/장)가 이 값입니다. iou50 · iou30도 같이 기록되지만 판정에는 in50만 씁니다
- 목표: Recall 우선 + Precision 하한 — **달성 R 0.50 · P 0.10** · 도전 R 0.70 · P 0.30 · 지키는 선 P ≥ 0.066
- ⚠️ `CV-main/`의 평가는 "겹치면 적중" + 잘린 정답 제외라 **숫자를 서로 비교하면 안 됩니다**

### 4. 기록 — 자동 + 사람

| 파일 | 누가 | 무엇 |
|---|---|---|
| [`results/runs.csv`](results/runs.csv) | **자동** — 정답이 있는 실행마다 한 줄씩 덧붙음 | 시각 · 실행 이름 · 설정 이름 · 데이터 · 균열/포트홀 정답 수 · 맞힘 · 가짜 · R · P · F1 · 가짜/장 · 코드 버전(커밋, 고친 채 돌리면 `+수정`) · `--note` 메모 |
| [`results/experiments.md`](results/experiments.md) | **사람** — 실험 하나 끝날 때 | 무엇을 바꿨나 · 왜 · 미리 정한 결정 규칙 · 결과 · 결정(✅/❌) · 실행 이름 |

- 기록을 남기기 싫은 시험 실행은 `--no-log`. `--limit`으로 일부만 돌린 것도 기록되며 `limit` 칸에 장수가 남습니다
- 공유: 실험 후 `results/runs.csv` · `results/experiments.md` · 새 `configs/*.json`을 커밋해서 올립니다 (`outputs/`는 올리지 않음)

### 4-1. 찾기를 바꿨으면 강한 기준부터 맞추기

```bash
python analysis/calibrate.py --config configs/p2bd.json --set guided_filter='{"r":4,"eps":"var","eps_scale":2.0}' --save configs/새이름.json
```
강한 흔적 총수를 기준과 같게 하는 `line_hi_abs`를 찾아 새 설정에 넣습니다 (이유 · 언제 필요한지는 [`EVALUATION.md`](EVALUATION.md) 3절 ③).

### 5. 두 실행 비교 — 의미 있는 차이인가

```bash
python analysis/compare_runs.py 기준=run_20261008_005056 새것=run_<시각>
```
- 같은 사진끼리 짝지어 1000번 다시 뽑아 Recall · Precision 차이의 **95% 범위**를 냅니다. 범위가 0을 포함하지 않으면 ✱
- 전체 · 흐림 · 국소 조도 · 정상 그룹별로 나옵니다. 포트홀은 `--kind pothole`
- 실행 안에 조건 × 검출기가 여러 개면 `--pick none/D1hv`처럼 하나를 고릅니다
- 여러 값 중 고를 때는 `--split`: 튜닝 394 · 검증 169로 나눠 보여 줍니다 (튜닝에서 고르고 검증에서 확인)

### 6. 데이터 세트 규칙

| 이름 | 장수 | 쓰는 곳 |
|---|---|---|
| `rdd_dev` | 563 | 실험 · 값 고르기는 여기서만 |
| `rdd_tune` / `rdd_val` | 394 / 169 | `rdd_dev`를 나눈 것 — 튜닝에서 고르고 검증에서 확인 ([`labels/split_rdd_dev.csv`](labels/split_rdd_dev.csv)) |
| `rdd_test` | 241 | **최종 확인용** — 결과를 보고 설정을 고치면 안 됨. 쓸 때마다 `results/experiments.md`의 "테스트 세트 사용 기록"에 적기 |
| `provided` · `captured` | 13 · 촬영분 | 정답 CSV(`labels/<이름>.csv`)가 생기면 자동으로 평가 |

---


```
① 사진 고치기 (개발 A)  →  ② 찾기 (개발 B)  →  ③ 재기 (개발 A)
preprocess.py             detect.py           metrics.py · evaluate.py (초안)
                    └──── src/run_pipeline.py 로 한 번에 ────┘
```

## 폴더

| 폴더 | 내용 | git |
|---|---|---|
| `src/` | 파이프라인 코드 — A: `preprocess.py` `metrics.py` `A.py` / B: `detect.py` `visualize.py` `run_detect.py` / 공통: `paths.py` `data.py`(정답 로더) `evaluate.py`(hit/miss) `run_pipeline.py`(전체 실행) | 올림 |
| `analysis/` | 데이터 분석·평가 스크립트 (13장 실측, RDD 분석, 색 채널, 검출기 평가, 계획서 그림) | 올림 |
| `labels/` | 정답 박스 CSV (문서 담당) | 올림 |
| `results/` | **팀이 공유할 결과** (CSV·요약 md). 실행 출력 중 남길 것만 골라서 옮긴다 | 올림 |
| `data/` | 데이터 — 각자 넣는다 ([`data/README.md`](data/README.md)) | 안 올림 |
| `outputs/` | 실행할 때마다 생기는 출력 (이미지·CSV·그래프) | 안 올림 |

## 환경 준비

Python 3.12에서 확인 (OpenCV 4.13, numpy 2.5). Windows·macOS 둘 다 같은 명령으로 동작한다.

```bash
pip install -r requirements.txt
```

데이터는 `data/` 아래에 넣는다 → [`data/README.md`](data/README.md). 다른 위치에 있으면 `CV_DATA_DIR` 환경변수로 지정.

## 실행 (저장소 맨 위 폴더에서)

**전체 (A → B → 평가)**

```bash
python src/run_pipeline.py                     # 제공 13장 × P0/P1/P1+ × D0/D1
python src/run_pipeline.py --dataset rdd_dev   # RDD 개발 세트 563장 — 값(ROI·기준값·파라미터) 고르기는 여기서만
python src/run_pipeline.py --dataset rdd_test  # RDD 테스트 세트 241장 — 최종 설정이 정해진 뒤 한 번만
python src/run_pipeline.py --dataset rdd       # RDD 804장 전체 — 정답 있음 → precision·recall·F1 (약 3분)
python src/run_pipeline.py --dataset rdd_dev --limit 50 --no-keypoints   # 빠르게 확인
python src/run_pipeline.py --dataset rdd_dev --roi auto   # 노면 영역(ROI): bottom_half(기본) / full / bottom_<N> / auto(사진마다 도로 시작 높이)
python src/run_pipeline.py --dataset rdd_dev --roi auto --conditions none gamma gaussian gamma+gaussian   # 보정 단계를 골라 켜기
```

→ `outputs/pipeline/run_<시각>/` 에 `results.csv`(사진×조건×검출기), `summary.csv`(조건×검출기×그룹, F1 95% 범위 포함), `compare.csv`(기준 조건 `--reference` 대비 F1 차이와 95% 범위), `images/`(결과 박스, 정답은 초록)
RDD는 **dev 70% / test 30%로 나눠 쓴다** (`labels/split_rdd.csv`, 기준·규칙은 [`labels/README.md`](labels/README.md)). test를 보고 값을 고치면 test가 아니게 되므로, 실험은 `rdd_dev`로 한다.
제공 13장·촬영분은 정답 CSV가 `labels/provided.csv`, `labels/captured.csv`에 생기면 자동으로 precision·recall·F1까지 계산한다 ([`labels/README.md`](labels/README.md)).

**전처리 조건** — `P0_reference`(보정 없음) · `P1`(gamma+gaussian) · `P1+`(gamma+clahe+unsharp+gaussian) 묶음, 또는 켤 단계를 `+`로 이어 직접 지정(`flatten` · `gamma` · `clahe` · `unsharp` · `gaussian`, `none`). 단계는 입력 순서와 상관없이 항상 flatten → gamma → clahe → unsharp → gaussian 순서로 적용, unsharp는 흐린 사진에만.
`flatten`(조명 펴기 = 그림자 처리): 밝기(L)를 큰 closing으로 만든 "조명 배경"으로 나눠 그림자 · 밝기 차이를 고르게 함. 배경 크기는 `--flatten-ksize`(기본 61).
실험 원칙: 이미 정한 단계는 켜고, 아직 안 정한 단계는 끈다.

**평가 지표** — 지금 판정은 위 [가이드 3절](#3-측정-방법--recall--precision-in50) (in50 Recall 우선 + Precision 하한). 아래는 처음 정한 방법의 기록: 박스 단위 **F1** (`src/evaluate.py`). 정답 판정 기준 3개를 모두 기록한다:
`iou50` IoU > 0.5 (표준, RDD 대회와 같은 정의) / `iou30` IoU > 0.3 / `in50` 후보 면적의 절반 이상이 정답 박스 안 (같은 정답 안의 추가 조각은 제외).
기준선에서 "조건 간 차이를 오차 범위보다 크게 구분하는 기준 중 가장 엄격한 것"을 골라 이후 실험에 고정한다.
오차 범위는 사진 단위 부트스트랩(`--bootstrap 1000`, `src/stats.py`)으로 구하고, 콘솔 마지막 표에 판정 기준별로 "의미 있는 차이" 수가 나온다.

**단계별**

```bash
python src/A.py                          # A: RDD 804장 × P0/P1/P1+ 전처리 + 품질 지표 → outputs/A/run_.../
python src/A.py --input data/provided    # A: 제공 13장
python src/A.py --limit 10               # A: 처음 10장만

python src/run_detect.py                 # B: 제공 13장 × D0/D1 검출 → outputs/detect/
python analysis/eval_detector.py         # B: RDD 박스 단위 검출 평가
```

코드에서 연결:

```python
from paths import PROVIDED_DIR, imread
from preprocess import preprocess
from detect import detect

img = imread(PROVIDED_DIR / "United_States_005996.jpg")
fixed = preprocess(img, {"condition": "P1+"})   # A: 노면 영역 + 1024 + 보정, BGR
dets = detect(fixed)                              # B: [{bbox, type, area, elong, ...}]
```

## 규칙

- **경로는 `src/paths.py`에서만** 가져온다. 코드에 개인 경로(`C:\Users\...`, `/Users/...`)를 쓰지 않는다
- 이미지 읽기·쓰기는 `paths.imread` / `paths.imwrite` (Windows 한글 경로 대응)
- 실행 결과는 `outputs/`에 쌓이고, 팀과 공유할 것만 `results/<이름_데이터_날짜>/`로 옮겨 올린다
- 줄바꿈은 `.gitattributes`로 LF 통일 (Windows에서도 diff가 깨지지 않게)
