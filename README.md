# CV — PBL 모듈 1: 저품질 도로 영상의 손상(균열·포트홀) 후보 추출

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
```

→ `outputs/pipeline/run_<시각>/` 에 `results.csv`(사진×조건×검출기), `summary.csv`(조건×검출기×그룹, F1 95% 범위 포함), `compare.csv`(기준 조건 `--reference` 대비 F1 차이와 95% 범위), `images/`(결과 박스, 정답은 초록)
RDD는 **dev 70% / test 30%로 나눠 쓴다** (`labels/split_rdd.csv`, 기준·규칙은 [`labels/README.md`](labels/README.md)). test를 보고 값을 고치면 test가 아니게 되므로, 실험은 `rdd_dev`로 한다.
제공 13장·촬영분은 정답 CSV가 `labels/provided.csv`, `labels/captured.csv`에 생기면 자동으로 precision·recall·F1까지 계산한다 ([`labels/README.md`](labels/README.md)).

**평가 지표** — 개선 판정은 박스 단위 **F1** 하나로 한다 (`src/evaluate.py`). 정답 판정 기준 3개를 모두 기록한다:
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
