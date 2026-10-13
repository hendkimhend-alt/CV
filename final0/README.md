# final0 — 최종 구성 한 곳에 모음

2차 계획서(10/12)의 최종 구성을 실행하는 코드와 설정만 저장소 여러 곳에서 **복사**해 모은 폴더다.
원본 파일은 그대로 두었다. 이 폴더만으로 실행된다 (데이터는 `data/`에 따로 넣는다).

개발 563장 기록 (results/experiments.md): 균열 R 0.587 · P 0.355 · 가짜 6.4개/장 · 포트홀 39/94 · R 0.415 · P 0.044 · 가짜 2.0개/장

## 처리 흐름

```text
사진 → [전처리] 자동 노면 마스크 (FAIL이면 검증 안 된 후보 마스크로 계속)
     → [연결] 노면 외접 영역 자르기(여백 16px) · 긴 변 1024
     → [보정] 노면 마스크 안에만
          선명한 사진:   가우시안 σ0.8 → 막대 닫힘 7px(12방향) → CLAHE 1.0
          아주 흐린 사진(노면 선명도 < 2.76): 언샤프 3.0 · σ3 → CLAHE 1.0
     → [균열]   Hessian 찾기 → 텐서 전파 → 버리기 (길이 45 · 양쪽 확인 · 노면 단서 · 깊이)
       [포트홀] 주변 노면 고리보다 1.8배 거친 곳 → 색 · 어두운 부분 · 모양 · 노면 색 · 높이로 버리기
     → [도로 위 판정] 후보 박스의 50% 이상이 노면이면 남김 → 원본 좌표
     → [채점] 정답이 있으면 in50
```

## 실행

```bash
pip install -r requirements.txt
python run_final.py                       # rdd_dev 563장 → final-result/run_<시각>/
python run_final.py --limit 20            # 빠른 확인
python run_final.py --dataset provided    # 제공 13장
python run_final.py --dataset <폴더>       # 아무 사진 폴더 (하위 폴더까지 · 폴더별 요약도 냄)
python run_final.py --dataset <폴더> --gt <정답 파일>       # 정답 파일 직접 지정 (.json · .csv)
python run_final.py --dataset rdd_test --allow-rdd-test   # 테스트 세트 — 설정 고정 뒤 한 번만
```

결과는 실행할 때마다 `final-result/run_<시각>/`에 새로 저장된다 (앞 결과를 덮어쓰지 않음).
사진 파일 이름은 `하위폴더__이름`으로 저장해, 폴더가 다르고 이름이 같은 사진도 덮어쓰지 않는다.

| 파일 | 내용 |
|---|---|
| `images/` | 결과 그림 — 노면 마스크(초록 칠 · 노란 테두리) · 균열(빨강) · 포트홀(파랑) · 정답(흰색, 있을 때). 위에 사진 이름 · 개수 · 맞힌 수/정답 수 |
| `road_mask/` | 노면 마스크 (흰색 = 노면, 원본 크기 PNG) |
| `road_view/` | 원본에서 노면 밖을 어둡게 한 그림 (계획서 그림 2 ②) |
| `preprocessed/` | 검출기에 실제로 들어간 영상 (노면 영역 자르기 · 긴 변 1024 · 질감 줄이기 · CLAHE 또는 언샤프) |
| `review_sheet.jpg` | 결과 그림 모음 (처음 12장) |
| `results.csv` | 사진별 — 아래 표 |
| `summary.csv` | 전체 · 그룹(RDD 흐림 · 그림자 · 정상) · 폴더(하위 폴더가 둘 이상일 때)별 정확도와 측정값 평균 |
| `detections.json` | 사진별 박스 (원본 좌표) |
| `run_config.json` | 쓴 설정 · 정답 출처 · `summary.csv`와 같은 요약 |

`results.csv`의 주요 칸:

| 칸 | 뜻 |
|---|---|
| `status` · `mask_check` · `unverified_mask` | 처리 상태 · 자동 노면 마스크 검증(PASS/FAIL) · FAIL이라 검증 안 된 마스크로 계속했나 |
| `road_ratio` | 사진에서 노면 마스크가 차지하는 비율 |
| `sharpness` · `blur_judge` | 노면 선명도 · 판정 (`blur_rt` = 아주 흐림 < 2.76 → 언샤프 쪽 보정) |
| `n_crack` · `n_pothole` · `cand_box_area` | 후보 수 · 후보 박스 면적 합 (원본 px) |
| `edges` · `keypoints` | 보정 뒤 노면 안 에지 수(Canny 100/200) · SIFT 특징점 수 — 계획서 3-6 표의 '전처리 후'와 같은 방법 |
| `pre_ms` · `det_ms` · `total_ms` | 처리 시간 (읽기 ~ 전처리 · 보정 ~ 검출 · 합) |
| `crack_in50_tp/fp/fn` 등 | 정답이 있을 때 채점 (in50 · iou30 · iou50) |

`summary.csv`의 정확도: `*_recall` · `*_precision` · `*_f1` · `*_fake_per_img`(가짜/장) · `*_frag_per_hit`(조각/맞힘) — 계획서와 같은 계산.

화면에는 진행 상황, 측정값 평균, 정확도(전체 · 그룹 · 폴더별)가 나온다.
CSV만 원하면 `--no-images`, 다른 곳에 저장하려면 `--out <폴더>`.

### 정확도(정답)를 찾는 순서

정답이 없으면 정확도 칸은 비고, 검출 · 그림 · 측정값만 나온다.

1. `--gt <파일>` 로 직접 지정
2. RDD(`rdd_dev` 등 이름으로 실행): `data/RDD2020_train/train/ann/<이미지 이름>.json`
3. `labels/<데이터 이름>.csv` (폴더로 실행하면 데이터 이름 = 폴더 이름)
4. 사진 폴더 안의 `annotations.json` 또는 `annotations.csv`
5. 제공 13장(`provided`): `data/road_damage_annotated/annotations.json` (저장소에서 쓰던 초안 정답)

정답 파일 형식:
- `.json`: `{"images": [{"source": "사진.jpg", "boxes": [{"class": "crack", "bbox_xyxy": [x1, y1, x2, y2]}]}]}`
- `.csv`: 열 `image, x, y, w, h, type`
- 종류 이름에 `pothole`이 들어가면 포트홀, 그 밖은 균열. 정답 파일에 없는 사진은 "손상 없음"으로 채점한다.

데이터 위치는 `data/README.md` 구조를 따른다. 다른 곳에 있으면 환경변수 `CV_DATA_DIR`로 지정한다.

## PBL 데이터셋 실행 (`run_final_PBL.py`)

`run_final.py`의 처리 함수(`prepare` · `detect_image`)와 설정 파일을 그대로 불러 쓰고, PBL 사진 폴더에만 돌린다.
정답이 없으므로 정확도는 계산하지 않는다. `run_final.py`는 바꾸지 않았다.

```bash
python run_final_PBL.py                       # data/PBL module 1 - dataset/ (하위 폴더까지)
python run_final_PBL.py --input <다른 폴더>
```

결과: `final-result-PBL/run_<시각>/` (실행마다 새 폴더, `final-result/`와 따로) — 사진마다 다섯 장

| 파일 | 내용 |
|---|---|
| `<사진 이름>_1_original.jpg` | 원본 그대로 복사 (확장자는 원본과 같음) |
| `<사진 이름>_2_road_mask.jpg` | 원본 위에 노면 마스크만 초록 칠 · 노란 테두리 (박스 없음) |
| `<사진 이름>_3_preprocessed.jpg` | 전처리된 영상 = 검출기에 실제로 들어간 영상 (노면 영역 자르기 · 긴 변 1024 · 노면 안 보정). `run_final.py`의 `preprocessed/`와 같은 것 |
| `<사진 이름>_4_crack_bbox.jpg` | 원본 위에 균열 박스(빨강)만 (초록 칠 · 포트홀 없음) |
| `<사진 이름>_5_mask_bbox.jpg` | 원본 위에 초록 노면 칠 + 균열(빨강) · 포트홀(파랑) 박스. `run_final.py`의 `images/`와 같은 그림 |

전처리(노면 마스크)에 실패한 사진은 2 · 3번에 실패 상태를 적고, 4 · 5번은 박스 없이 저장한다. 읽을 수 없는 파일은 건너뛰고 끝에 알린다.

## 한 장 테스트 (`run_final_test.py`)

`data/test/`에 사진 한 장을 넣고 실행하면, 같은 처리 뒤 원본 위에 박스만 그린 그림 한 장을 저장한다
(균열 빨강 · 포트홀 파랑, 노면 마스크 칠 없음, 정확도 없음). 처리 함수는 `run_final.py`의 것을 그대로 쓴다.

```bash
python run_final_test.py                        # data/test/ 의 사진
python run_final_test.py --input <사진 파일>     # 다른 사진 한 장
```

결과: `final-result-test/run_<시각>/<사진 이름>_bbox.jpg` (실행마다 새 폴더).
폴더에 사진이 없으면 안내 후 멈추고, 여러 장이면 모두 처리한다.

## 문서

원본 저장소의 문서를 final0에 맞게 고쳐 넣었다 (원본 문서는 그대로 둠).

| 문서 | 내용 | 원본 |
|---|---|---|
| [`src/preprocessing/README.md`](src/preprocessing/README.md) | 전처리 파트 — 자동 노면 마스크(①~⑮) · 검출기 입력 자르기 · 보정 · 전처리 결과가 나오는 곳 | `CV-main/src/preprocessing/README.md` (= 저장소 맨 위 `README.md`) |
| [`docs/pothole.md`](docs/pothole.md) | 포트홀 검출 — 정의 · 찾기(주변 고리 비) · 검증 ①~⑤ · 설정 · 기록 | `docs/pothole.md` |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | 성능 비교 방법 — in50 · 지표 정의 · 판정 규칙 · 테스트 세트 규칙 · 결과 파일 칸 | `EVALUATION.md` |
| [`docs/experiments.md`](docs/experiments.md) | 실험 기록 (지나간 기록 — 본문은 원본 그대로, 경로 · 명령은 원본 저장소 기준) | `results/experiments.md` |
| [`labels/README.md`](labels/README.md) | RDD 분할 · 시점 · 정답 CSV 형식 | `labels/README.md` |
| [`data/README.md`](data/README.md) | 데이터 넣는 위치 | `data/README.md` (그대로) |

성능 수치는 모두 원본 저장소에서 잰 기록이다. final0으로 RDD를 다시 재지는 않았다.

## 파일과 원본 위치

| final0 | 원본 | 바꾼 것 |
|---|---|---|
| `run_final.py` | `analysis/run_road_pipeline.py` | 최종 설정만 돌리도록 새로 정리 (실험 옵션 `--extra` · `--calibrate` · `--degrade` · `--cache` · `--det-set` 뺌, 처리 순서 · 계산은 같음) · 결과 그림 · 노면 마스크 · 전처리 영상 저장, 측정값(에지 · 특징점 · 후보 면적 · 시간, `analysis/measure_prepost.py`와 같은 방법), 정답 파일 찾기, 그룹 · 폴더별 요약 추가 · 지표 계산은 `src/run_pipeline.py`의 `run_log_metrics`와 같음 · `results/runs.csv`에는 기록하지 않고 `summary.csv`에 남김 |
| `configs/preprocessing.json` | `CV-main/configs/preprocessing.json` | 없음 |
| `configs/detection_final.json` | `CV-main/configs/detection_final.json` | 없음 (= `detection_p2bd.json` + 강한 기준 132.17 + 보정 = `configs/final_crack.json`) |
| `configs/pothole_final.json` | `configs/final.json`의 `"pothole"` | 새 파일 (값 그대로) |
| `src/detect.py` | `CV-main/src/detect.py` | 없음 (균열 검출기) |
| `src/roadcue.py` | `CV-main/src/roadcue.py` (`src/roadcue.py`와 내용 같음) | 없음 |
| `src/pothole.py` | `src/pothole.py` | 포트홀 기본값 `POTHOLE_DEFAULTS` 추가 (원래 `src/detect.py` DEFAULT_CFG의 `pothole_*` 키, 값 그대로) · 설명 주석 |
| `src/paths.py` | `src/paths.py` | 설명 주석만 (기준 폴더 = final0) |
| `src/data.py` | `src/data.py` | 정답 파일(.json · .csv)을 읽는 `load_gt_file` 추가 (CSV 읽기는 원래 코드를 이 함수로 옮김) |
| `src/evaluate.py` | `src/evaluate.py` | 없음 |
| `src/preprocessing/` | `CV-main/src/preprocessing/` | 없음 · 수동 마스크 편집기(`mask_editor` · `edit_road_mask`)와 전처리 단독 실행기(`run_preprocess`)는 뺌 |
| `src/road_detection/` | `CV-main/src/road_detection/` | `config.py`의 opencv-contrib 없음 안내 문구만 (final0에 없는 `detection_default.json` 대신 설치 방법) · `render.py` = 결과 그림 |
| `labels/` | `labels/` | 없음 (RDD 분할 · 시점) |

이름이 겹치던 모듈(`detect` · `roadcue` · `paths` · `data`)은 위처럼 한 벌만 남겨, 원본 실행기처럼
`sys.path`를 바꿔 가며 두 `src`를 번갈아 부르지 않는다.

## 원본과 같은지 확인 (10/13)

합성 도로 사진 8장(균열 · 포트홀 · 그림자 · 아주 흐림 1장 포함)을 원본 경로(`analysis/run_road_pipeline.py`의
함수 + `configs/final.json`)와 `run_final.py`에 넣어 비교 — 8장 모두 보정된 검출기 입력 영상(해시)과
최종 박스(균열 · 포트홀)가 같았다. 결과 그림 · 노면 마스크 · 측정값 · 정확도 저장을 더한 뒤에도 다시 비교해 같았다
(그림 · 측정은 검출이 끝난 뒤에 따로 계산하므로 검출 결과에 영향 없음). RDD 데이터로는 아직 다시 재지 않았다.

## 들어 있지 않은 것

- 5장 개선 계획 (노면 마스크에 그림자 더하기 · 가이드 필터) — 아직 최종 구성이 아님
- 실험 · 분석 스크립트 (`analysis/`), 1차 방식 코드, 테스트
