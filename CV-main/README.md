# 도로 손상 검출 시스템 — 전처리 + 검출

도로 영상에서 **자동으로 도로 영역(Road Mask)을 찾고 보정한 뒤, 그 도로 위의 균열 · 포트홀 후보를 검출**한다.
고전 영상처리만 사용한다 (AI · 딥러닝 없음).

```text
입력 영상
 → [전처리 src/preprocessing]  자동 Road Mask(격자 특징 · 다중 시드 · Region Growing · 정리) → PASS / FAIL 검증
                               → (FAIL이면 수동 마스크 또는 후보 마스크) → analysis_mask → 품질 측정
                               → 조건부 Gamma (어두울 때, 후보 γ 중 적응 선택) · Gaussian (설정상 끔)
 → [연결 src/road_detection]   도로 영역만 잘라 긴 변 1024로 맞춤 (검출기가 가정하는 배율)
 → [검출 src/detect.py]        균열 · 포트홀 후보 검출
 → [연결]                      후보 박스가 도로 위인지 판정(박스 면적 중 도로 비율 ≥ 0.5) · 원본 좌표로 되돌림
 → 결과: 보정 영상 · 도로 마스크 · 후보 목록(CSV/JSON) · 결과 그림
```

## 실행 (PowerShell, 저장소 맨 위 폴더)

```powershell
# 한 장
py src/run_road_detection.py --input data/RDD2020_train/train/img/Japan_000015.jpg
# 폴더 · 데이터셋 (rdd_test는 막혀 있음, 최종 확인 때만 --allow-rdd-test)
py src/run_road_detection.py --input data/captured
py src/run_road_detection.py --dataset rdd_dev --limit 20
# 자동 마스크가 FAIL인 사진은 건너뛰기 (기본은 검증 안 된 후보 마스크로 검출)
py src/run_road_detection.py --dataset rdd_dev --limit 20 --on-fail skip
# 수동 마스크 만들기 · 사용 / FAIL이면 편집기 열기
py src/preprocessing/edit_road_mask.py --image data/RDD2020_train/train/img/China_Drone_000785.jpg
py src/run_road_detection.py --dataset rdd_dev --images China_Drone_000785.jpg --manual-mask-dir outputs/preprocessing/manual_masks
py src/run_road_detection.py --input <사진> --edit-failed
# 최종 검출기 설정 (opencv-contrib-python 필요)
py src/run_road_detection.py --input <사진 또는 폴더> --detection-config configs/detection_p2bd.json
# 전처리만 (검출 없이)
py src/preprocessing/run_preprocess.py --dataset rdd_dev --limit 20
# 테스트
py -m unittest discover -s tests
```

## 출력 (`outputs/road_detection/run_<시각>/`)

| 파일 | 내용 |
|---|---|
| `images/<id>/result.jpg` | 원본 위에 도로 마스크(초록) · 도로 위 균열(빨강) · 포트홀(파랑) · 도로 밖 후보(회색) |
| `images/<id>/detections.json` | 후보마다 종류 · 도로 위 여부 · 도로 비율 · 원본 좌표 박스 · 검출기 좌표 박스 · 형태값, 좌표 변환 정보 |
| `images/<id>/processed_image.png` · `road_mask.png` · `analysis_mask.png` | 전처리 결과 (원본 크기 · 좌표, 도로 = 255) |
| `images/<id>/metadata.json` | 전처리(시드 · 검증 · 품질 · Gamma/Gaussian) + 검출 기록 |
| `summary.csv` | 이미지당 한 줄: 상태 · 마스크 판정 · 도로 위 균열/포트홀 수 · 도로 밖 후보 수 · 시간 |
| `detections.csv` | 후보당 한 줄 |
| `run_summary.json` · `run_config.json` · `review_sheet.jpg` | 집계 · 실제 설정 · 결과 그림 모음 |

자동 마스크가 FAIL이고 수동 마스크가 없으면 기본은 **검증 안 된 후보 마스크로 계속 검출**하고 `unverified_mask = True`로 표시한다
(`--on-fail skip`이면 건너뛰고 `skipped_manual_required`로 기록).

## 구성

| 경로 | 역할 |
|---|---|
| `src/preprocessing/` | 전처리 전부: 자동 Road Mask · 검증 · 수동 마스크 편집기 · analysis_mask · 품질 측정 · Gamma · Gaussian · 검출기 입력 변환, 실행기 `run_preprocess.py`(전처리만) · `edit_road_mask.py`(수동 마스크 편집). 로직 · 수치 설명은 [`src/preprocessing/README.md`](src/preprocessing/README.md) |
| `src/detect.py` · `src/roadcue.py` | 검출: D0 / D1 검출기, 노면 단서 거르기 |
| `src/road_detection/` · `src/run_road_detection.py` | 전처리 → 검출 연결과 실행기 |
| `src/paths.py` · `src/data.py` | 경로 · 한글 경로 입출력 · 입력 이미지 목록 (RDD 분할) |
| `configs/preprocessing.json` | 전처리 설정 v1.1 (잠정값 포함) |
| `configs/detection_default.json` | 기본 검출 설정 D1v (opencv-contrib 불필요) |
| `configs/detection_p2bd.json` | 최종 검출기 설정 D1hv (opencv-contrib 필요) |
| `data/` · `labels/` | 입력 데이터 · RDD 분할 (rdd_test 보호에 사용) |
| `tests/` | 전처리 · 연결 시스템 테스트 |

## 검출 설정 (`configs/detection_*.json`)

```json
{
  "detector": "D1v",                 // D0 / D1 + v(양쪽 확인) · l(조각 잇기) · h(Hessian 선 찾기)
  "valley_ratio": 0.35,
  "set": {},                          // detect.py DEFAULT_CFG 키 덮어쓰기
  "integration": {
    "input_mode": "crop_to_road",     // 도로 외접 영역만 잘라 검출 / "full" = 영상 전체
    "crop_margin": 16, "long_side": 1024,
    "min_road_overlap": 0.5,          // 후보 박스 면적 중 도로 비율 기준
    "on_fail": "use_candidate"        // 마스크 FAIL: 검증 안 된 후보 마스크로 검출 / "skip" = 건너뜀
  }
}
```

## 환경

`pip install -r requirements.txt`. 이 PC(Python 3.14.7, OpenCV 4.10.0)에는 `cv2.ximgproc`가 없어
`detection_p2bd.json`은 실행 전에 안내 메시지와 함께 멈춘다. 그 설정을 쓰려면 opencv-contrib-python을 설치한다.
OpenCV가 headless 빌드라 창 기능이 없으므로 수동 마스크 편집기는 matplotlib(TkAgg)를 쓴다.

## 성능 (2026-10-10 측정, 참고)

- 자동 Road Mask: 사람이 그린 정답 도로 마스크 36장(rdd_dev)에서 평균 IoU 0.59 [95% 0.50~0.67], 고정 사다리꼴 0.43.
- 균열 검출 (rdd_dev 563장, 검출기 D1v, in50): 전처리 v1.1 Precision 0.179 · Recall 0.371 · F1 0.241 · 가짜 2.26개/장
  (전처리 없이 영상 전체: 0.062 · 0.276 · 0.101 · 5.49). 포트홀 Recall 0. rdd_test로는 재지 않았다.
- v1.1 변경(Gaussian 끔 · 적응형 Gamma · FAIL도 검출)은 dev 진단으로 정했다. 진단 자료와 v1.0 설정은 아래 보관 폴더에 있다.

## 정리 기록 (2026-10-10)

전처리와 검출 외의 부분은 삭제하지 않고 저장소 밖으로 옮겼다 (각 폴더의 `MANIFEST_sha256.json`). 되돌리려면 같은 상대 경로로 옮겨 오면 된다.
- `../CV-main-removed-20261010/` (1차, 148개): 이전 전처리, 평가 · 실험 도구, 분석 스크립트, 실험 기록, 이전 검출 실험 설정, 사람이 그린 정답 도로 마스크 36장
- `../CV-main-removed-20261010-b/` (2차, 12,469개): 검출 정확도 채점(`evaluate_road_detection.py` · `road_detection/accuracy.py`),
  후보 지표 `preprocessing/seed_agreement.py`, 쓰이지 않던 `visualize.py`, 시점 라벨 `labels/viewpoint_rdd.csv`, 해당 테스트,
  `outputs/`의 실행 · 진단 결과 전부 (전처리 진단 `preprocessing_eval_20261010/` 포함)
