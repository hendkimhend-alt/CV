# labels/ — RDD 분할 · 시점 · 정답 CSV (final0)

> 원본 저장소 `labels/README.md`를 final0에 맞게 고친 것. 파일 세 개는 원본과 같다 (복사).

## 들어 있는 파일

| 파일 | 내용 |
|---|---|
| `split_rdd.csv` | RDD 804장 → dev 563 (70%) / test 241 (30%). 열: `image,source,group,has_damage,n_crack,n_pothole,split` |
| `split_rdd_dev.csv` | dev 563장을 다시 나눈 것 → tune 394 (값 고르기) / val 169 (확인). 열은 `split_rdd.csv`와 같음 |
| `viewpoint_rdd.csv` | 사진별 시점. 열: `image,viewpoint,note` — `far`(위쪽에 하늘·원경) / `road_full`(화면 거의 전체가 노면) / `ambiguous` |

- 분할 기준: **출처 × 손상 유무 × 품질 그룹** 조합마다 7:3 (seed 고정). 원본 저장소 `analysis/make_split.py`로 한 번 만들고 고정했다 — 다시 만들지 않는다
- `group` 열(`blur` 흐림 · `local_illumination` 그림자 · `normal` 정상)은 `run_final.py`의 `summary.csv`에서 그룹별 요약(`그룹:blur` 등)에 쓴다. 검출은 이 값을 보지 않는다
- 이미지는 들어 있지 않다 — `data/RDD2020_train/train/img/`의 RDD 이미지에 이 목록이 그대로 적용된다
- `viewpoint_rdd.csv`는 썸네일을 보고 분류한 것 → 사람 검수 필요 (특히 `ambiguous` 8장). 결과 CSV의 `viewpoint` 칸에만 쓴다

사용:

```bash
python run_final.py                                   # rdd_dev 563장
python run_final.py --dataset rdd_tune                # 394장 (rdd_val 169장)
python run_final.py --dataset rdd_test --allow-rdd-test   # 241장
```

규칙: 값(기준값 · 파라미터)은 **dev에서만** 고른다. test는 최종 설정이 정해진 뒤 한 번만 돌린다.
`rdd` · `rdd_test`는 `--allow-rdd-test` 없이는 실행되지 않는다. RDD 이미지 폴더를 경로로 직접 주면 분할을 무시하고
804장 전부(테스트 포함)를 돌리므로, RDD는 이름(`rdd_dev` 등)으로 실행한다.

## 정답 CSV를 더할 때 (제공 · 촬영 · 그 밖의 사진 폴더)

`labels/<데이터 이름>.csv`를 만들면 `run_final.py`가 정확도를 계산한다. 데이터 이름은 `provided` · `captured`,
또는 폴더 경로로 실행할 때 그 폴더의 이름이다 (예: `--dataset data/내사진` → `labels/내사진.csv`).
정답을 찾는 전체 순서와 JSON 형식은 `README.md`의 "정확도(정답)를 찾는 순서".

열: `image,x,y,w,h,type`

- `image`: 파일 이름 (예: `United_States_005996.jpg`)
- `x,y,w,h`: **원본 이미지 좌표**의 박스 (왼쪽 위 x, y, 너비, 높이, 픽셀)
- `type`: 이름에 `pothole`이 들어가면 포트홀, 그 밖은 균열 (세로 · 가로 · 거북등 균열은 모두 `crack`)
- 손상이 여러 개면 한 줄씩. CSV에 없는 사진은 "손상 없음"으로 본다

```csv
image,x,y,w,h,type
United_States_005996.jpg,240,300,80,400,crack
Japan_001959.jpg,120,700,160,120,pothole
```

(위 두 줄은 형식 예시 — 실제 정답이 아니다.) RDD2020은 자체 라벨(`data/RDD2020_train/train/ann/`)을 쓴다.
