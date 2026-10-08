# labels/ — 정답 박스 (git에 올림)

| 파일 | 대상 |
|---|---|
| `provided.csv` | 제공 13장 (`data/provided/`) |
| `captured.csv` | 직접 촬영분 (`data/captured/`) |

열: `image,x,y,w,h,type`

- `image`: 파일 이름 (예: `United_States_005996.jpg`)
- `x,y,w,h`: **원본 이미지 좌표**의 박스 (왼쪽 위 x, y, 너비, 높이, 픽셀)
- `type`: `crack` 또는 `pothole` (세로·가로·거북등 균열은 모두 `crack`)
- 손상이 여러 개면 한 줄씩. CSV에 없는 사진은 "손상 없음"으로 본다

```csv
image,x,y,w,h,type
United_States_005996.jpg,240,300,80,400,crack
Japan_001959.jpg,120,700,160,120,pothole
```

RDD2020은 자체 라벨(`data/RDD2020_train/train/ann/`)을 쓴다.

## RDD 분할 · 시점 (값 고르기용 dev / 마지막 확인용 test)

| 파일 | 내용 |
|---|---|
| `split_rdd.csv` | RDD 804장 → dev 563 (70%) / test 241 (30%). 열: `image,source,group,has_damage,n_crack,n_pothole,split` |
| `viewpoint_rdd.csv` | 사진별 시점. 열: `image,viewpoint,note` — `far`(위쪽에 하늘·원경) / `road_full`(화면 거의 전체가 노면) / `ambiguous` |

- 분할 기준: **출처 × 손상 유무 × 품질 그룹** 조합마다 7:3 (seed 고정). 현재 분할 목록은 고정하여 사용하며 삭제된 분석 도구를 실행할 필요가 없다
- 사용: `python src/run_pipeline.py --dataset rdd_dev` / `--dataset rdd_test`
- 규칙: 값(ROI·기준값·파라미터)은 **dev에서만** 고른다. test는 최종 설정이 정해진 뒤 한 번만 돌린다
- 이미지는 git에 없다 — 각자 `data/`의 RDD 이미지에 이 목록이 그대로 적용된다
- `viewpoint_rdd.csv`는 Claude가 썸네일을 보고 분류한 것 → 사람 검수 필요 (특히 `ambiguous` 8장). 시점은 분할 기준에 넣지 않았지만 dev/test 비율 0.29~0.30으로 맞는 것을 확인
