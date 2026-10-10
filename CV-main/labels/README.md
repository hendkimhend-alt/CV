# labels/ — RDD 분할

| 파일 | 내용 |
|---|---|
| `split_rdd.csv` | RDD 804장 → dev 563 (70%) / test 241 (30%). 열: `image,source,group,has_damage,n_crack,n_pothole,split` |
| `split_rdd_dev.csv` | dev 563장 → tune 394 / val 169 |

- 분할 기준: 출처 × 손상 유무 × 품질 그룹 조합마다 7:3 (seed 고정). 한 번 만들고 고정했으며 다시 만들지 않는다
- 사용: `--dataset rdd_dev` · `rdd_tune` · `rdd_val` (`src/data.py`)
- 규칙: 값은 dev에서만 고른다. test(`rdd_test`)는 최종 설정이 정해진 뒤 한 번만 돌린다 — 실행기가 기본으로 막는다
- RDD2020 정답 주석은 `data/RDD2020_train/train/ann/`에 있다
