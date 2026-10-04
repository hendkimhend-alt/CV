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
