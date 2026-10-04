# data/ — 데이터 위치 (git에 올리지 않음)

각자 아래 구조로 넣는다. 폴더 복사 또는 바로가기(심볼릭 링크) 둘 다 된다.

```
data/
├── provided/                  제공 13장 (PBL 모듈 1 dataset의 jpg)
├── captured/                  직접 촬영분
└── RDD2020_train/train/
    ├── img/                   RDD 이미지 804장
    └── ann/                   RDD 라벨 json
```

다른 위치에 두고 싶으면 환경변수로 지정한다 (코드 수정 불필요).

- Windows (PowerShell): `$env:CV_DATA_DIR = "C:\Users\USER\Desktop\CV\img"`
- macOS (zsh): `export CV_DATA_DIR=~/somewhere/data`

이 경우에도 그 폴더 안 구조는 위와 같아야 한다.
