# data/

`python run.py`를 인자 없이 실행하면 `data/test/` 안의 사진을 처리한다. 사진 한 장만 넣어도 되고 여러 장도 된다.

데이터를 다른 곳에 두려면 환경변수 `CV_DATA_DIR`로 지정한다. 그 폴더 안의 `test/`를 읽는다.

- Windows (PowerShell): `$env:CV_DATA_DIR = "D:\road\data"`
- macOS / Linux: `export CV_DATA_DIR=~/road/data`

참고로 이 `data/` 안에 하위 폴더가 하나도 없으면, 위 폴더(세 단계까지)에 있는 `data/`를 대신 찾아본다.
여러 버전을 한 상위 폴더에서 같은 데이터로 돌리던 구조 때문에 남아 있는 동작이라, 보통은 `data/test/`에 넣어 쓰면 된다.
