# 도로 손상 검출

도로 사진을 넣으면 노면을 자동으로 찾고, 노면 위의 균열과 포트홀에 박스를 그린 사진을 저장한다.
딥러닝 없이 고전 영상처리로만 만들었다.

## 실행

Python 3.9 이상이 필요하다.

1. 코드 폴더로 이동해 라이브러리를 설치한다.

```bash
cd final_version
pip install -r requirements.txt
```

opencv-python이 이미 깔려 있으면 먼저 `pip uninstall opencv-python`을 하고 설치한다.
균열 검출에 opencv-contrib-python에만 있는 기능(cv2.ximgproc)을 쓰는데, 둘을 같이 깔면 충돌한다.

2. `data/test/` 폴더를 만들고 검출할 사진을 넣는다.

```bash
mkdir -p data/test        # Windows: mkdir data\test
```

3. 실행한다.

```bash
python run.py
```

다른 사진이나 폴더를 바로 지정할 수도 있다.

```bash
python run.py 사진.jpg
python run.py 사진폴더              # 하위 폴더까지 전부
python run.py 사진폴더 --out 결과폴더
```

## 결과

`result/run_<시각>/<사진 이름>.jpg`로 저장된다. 실행할 때마다 새 폴더를 만들어서 이전 결과를 덮어쓰지 않는다.

그림에는 자동으로 찾은 노면(초록 칠, 노란 테두리), 균열(빨간 박스), 포트홀(파란 박스)이 그려지고,
왼쪽 위에 사진 이름과 균열, 포트홀 개수가 적힌다.

노면을 찾지 못하면 박스 없이 원본에 `road surface not found`를 적고,
처리 중 오류가 나면 원본에 `processing failed`를 적어 저장한 뒤 다음 사진으로 넘어간다.
읽을 수 없는 파일은 건너뛴다.

## 처리 과정

1. 노면 마스크: 사진을 32px 칸으로 나눠 도로다운 칸을 시드로 고르고, 비슷한 화소로 영역을 넓힌다(Region Growing).
   다듬은 뒤 네 가지 지표로 검증하는데, 검증에 실패해도 그 마스크로 계속 간다(화면에 "검증 안 된 마스크"로 표시).
2. Gamma: 노면이 어두울 때만 노면 안에 건다. 0.9, 0.8, 0.7 순서로 써 보고 어두움이 풀리는 첫 값을 쓴다.
3. 검출기 입력: 노면을 감싸는 영역을 여백 16px로 잘라 긴 변을 1024px로 맞춘다.
4. 보정 (노면 안에만): 선명한 사진은 가우시안(sigma 0.8)과 막대 닫힘(7px)으로 잔 질감을 줄이고 CLAHE를 건다.
   노면 선명도가 2.76보다 낮은 아주 흐린 사진은 언샤프(3.0)와 CLAHE만 한다.
5. 균열: Hessian으로 어두운 선을 찾고 (아주 흐린 사진은 sigma 3까지 넓혀서) 텐서 전파로 끊긴 조각을 잇는다. 길이 45px 이상, 세장비 3 이상이고 위쪽 가장자리에
   닿지 않은 것 중에서, 양옆이 밝은 골짜기이고 주변이 도로처럼 보이는 것만 남긴다.
6. 포트홀: 주변 노면보다 1.8배 넘게 거친 영역을 찾고, 색, 어두운 부분, 모양, 위치, 주변 노면 색으로 거른다.
7. 도로 위 판정: 박스 넓이의 절반 이상이 노면인 후보만 남기고 원본 좌표로 되돌려 그린다.

## 코드 구조

`run.py`가 사진마다 `src/pipeline.py`의 `process()`를 부른다. 처리 순서는 다음과 같다.

```text
run.py
  pipeline.process()
    preprocessing/preprocess.py            노면 마스크, 검증, Gamma
      road_mask.py                         grid_features, seeds, region_growing, mask_refine 순서로 부름
      mask_validation.py
      gamma.py
    preprocessing/detector_adapter.py      노면 영역 자르기, 크기 맞추기
    preprocessing/detector_corrections.py  흐림 판정, 검출 직전 보정
    detect.py                              균열
    pothole.py                             포트홀
  draw.py                                  결과 그림
```

`roadcue.py`는 균열과 포트홀이 함께 쓰는 노면 단서(주변이 도로처럼 보이는지)를 잰다. 파일별 자세한 설명은 [`src/README.md`](src/README.md)에 있다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `run.py` | 실행 파일 |
| `src/` | 처리 코드. 파일별 설명은 [`src/README.md`](src/README.md) |
| `configs/preprocessing.json` | 노면 마스크, 검증, Gamma 설정값 |
| `configs/detection.json` | 자르기, 보정, 균열, 포트홀 설정값 |
| `data/test/` | `python run.py`를 인자 없이 실행할 때 처리하는 사진 |

설정 파일은 실행 전에 검사한다. 검증, Gamma, 자르기, 보정, 균열, 포트홀 항목에 모르는 키가 있거나 키가 빠지면
바로 오류를 낸다. 노면 마스크의 세부 값과 보정 단계 안쪽 값은 이 검사에 들어가지 않아서, 빠지면 실행 도중에 오류가 난다.
