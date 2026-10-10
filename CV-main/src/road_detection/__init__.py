"""도로 손상 검출 시스템 — 전처리(src/preprocessing)와 검출(src/detect.py)을 연결한다.

  config    검출 설정 파일(configs/detection_*.json) 읽기 · 검출기 이름 해석 · 실행 환경 확인
  pipeline  한 장 처리: 전처리 → 검출기 입력 변환 → detect() → 도로 위 후보 판정 · 원본 좌표
  render    검토용 그림

실행기: src/run_road_detection.py. src/가 sys.path에 있어야 한다.
"""
