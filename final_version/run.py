"""도로 사진을 넣으면 노면 마스크와 균열, 포트홀 박스를 그린 사진을 저장한다.

  python run.py                  # data/test/ 의 사진
  python run.py 사진.jpg          # 사진 한 장
  python run.py 사진폴더          # 폴더 안 사진 전부 (하위 폴더 포함)

결과: result/run_<시각>/<사진 이름>.jpg (실행할 때마다 새 폴더)
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import ConfigError
from draw import draw_result
from paths import DATA_DIR, ROOT, imread, imwrite
from pipeline import check_opencv, load_settings, process

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(path):
    """사진 파일이면 그 한 장, 폴더면 하위 폴더까지의 사진 목록(숨김 폴더 제외)을 돌려준다."""
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                  and not any(part.startswith(".") for part in p.relative_to(path).parts))


def output_names(images, base):
    """사진마다 결과 파일 이름을 정한다."""
    # 보통은 파일 이름, 하위 폴더에 같은 이름이 있으면 폴더__이름
    stems = [p.stem for p in images]
    return {p: p.stem if stems.count(p.stem) == 1
            else "__".join(p.relative_to(base).with_suffix("").parts)
            for p in images}


def new_run_dir(parent):
    """result/run_<시각> 폴더를 새로 만든다. 같은 이름이 있으면 _2, _3을 붙인다."""
    base = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    for n in range(1, 100):
        run_dir = parent / (base if n == 1 else f"{base}_{n}")
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            return run_dir
    raise RuntimeError(f"결과 폴더를 만들 수 없음: {parent / base}")


def main():
    """인자를 읽고 사진마다 처리해 결과 그림을 저장한다."""
    p = argparse.ArgumentParser(description="도로 사진에 노면 마스크와 균열, 포트홀 박스를 그려 저장")
    p.add_argument("input", nargs="?", type=Path, default=DATA_DIR / "test",
                   help="사진 파일 또는 폴더 (기본 data/test/)")
    p.add_argument("--out", type=Path, default=ROOT / "result", help="결과 폴더 (기본 result/)")
    a = p.parse_args()

    if not a.input.exists():
        sys.exit(f"사진 파일이나 폴더가 없습니다: {a.input}")
    if a.input.is_file() and a.input.suffix.lower() not in IMAGE_EXTENSIONS:
        sys.exit(f"사진 파일이 아닙니다: {a.input}")
    images = list_images(a.input)
    if not images:
        sys.exit(f"사진이 없습니다: {a.input}")
    try:
        check_opencv()
        settings = load_settings()
    except ConfigError as exc:
        sys.exit(str(exc))

    names = output_names(images, a.input if a.input.is_dir() else a.input.parent)
    run_dir = new_run_dir(a.out)
    print(f"{len(images)}장, 결과 폴더: {run_dir}")
    failed = 0
    for i, path in enumerate(images, 1):
        t0 = time.perf_counter()
        try:
            image = imread(path)
        except ValueError:
            print(f"  [{i}/{len(images)}] {path.name}  읽을 수 없는 파일이라 건너뜀")
            failed += 1
            continue
        try:
            r = process(image, settings)
        except Exception as exc:  # 한 장이 실패해도 나머지는 계속
            print(f"  [{i}/{len(images)}] {path.name}  처리 실패: {type(exc).__name__}: {exc}")
            vis = draw_result(image, None, [], title=f"{path.name}  processing failed")
            imwrite(run_dir / f"{names[path]}.jpg", vis)
            failed += 1
            continue
        n_crack = sum(b["type"] == "crack" for b in r.boxes)
        n_pothole = sum(b["type"] == "pothole" for b in r.boxes)
        if r.ok:
            title = f"{path.name}  crack {n_crack} / pothole {n_pothole}"
            note = ("아주 흐림(언샤프)" if r.blurry else "선명") + (", 검증 안 된 마스크" if r.unverified_mask else "")
        else:
            title = f"{path.name}  road surface not found"
            note = "노면을 찾지 못해 박스 없음"
        imwrite(run_dir / f"{names[path]}.jpg", draw_result(image, r.road_mask, r.boxes, title=title))
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [{i}/{len(images)}] {path.name}  균열 {n_crack}, 포트홀 {n_pothole}, {note}, {ms:.0f} ms")
    print(f"완료: {run_dir}" + (f" (실패 {failed}장)" if failed else ""))


if __name__ == "__main__":
    main()
