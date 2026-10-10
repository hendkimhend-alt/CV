"""도로 마스크를 다각형으로 직접 고쳐 PNG로 저장한다 (GUI 필요).

  py src/preprocessing/edit_road_mask.py --image <사진> [--candidate <시작 마스크 PNG>]

저장 위치 기본값: outputs/preprocessing/manual_masks/<이름>.png
"""
import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path[0] = str(Path(__file__).resolve().parents[1])

from paths import REPO_ROOT  # noqa: E402
from preprocessing.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from preprocessing.image_io import read_image, read_mask, write_image  # noqa: E402
from preprocessing.mask_editor import edit_mask_interactive  # noqa: E402
from preprocessing.road_mask import extract_road_mask  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="수동 도로 마스크 편집")
    parser.add_argument("--image", required=True)
    parser.add_argument("--candidate", help="시작 마스크 (없으면 자동 추출)")
    parser.add_argument("--output", help="저장할 PNG 경로")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    image_path = Path(args.image)
    image = read_image(image_path)
    if args.candidate:
        initial = read_mask(args.candidate, image.shape, cfg["manual_correction"]["mask_threshold"])
    else:
        initial = extract_road_mask(image, cfg["roi"]).mask

    if args.output:
        output = Path(args.output)
    else:
        output = REPO_ROOT / "outputs" / "preprocessing" / "manual_masks" / f"{image_path.stem}.png"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"error:이미 있음 {output} (--overwrite로 덮어쓰기)")

    mask = edit_mask_interactive(image, initial, image_path.name)
    if mask is None:
        print("취소 — 저장하지 않음")
        return 1
    write_image(output, mask)
    print(f"저장 → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
