import argparse
import re
from pathlib import Path

import cv2


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def find_images(input_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    images.sort(key=natural_key)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/ssd1/hk/projects/HZGS/outputs/horizongs_2d/real/road/fine/train/ours_40000/street/renders",
    )
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", type=str, default="mp4v")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是文件夹: {input_dir}")

    images = find_images(input_dir)
    if not images:
        raise RuntimeError(f"目录下没有可用图片: {input_dir}")

    output_path = Path(args.output) if args.output else input_dir / "renders.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"无法读取首帧: {images[0]}")

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    valid_frames = 0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"跳过无法读取的图片: {img_path}")
            continue
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        valid_frames += 1

    writer.release()
    if valid_frames == 0:
        raise RuntimeError("没有任何有效帧写入视频")

    print(f"完成: {output_path}")
    print(f"总帧数: {valid_frames}, FPS: {args.fps}")


if __name__ == "__main__":
    main()
