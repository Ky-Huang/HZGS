#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 255.0, identical_psnr: float = 100.0) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return identical_psnr
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir1",
        type=Path,
        default=Path("/ssd1/hk/projects/HZGS/outputs/horizongs_2d/real/road/fine/test_unfix"),
    )
    parser.add_argument(
        "--dir2",
        type=Path,
        default=Path("/ssd1/hk/projects/HZGS/outputs/horizongs_2d/real/road/fine/test_fix"),
    )
    parser.add_argument("--identical-psnr", type=float, default=100.0)
    parser.add_argument("--print-each", action="store_true")
    args = parser.parse_args()

    dir1 = args.dir1.resolve()
    dir2 = args.dir2.resolve()

    if not dir1.exists():
        raise FileNotFoundError(f"dir1 不存在: {dir1}")
    if not dir2.exists():
        raise FileNotFoundError(f"dir2 不存在: {dir2}")

    psnr_values = []
    missing = []
    shape_mismatch = []
    read_fail = []

    files1 = sorted(iter_images(dir1))
    for f1 in files1:
        rel = f1.relative_to(dir1)
        f2 = dir2 / rel
        if not f2.exists():
            missing.append(str(rel))
            continue

        try:
            img1 = load_image_rgb(f1)
            img2 = load_image_rgb(f2)
        except Exception as e:
            read_fail.append((str(rel), str(e)))
            continue

        if img1.shape != img2.shape:
            shape_mismatch.append((str(rel), img1.shape, img2.shape))
            continue

        v = psnr(img1, img2, identical_psnr=args.identical_psnr)
        psnr_values.append(v)

        if args.print_each:
            print(f"{rel}\tPSNR={v:.6f}")

    total_pairs = len(files1)
    valid_pairs = len(psnr_values)

    print(f"dir1: {dir1}")
    print(f"dir2: {dir2}")
    print(f"dir1 图片数: {total_pairs}")
    print(f"成功匹配并计算: {valid_pairs}")
    print(f"dir2 缺失对应文件: {len(missing)}")
    print(f"尺寸不一致: {len(shape_mismatch)}")
    print(f"读取失败: {len(read_fail)}")

    if valid_pairs == 0:
        print("没有可用的匹配图片对，无法计算均值。")
        return

    mean_psnr = float(np.mean(psnr_values))
    print(f"PSNR 均值: {mean_psnr:.6f} dB")


if __name__ == "__main__":
    main()
