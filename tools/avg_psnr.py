from pathlib import Path
import sys

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DIR1 = "acceptance/ours_40000"
DEFAULT_DIR2 = "acceptance/fix_lod/fix-lod-4/ours_40000"


def get_dirs():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    return Path(args.get("dir1", DEFAULT_DIR1)), Path(args.get("dir2", DEFAULT_DIR2))


def image_files(root):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def image_map(root):
    return {p.relative_to(root): p for p in image_files(root)}


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    dir1, dir2 = get_dirs()
    files1 = image_map(dir1)
    files2 = image_map(dir2)
    if set(files1) != set(files2):
        only1 = sorted(set(files1) - set(files2))
        only2 = sorted(set(files2) - set(files1))
        raise SystemExit(f"name mismatch\nonly in dir1: {only1[:5]}\nonly in dir2: {only2[:5]}")

    scores = []

    for rel in sorted(files1):
        img1 = read_rgb(files1[rel])
        img2 = read_rgb(files2[rel])
        if img1.shape != img2.shape:
            raise SystemExit(f"shape mismatch: {rel} {img1.shape} != {img2.shape}")
        score = peak_signal_noise_ratio(img1, img2, data_range=255)
        scores.append(score)
        print(f"{rel}: {score:.4f}")

    print(f"\ncount: {len(scores)}")
    print(f"avg_psnr: {float(np.mean(scores)):.4f}")


if __name__ == "__main__":
    main()
