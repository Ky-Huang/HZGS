import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.read_write_model import read_images_binary, read_images_text


def load_selected_images(subset_sparse_dir: Path):
    images_bin = subset_sparse_dir / "images.bin"
    images_txt = subset_sparse_dir / "images.txt"
    if images_bin.exists():
        images = read_images_binary(str(images_bin))
    elif images_txt.exists():
        images = read_images_text(str(images_txt))
    else:
        raise FileNotFoundError(f"images.bin/images.txt not found in {subset_sparse_dir}")
    names = [images[k].name for k in images]
    names.sort()
    return names


def copy_file(src_root: Path, src_file: Path, dst_root: Path, copied_relpaths: set):
    rel = src_file.relative_to(src_root)
    rel_key = rel.as_posix()
    if rel_key in copied_relpaths:
        return False
    dst_file = dst_root / rel
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    copied_relpaths.add(rel_key)
    return True


def candidate_files(data_dir: Path, rel_name: str):
    p = Path(rel_name)
    stem = p.stem
    exts = [p.suffix, ".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".exr", ".npy", ".npz", ".pfm"]
    seen = set()
    base_candidates = [data_dir / p, data_dir / p.name]
    for c in base_candidates:
        k = c.as_posix()
        if k not in seen:
            seen.add(k)
            yield c
    for ext in exts:
        if not ext:
            continue
        if p.suffix:
            c1 = data_dir / p.with_suffix(ext)
        else:
            c1 = data_dir / f"{p.as_posix()}{ext}"
        c2 = data_dir / p.parent / f"{stem}{ext}"
        c3 = data_dir / f"{stem}{ext}"
        for c in (c1, c2, c3):
            k = c.as_posix()
            if k not in seen:
                seen.add(k)
                yield c


def copy_sparse_model(subset_sparse_dir: Path, dst_sparse_dir: Path):
    dst_sparse_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
        src = subset_sparse_dir / name
        if src.exists():
            dst = dst_sparse_dir / name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied.append(name)
    if not any(x in copied for x in ["cameras.bin", "cameras.txt"]):
        raise FileNotFoundError("Missing cameras.bin/cameras.txt in subset sparse dir")
    if not any(x in copied for x in ["images.bin", "images.txt"]):
        raise FileNotFoundError("Missing images.bin/images.txt in subset sparse dir")
    if not any(x in copied for x in ["points3D.bin", "points3D.txt"]):
        raise FileNotFoundError("Missing points3D.bin/points3D.txt in subset sparse dir")
    return copied


def copy_depth_params(src_dataset: Path, subset_sparse_dir: Path, dst_sparse_dir: Path, selected_names):
    subset_json = subset_sparse_dir / "depth_params.json"
    src_json = src_dataset / "sparse" / "0" / "depth_params.json"
    json_path = subset_json if subset_json.exists() else src_json
    if not json_path.exists():
        return 0
    with open(json_path, "r", encoding="utf-8") as f:
        depth_params = json.load(f)
    selected_full = {n.lower() for n in selected_names}
    selected_base = {Path(n).name.lower() for n in selected_names}
    selected_stem = {Path(n).stem.lower() for n in selected_names}
    kept = {}
    for k, v in depth_params.items():
        kl = k.lower()
        if kl in selected_full or kl in selected_base or Path(k).stem.lower() in selected_stem:
            kept[k] = v
    dst_sparse_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_sparse_dir / "depth_params.json", "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    return len(kept)


def generate_configs(ref_config_dir: Path, out_config_dir: Path, new_source_path: str):
    out_config_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r'source_path:\s*(".*?"|\'.*?\'|[^,\n\}]+)')
    generated = []
    for name in ["config.yaml", "coarse.yaml", "fine.yaml"]:
        src = ref_config_dir / name
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8")
        text_new = pattern.sub(f'source_path: "{new_source_path}"', text)
        dst = out_config_dir / name
        dst.write_text(text_new, encoding="utf-8")
        generated.append(str(dst))
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dataset", required=True)
    parser.add_argument("--subset_sparse_dir", required=True)
    parser.add_argument("--dst_dataset", required=True)
    parser.add_argument("--ref_config_dir", default="/ssd1/hk/projects/HZGS/config/ours/urbangs/real/road")
    parser.add_argument("--out_config_dir", required=True)
    parser.add_argument("--config_source_path", default="")
    args = parser.parse_args()

    src_dataset = Path(args.src_dataset).resolve()
    subset_sparse_dir = Path(args.subset_sparse_dir).resolve()
    dst_dataset = Path(args.dst_dataset).resolve()
    ref_config_dir = Path(args.ref_config_dir).resolve()
    out_config_dir = Path(args.out_config_dir).resolve()
    config_source_path = args.config_source_path if args.config_source_path else str(dst_dataset)

    selected_names = load_selected_images(subset_sparse_dir)
    copied_relpaths = set()

    dst_sparse_dir = dst_dataset / "sparse" / "0"
    sparse_files = copy_sparse_model(subset_sparse_dir, dst_sparse_dir)

    copied_images = 0
    for rel_name in selected_names:
        src_img = src_dataset / "images" / rel_name
        if src_img.exists() and src_img.is_file():
            if copy_file(src_dataset, src_img, dst_dataset, copied_relpaths):
                copied_images += 1

    copied_extra = 0
    for d in src_dataset.iterdir():
        if not d.is_dir():
            continue
        if d.name in {"sparse", "images"}:
            continue
        for rel_name in selected_names:
            for cand in candidate_files(d, rel_name):
                if cand.exists() and cand.is_file():
                    if copy_file(src_dataset, cand, dst_dataset, copied_relpaths):
                        copied_extra += 1
                    break

    kept_depth_params = copy_depth_params(src_dataset, subset_sparse_dir, dst_sparse_dir, selected_names)
    generated_cfgs = generate_configs(ref_config_dir, out_config_dir, config_source_path)

    print(f"selected images: {len(selected_names)}")
    print(f"copied sparse files: {sparse_files}")
    print(f"copied images: {copied_images}")
    print(f"copied extra files: {copied_extra}")
    print(f"kept depth_params: {kept_depth_params}")
    print("generated configs:")
    for p in generated_cfgs:
        print(p)


if __name__ == "__main__":
    main()
