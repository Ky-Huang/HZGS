import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scene as scene_models
from gaussian_renderer.render import _project_gaussians_to_2d, apply_runtime_anchor_limits, prefilter_voxel
from scene.cameras import MiniCam
from utils.general_utils import parse_cfg
from utils.graphics_utils import getProjectionMatrixFromIntrinsics
from utils.system_utils import searchForMaxIteration
from xr.frame_sources import load_xr_frames
from xr.openxr_bridge import build_minicam_from_openxr_view, load_xr_session_config


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_cuda(fn):
    _sync_cuda()
    start = time.perf_counter()
    result = fn()
    _sync_cuda()
    return result, (time.perf_counter() - start) * 1000.0


def _to_float(value, digits=4):
    return round(float(value), digits)


def _tensor_summary(values, quantiles=(0.5, 0.95, 0.99)):
    if values is None or values.numel() == 0:
        return {"count": 0}
    values = values.detach().float()
    result = {
        "count": int(values.numel()),
        "min": _to_float(values.min().item()),
        "mean": _to_float(values.mean().item()),
        "max": _to_float(values.max().item()),
    }
    for q in quantiles:
        result[f"p{int(q * 100):02d}"] = _to_float(torch.quantile(values, q).item())
    return result


def _histogram_int(values, minlength=0):
    if values is None or values.numel() == 0:
        return {}
    values = values.detach().long().view(-1)
    counts = torch.bincount(values, minlength=minlength).detach().cpu().tolist()
    return {str(i): int(count) for i, count in enumerate(counts) if count}


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _load_model(model_path, iteration):
    model_path = Path(model_path)
    cfg = _load_yaml(model_path / "config.yaml")
    lp, _op, pp = parse_cfg(cfg)
    lp.model_path = str(model_path)

    model_config = lp.model_config
    gaussians = getattr(scene_models, model_config["name"])(**model_config["kwargs"])
    gaussians.explicit_gs = False

    loaded_iter = int(iteration)
    if loaded_iter < 0:
        loaded_iter = searchForMaxIteration(str(model_path / "point_cloud"))
    checkpoint_dir = model_path / "point_cloud" / f"iteration_{loaded_iter}"
    gaussians.load_ply(str(checkpoint_dir / "point_cloud.ply"))
    gaussians.load_mlp_checkpoints(str(checkpoint_dir))
    gaussians.eval()

    pp.add_prefilter = bool(getattr(pp, "add_prefilter", True))
    return gaussians, pp, loaded_iter, cfg


def _load_slow_frame_payloads(path):
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            payload = record.get("payload", record)
            if isinstance(payload, dict):
                frames.append(payload)
    return frames


def _load_frames(args):
    frames = []
    if args.xr_input:
        frames.extend(load_xr_frames(args.xr_input))
    if args.slow_frame_log:
        frames.extend(_load_slow_frame_payloads(args.slow_frame_log))
    if not frames:
        raise ValueError("Provide --xr_input and/or --slow_frame_log.")

    if args.frame_id:
        wanted = {int(frame_id) for frame_id in args.frame_id}
        frames = [frame for frame in frames if int(frame.get("frame_id", -1)) in wanted]
    if args.frame_index:
        selected = []
        for idx in args.frame_index:
            if idx < 0 or idx >= len(frames):
                raise IndexError(f"--frame_index {idx} is outside 0..{len(frames) - 1}")
            selected.append(frames[idx])
        frames = selected
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    return frames


def _rotation_about_axis(axis, degrees, device):
    axis = torch.tensor(axis, dtype=torch.float32, device=device)
    axis = axis / axis.norm().clamp_min(1e-8)
    radians = torch.tensor(float(degrees) * np.pi / 180.0, dtype=torch.float32, device=device)
    x, y, z = axis
    c = torch.cos(radians)
    s = torch.sin(radians)
    one_c = 1.0 - c
    return torch.stack(
        [
            torch.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s]),
            torch.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s]),
            torch.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c]),
        ]
    )


def _yaw_camera(camera, degrees, up_axis=(0.0, 0.0, 1.0)):
    if abs(float(degrees)) < 1e-8:
        return camera
    c2w = camera.c2w.clone()
    rot = _rotation_about_axis(up_axis, degrees, c2w.device)
    c2w[:3, :3] = rot @ c2w[:3, :3]
    world_view_transform = torch.inverse(c2w).transpose(0, 1).contiguous()
    projection_matrix = getProjectionMatrixFromIntrinsics(
        width=camera.image_width,
        height=camera.image_height,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        znear=camera.znear,
        zfar=camera.zfar,
    ).to(device=world_view_transform.device, dtype=world_view_transform.dtype)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    return MiniCam(
        width=camera.image_width,
        height=camera.image_height,
        fovy=camera.FoVy,
        fovx=camera.FoVx,
        znear=camera.znear,
        zfar=camera.zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        projection_matrix=projection_matrix,
        resolution_scale=camera.resolution_scale,
        image_name=f"{camera.image_name}:yaw{float(degrees):.1f}",
        image_path=camera.image_path,
        image_type=camera.image_type,
    )


def _camera_vectors(camera):
    c2w = camera.get_camera_to_world()
    forward = c2w[:3, 2]
    up = c2w[:3, 1]
    return {
        "center": [_to_float(x) for x in camera.camera_center.detach().cpu().tolist()],
        "forward": [_to_float(x) for x in forward.detach().cpu().tolist()],
        "up": [_to_float(x) for x in up.detach().cpu().tolist()],
    }


def _camera_depths(camera, xyz):
    viewmat = camera.world_view_transform.transpose(0, 1)
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    camera_xyz = torch.cat([xyz, ones], dim=1) @ viewmat
    return camera_xyz[:, 2]


def _bbox_from_tensor(xyz):
    return {
        "min": [_to_float(x) for x in xyz.min(dim=0).values.detach().cpu().tolist()],
        "max": [_to_float(x) for x in xyz.max(dim=0).values.detach().cpu().tolist()],
    }


def _load_reference_bbox(path):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    return {
        "path": str(path),
        "min": xyz.min(axis=0).astype(float).tolist(),
        "max": xyz.max(axis=0).astype(float).tolist(),
    }


def _outside_bbox_mask(xyz, bbox, margin):
    if bbox is None:
        return None
    lo = torch.tensor(bbox["min"], dtype=xyz.dtype, device=xyz.device) - float(margin)
    hi = torch.tensor(bbox["max"], dtype=xyz.dtype, device=xyz.device) + float(margin)
    return ((xyz < lo) | (xyz > hi)).any(dim=1)


def _opacity_offset_stats(pc, camera, visible_mask):
    anchor = pc.get_anchor[visible_mask]
    if anchor.shape[0] == 0:
        return {
            "positive_offsets": 0,
            "active_anchors": 0,
            "inactive_anchors": 0,
            "active_offsets_per_anchor": {"count": 0},
            "positive_offset_ratio": 0.0,
            "active_anchor_ratio": 0.0,
        }
    feat = pc.get_anchor_feat[visible_mask]
    ob_view = anchor - camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True).clamp_min(1e-8)
    ob_view = ob_view / ob_dist
    cat_local_view = torch.cat([feat, ob_view], dim=1) if pc.view_dim > 0 else feat
    opacity = pc.get_opacity_mlp(cat_local_view) * pc.smooth_complement(visible_mask)
    offset_positive = (opacity.reshape(-1) > 0.0)
    per_anchor_positive = offset_positive.view(-1, pc.n_offsets).sum(dim=1)
    active_anchor = per_anchor_positive > 0
    return {
        "positive_offsets": int(offset_positive.sum().item()),
        "active_anchors": int(active_anchor.sum().item()),
        "inactive_anchors": int((~active_anchor).sum().item()),
        "active_offsets_per_anchor": _tensor_summary(per_anchor_positive.float()),
        "positive_offset_ratio": _to_float(offset_positive.float().mean().item()),
        "active_anchor_ratio": _to_float(active_anchor.float().mean().item()),
    }


def _analyze_camera(pc, pipe, camera, reference_bbox, bbox_margin, run_full_generation):
    result = {
        "camera": camera.image_name,
        "image_type": getattr(camera, "image_type", ""),
        "size": [int(camera.image_width), int(camera.image_height)],
        "resolution_scale": float(camera.resolution_scale),
        "camera_pose": _camera_vectors(camera),
    }

    total_anchors = int(pc.get_anchor.shape[0])
    levels = pc.get_level.squeeze()
    result["total_anchors"] = total_anchors
    result["n_offsets"] = int(pc.n_offsets)
    result["all_level_hist"] = _histogram_int(levels, minlength=int(getattr(pc, "street_levels", 0)))
    result["anchor_bbox"] = _bbox_from_tensor(pc.get_anchor)

    _, set_mask_ms = _timed_cuda(lambda: pc.set_anchor_mask(camera.camera_center, camera.resolution_scale))
    pc._anchor_mask = apply_runtime_anchor_limits(camera, pc, pipe, pc._anchor_mask, allow_budget=False)
    lod_mask = pc._anchor_mask.clone()
    lod_xyz = pc.get_anchor[lod_mask]
    result["lod"] = {
        "selected": int(lod_mask.sum().item()),
        "level_hist": _histogram_int(levels[lod_mask], minlength=int(getattr(pc, "street_levels", 0))),
        "distance": _tensor_summary((lod_xyz - camera.camera_center).norm(dim=1)),
        "camera_z": _tensor_summary(_camera_depths(camera, lod_xyz)),
        "set_anchor_mask_ms": _to_float(set_mask_ms),
    }

    def make_prefilter():
        if bool(getattr(pipe, "add_prefilter", True)):
            return prefilter_voxel(camera, pc).squeeze()
        return pc._anchor_mask.clone()

    prefilter_mask, prefilter_ms = _timed_cuda(make_prefilter)
    prefilter_mask = apply_runtime_anchor_limits(camera, pc, pipe, prefilter_mask, allow_budget=True)
    visible_xyz = pc.get_anchor[prefilter_mask]
    visible_levels = levels[prefilter_mask]
    visible_depth = _camera_depths(camera, visible_xyz)
    visible_dist = (visible_xyz - camera.camera_center).norm(dim=1)
    outside_reference = _outside_bbox_mask(visible_xyz, reference_bbox, bbox_margin)

    result["prefilter"] = {
        "selected_anchors": int(prefilter_mask.sum().item()),
        "estimated_gaussians": int(prefilter_mask.sum().item()) * int(pc.n_offsets),
        "level_hist": _histogram_int(visible_levels, minlength=int(getattr(pc, "street_levels", 0))),
        "distance": _tensor_summary(visible_dist),
        "camera_z": _tensor_summary(visible_depth),
        "camera_z_le_near_count": int((visible_depth <= float(camera.znear)).sum().item()),
        "camera_z_gt_far_count": int((visible_depth > float(camera.zfar)).sum().item()),
        "prefilter_ms": _to_float(prefilter_ms),
    }
    if outside_reference is not None:
        outside_count = int(outside_reference.sum().item())
        result["prefilter"]["outside_reference_bbox_count"] = outside_count
        result["prefilter"]["outside_reference_bbox_ratio"] = _to_float(outside_count / max(1, visible_xyz.shape[0]))

    opacity_stats, opacity_ms = _timed_cuda(lambda: _opacity_offset_stats(pc, camera, prefilter_mask))
    opacity_stats["opacity_mlp_ms"] = _to_float(opacity_ms)
    result["opacity_selection"] = opacity_stats

    if run_full_generation:
        def generate():
            xyz, _offset, _color, _opacity, scaling, rot, _sh_degree, selection_mask = pc.generate_neural_gaussians(
                camera, prefilter_mask
            )
            radii, _points2d = _project_gaussians_to_2d(xyz, rot, scaling, camera, pc.gs_attr)
            return xyz, selection_mask, radii

        (xyz, selection_mask, radii), generate_ms = _timed_cuda(generate)
        visible_radii = radii[radii > 0]
        result["full_generation"] = {
            "generated_gaussians": int(xyz.shape[0]),
            "selection_count": int(selection_mask.sum().item()),
            "raster_visible_estimate": int(visible_radii.numel()),
            "radii": _tensor_summary(visible_radii),
            "generate_and_project_ms": _to_float(generate_ms),
        }

    return result


def _default_reference_bbox_path(model_path):
    path = Path(model_path) / "input.ply"
    return str(path) if path.exists() else ""


def _frame_label(frame, fallback):
    return int(frame.get("frame_id", fallback))


def run(args):
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic needs CUDA because the model and gsplat projection run on CUDA.")

    gaussians, pipe, loaded_iter, cfg = _load_model(args.model_path, args.iteration)
    xr_config = load_xr_session_config(args.xr_config)
    if args.resolution_scale > 0:
        xr_config["resolution_scale"] = float(args.resolution_scale)
    pipe.xr_lod_anchor_budget = int(args.lod_anchor_budget)
    pipe.xr_anchor_budget = int(args.anchor_budget)
    pipe.xr_max_anchor_distance = float(args.max_anchor_distance)
    frames = _load_frames(args)
    reference_bbox_path = args.reference_ply or _default_reference_bbox_path(args.model_path)
    reference_bbox = _load_reference_bbox(reference_bbox_path)

    report = {
        "model_path": args.model_path,
        "iteration": loaded_iter,
        "xr_config": args.xr_config,
        "reference_bbox": reference_bbox,
        "bbox_margin": float(args.bbox_margin),
        "model_config": cfg.get("model_params", {}).get("model_config", {}),
        "frames": [],
    }

    eyes = [eye.lower() for eye in args.eye]
    for frame_index, frame in enumerate(frames):
        frame_report = {"frame_index": frame_index, "frame_id": _frame_label(frame, frame_index), "eyes": []}
        for eye in eyes:
            base_camera = build_minicam_from_openxr_view(frame, eye, xr_config)
            for yaw_degrees in args.yaw_offsets:
                camera = _yaw_camera(base_camera, yaw_degrees, args.yaw_up_axis)
                eye_report = _analyze_camera(
                    gaussians,
                    pipe,
                    camera,
                    reference_bbox,
                    args.bbox_margin,
                    args.full_generation,
                )
                eye_report["eye"] = eye
                eye_report["yaw_offset_degrees"] = float(yaw_degrees)
                frame_report["eyes"].append(eye_report)
                print(json.dumps({"frame_id": frame_report["frame_id"], **eye_report}, ensure_ascii=False))
        report["frames"].append(frame_report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[diagnose] wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose XR anchor LoD/prefilter/opacity selection for a trained HorizonGS checkpoint."
    )
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--xr_config", required=True)
    parser.add_argument("--xr_input", default="")
    parser.add_argument("--slow_frame_log", default="")
    parser.add_argument("--frame_id", type=int, action="append", default=[])
    parser.add_argument("--frame_index", type=int, action="append", default=[])
    parser.add_argument("--max_frames", type=int, default=1)
    parser.add_argument("--eye", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--resolution_scale", type=float, default=-1.0)
    parser.add_argument("--yaw_offsets", type=float, nargs="+", default=[0.0])
    parser.add_argument("--yaw_up_axis", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--reference_ply", default="")
    parser.add_argument("--bbox_margin", type=float, default=1.0)
    parser.add_argument("--lod_anchor_budget", type=int, default=-1)
    parser.add_argument("--anchor_budget", type=int, default=-1)
    parser.add_argument("--max_anchor_distance", type=float, default=-1.0)
    parser.add_argument("--full_generation", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
