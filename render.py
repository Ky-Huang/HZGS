#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import imageio.v2 as imageio
import yaml
import cv2
from os import makedirs
import torch
import numpy as np
from PIL import Image

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
import torchvision
import importlib
from tqdm import tqdm
from utils.general_utils import safe_state, parse_cfg, visualize_depth, visualize_normal
from utils.image_utils import save_rgba
from utils.hybrid_camera_paths import (
    build_interpolated_path,
    build_sequence_interpolated_path,
    build_orbit_path,
    collect_camera_pool,
    resolve_orbit_axis,
    find_camera_by_name,
)
from xr import run_openxr_render_session
from argparse import ArgumentParser

HYBRID_RENDER_CAR_BASE_PRESET = {
    "enable_hybrid_render": True,
    "hybrid_mesh_backend": "nvdiffrast",
    "hybrid_scene_config": "config/hybrid/example_scene.yaml",
    "hybrid_verbose": True,
}

HYBRID_RENDER_CAR_ORBIT_PRESET = {
    "camera_path_mode": "orbit",
    "camera_path_reference_view": "X_009_00307",
    "camera_path_frames": 240,
    "camera_path_name": "demo_orbit",
    "camera_path_save_video": True,
    "camera_path_video_fps": 60,
    "orbit_center": [-14.914, 0.3, 17.358],
    "orbit_axis_mode": "auto_scene_up",
    "orbit_style": "reference_plane_circle",
    "orbit_radius": 8.0,
    "orbit_start_azimuth_deg": 0.0,
    "orbit_sweep_deg": 360.0,
}

HYBRID_RENDER_CAR_BASE_FLAGS = {
    "enable_hybrid_render": "--enable_hybrid_render",
    "hybrid_mesh_backend": "--hybrid_mesh_backend",
    "hybrid_scene_config": "--hybrid_scene_config",
    "hybrid_verbose": "--hybrid_verbose",
}

HYBRID_RENDER_CAR_ORBIT_FLAGS = {
    "camera_path_mode": "--camera_path_mode",
    "camera_path_reference_view": "--camera_path_reference_view",
    "camera_path_frames": "--camera_path_frames",
    "camera_path_name": "--camera_path_name",
    "camera_path_save_video": "--camera_path_save_video",
    "camera_path_video_fps": "--camera_path_video_fps",
    "orbit_center": "--orbit_center",
    "orbit_axis_mode": "--orbit_axis_mode",
    "orbit_style": "--orbit_style",
    "orbit_radius": "--orbit_radius",
    "orbit_start_azimuth_deg": "--orbit_start_azimuth_deg",
    "orbit_sweep_deg": "--orbit_sweep_deg",
}


def _apply_cli_preset(args, argv, preset_values, preset_flags):
    argv_flags = set(argv)
    for attr, value in preset_values.items():
        if preset_flags[attr] in argv_flags:
            continue
        setattr(args, attr, value)


def _apply_hybrid_render_car_preset(args, argv):
    if not getattr(args, "hybrid_render_car", False):
        return

    _apply_cli_preset(args, argv, HYBRID_RENDER_CAR_BASE_PRESET, HYBRID_RENDER_CAR_BASE_FLAGS)

    argv_flags = set(argv)
    if "--target_view" in argv_flags or "--xr_mode" in argv_flags:
        return

    camera_path_mode = getattr(args, "camera_path_mode", "")
    if camera_path_mode and camera_path_mode != "orbit":
        return

    _apply_cli_preset(args, argv, HYBRID_RENDER_CAR_ORBIT_PRESET, HYBRID_RENDER_CAR_ORBIT_FLAGS)


def _flow_uv_to_color(flow_uv, support=None, clip_percentile=99.0):
    flow = flow_uv.detach().permute(1, 2, 0).cpu().numpy()
    mag = np.linalg.norm(flow, axis=2)
    if support is None:
        valid = np.ones_like(mag, dtype=bool)
    else:
        valid = support.detach().squeeze(0).cpu().numpy() > 1e-4
    if valid.any():
        max_mag = np.percentile(mag[valid], clip_percentile)
    else:
        max_mag = 1.0
    max_mag = max(max_mag, 1e-6)

    ang = np.arctan2(flow[..., 1], flow[..., 0])
    hue = ((ang + np.pi) / (2 * np.pi) * 179.0).astype(np.uint8)
    sat = np.where(valid, 255, 0).astype(np.uint8)
    val = np.clip((mag / max_mag) * 255.0, 0, 255).astype(np.uint8)
    hsv = np.stack([hue, sat, val], axis=2)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def _motion_vec_filename(idx_a, idx_b):
    return f"{idx_a:05d}_{idx_b:05d}_MotionVector.png"

def _normalize_path(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))

def _filter_views_by_targets(views, target_paths):
    if not target_paths:
        return views
    target_norm = {_normalize_path(p) for p in target_paths}
    target_name = {os.path.basename(p).lower() for p in target_paths}
    target_stem = {os.path.splitext(os.path.basename(p))[0].lower() for p in target_paths}
    filtered = []
    for v in views:
        v_path = _normalize_path(getattr(v, "image_path", v.image_name))
        v_name = os.path.basename(v.image_name).lower()
        v_stem = os.path.splitext(v_name)[0]
        if (v_path in target_norm) or (v_name in target_name) or (v_stem in target_stem):
            filtered.append(v)
    return filtered


def _hybrid_enabled(pipe):
    return bool(getattr(pipe, "enable_hybrid_render", False))


def _select_render_fn(modules, pipe):
    return getattr(modules, "hybrid_render" if _hybrid_enabled(pipe) else "render")


def _resolve_lodmax(gaussians):
    candidates = []
    for attr in ("street_levels", "aerial_levels"):
        value = getattr(gaussians, attr, None)
        if value is not None:
            candidates.append(int(round(float(value))) - 1)
    levels = getattr(gaussians, "_level", None)
    if levels is not None and hasattr(levels, "numel") and levels.numel() > 0:
        candidates.append(int(levels.max().item()))
    return max([0] + candidates)


def _draw_point_cloud_text(image, text, origin):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (origin[0] + 1, origin[1] + 1), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _save_point_cloud_projection(points2d, selected_count, visible_count, output_path, lod_label, lod_value, width, height):
    width = int(width)
    height = int(height)
    density = np.zeros((height, width), dtype=np.float32)

    if points2d is not None and points2d.numel() > 0:
        coords = points2d.detach().cpu().numpy()
        x = np.rint(coords[:, 0]).astype(np.int64)
        y = np.rint(coords[:, 1]).astype(np.int64)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if np.any(valid):
            np.add.at(density, (y[valid], x[valid]), 1.0)

    if density.max() > 0:
        normalized = np.log1p(density) / np.log1p(density.max())
        heat = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        image[density == 0] = 0
    else:
        image = np.zeros((height, width, 3), dtype=np.uint8)

    _draw_point_cloud_text(image, f"{lod_label} (lod={lod_value})", (12, 24))
    _draw_point_cloud_text(image, f"visible: {visible_count}", (12, 48))
    _draw_point_cloud_text(image, f"selected: {selected_count}", (12, 72))
    imageio.imwrite(output_path, image)


def _project_fixed_lod_points(view, gaussians, pipe, lod):
    render_module = importlib.import_module("gaussian_renderer.render")
    had_fix_lod = hasattr(gaussians, "fix_lod")
    previous_lod = getattr(gaussians, "fix_lod", None)
    gaussians.fix_lod = int(lod)
    try:
        if gaussians.explicit_gs:
            gaussians.set_gs_mask(view.camera_center, view.resolution_scale)
            mask = gaussians._gs_mask
            selected_count = int(mask.sum().item())
            if selected_count == 0:
                empty = torch.empty((0, 2), dtype=torch.float32, device="cuda")
                return empty, 0, 0
            xyz, _color, _opacity, scaling, rot, _sh_degree, _selection_mask = gaussians.generate_explicit_gaussians(mask)
        else:
            gaussians.set_anchor_mask(view.camera_center, view.resolution_scale)
            anchor_count = int(gaussians._anchor_mask.sum().item())
            if anchor_count == 0:
                empty = torch.empty((0, 2), dtype=torch.float32, device="cuda")
                return empty, 0, 0
            mask = render_module.prefilter_voxel(view, gaussians).squeeze() if getattr(pipe, "add_prefilter", True) else gaussians._anchor_mask
            selected_count = int(mask.sum().item())
            if selected_count == 0:
                empty = torch.empty((0, 2), dtype=torch.float32, device="cuda")
                return empty, 0, 0
            xyz = gaussians.get_anchor[mask]
            scaling = gaussians.get_scaling[mask][:, :3]
            rot = gaussians.get_rotation[mask]

        radii, points2d = render_module._project_gaussians_to_2d(xyz, rot, scaling, view, gaussians.gs_attr)
        visible = radii > 0
        return points2d[visible], selected_count, int(visible.sum().item())
    finally:
        if had_fix_lod:
            gaussians.fix_lod = previous_lod
        else:
            delattr(gaussians, "fix_lod")


def _render_point_cloud_lod_pair(view, gaussians, pipe, output_paths, filename, lodmax):
    stats = {}
    lod_specs = [("lod0", 0), ("lodmax", lodmax)]
    for lod_label, lod_value in lod_specs:
        points2d, selected_count, visible_count = _project_fixed_lod_points(view, gaussians, pipe, lod_value)
        _save_point_cloud_projection(
            points2d,
            selected_count,
            visible_count,
            os.path.join(output_paths[lod_label], filename),
            lod_label,
            lod_value,
            view.image_width,
            view.image_height,
        )
        stats[lod_label] = {
            "lod": int(lod_value),
            "selected_count": selected_count,
            "visible_count": visible_count,
        }
    return stats


def _save_depth_with_alpha(depth_map, alpha_mask, path):
    vis_depth_map = visualize_depth(depth_map)
    vis_depth_map = torch.concat([vis_depth_map, alpha_mask], dim=0)
    torchvision.utils.save_image(vis_depth_map, path)


def _save_hybrid_buffers(render_pkg, alpha_mask, hybrid_buffer_path, idx):
    if render_pkg.get("gs_depth") is not None:
        _save_depth_with_alpha(
            render_pkg["gs_depth"],
            alpha_mask,
            os.path.join(hybrid_buffer_path["gs_depth"], "{0:05d}.png".format(idx)),
        )
    if render_pkg.get("mesh_depth") is not None and torch.any(render_pkg["mesh_alpha"] > 0.0):
        _save_depth_with_alpha(
            render_pkg["mesh_depth"],
            alpha_mask,
            os.path.join(hybrid_buffer_path["mesh_depth"], "{0:05d}.png".format(idx)),
        )
    if render_pkg.get("render_depth") is not None and torch.any(render_pkg.get("render_alpha", alpha_mask) > 0.0):
        _save_depth_with_alpha(
            render_pkg["render_depth"],
            alpha_mask,
            os.path.join(hybrid_buffer_path["final_depth"], "{0:05d}.png".format(idx)),
        )

    torchvision.utils.save_image(
        render_pkg["gs_alpha"],
        os.path.join(hybrid_buffer_path["gs_alpha"], "{0:05d}.png".format(idx)),
    )
    torchvision.utils.save_image(
        render_pkg["mesh_alpha"],
        os.path.join(hybrid_buffer_path["mesh_alpha"], "{0:05d}.png".format(idx)),
    )
    torchvision.utils.save_image(
        torch.clamp(render_pkg["mesh_rgb"], 0.0, 1.0),
        os.path.join(hybrid_buffer_path["mesh_rgb"], "{0:05d}.png".format(idx)),
    )
    torchvision.utils.save_image(
        torch.clamp(render_pkg["gs_rgb"], 0.0, 1.0),
        os.path.join(hybrid_buffer_path["gs_rgb"], "{0:05d}.png".format(idx)),
    )
    torchvision.utils.save_image(
        torch.clamp(render_pkg["sky_rgb"], 0.0, 1.0),
        os.path.join(hybrid_buffer_path["sky_rgb"], "{0:05d}.png".format(idx)),
    )
    if render_pkg.get("mesh_front_weight") is not None:
        torchvision.utils.save_image(
            torch.clamp(render_pkg["mesh_front_weight"], 0.0, 1.0),
            os.path.join(hybrid_buffer_path["front_weight"], "{0:05d}.png".format(idx)),
        )
    if render_pkg.get("mesh_normal") is not None:
        torchvision.utils.save_image(
            torch.clamp(0.5 * (render_pkg["mesh_normal"] + 1.0), 0.0, 1.0),
            os.path.join(hybrid_buffer_path["mesh_normal"], "{0:05d}.png".format(idx)),
        )

TARGET_RENDER_IMAGE_PATHS = set()
TARGET_RENDER_MAX_VIEWS = None


def _load_yaml_if_exists(path):
    if not path:
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _read_rgb_frame(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _pad_frame_to_block(frame, block_size):
    block_size = max(int(block_size), 1)
    if block_size == 1:
        return frame
    height, width = frame.shape[:2]
    padded_h = ((height + block_size - 1) // block_size) * block_size
    padded_w = ((width + block_size - 1) // block_size) * block_size
    pad_h = padded_h - height
    pad_w = padded_w - width
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _write_video_from_render_dir(render_dir, output_path, fps):
    frame_paths = sorted(
        [
            os.path.join(render_dir, name)
            for name in os.listdir(render_dir)
            if name.lower().endswith(".png") and "_motionvector" not in name.lower()
        ]
    )
    if not frame_paths:
        return False

    sample = _read_rgb_frame(frame_paths[0])
    macro_block_size = 16
    sample = _pad_frame_to_block(sample, macro_block_size)

    writer = imageio.get_writer(
        output_path,
        fps=int(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        writer.append_data(sample)
        for frame_path in frame_paths[1:]:
            frame = _read_rgb_frame(frame_path)
            frame = _pad_frame_to_block(frame, macro_block_size)
            writer.append_data(frame)
    finally:
        writer.close()
    return True


def _save_mesh_instances(render_pkg, output_dir):
    mesh_instances = render_pkg.get("mesh_instances", None)
    if not mesh_instances:
        return
    path = os.path.join(output_dir, "hybrid_mesh_instances.json")
    with open(path, "w") as f:
        json.dump(mesh_instances, f, indent=2)


def _resolve_render_output_root(model_path, fixed_lod=-1):
    if fixed_lod is None:
        return model_path
    fixed_lod = int(fixed_lod)
    if fixed_lod < 0:
        return model_path
    return os.path.join(model_path, f"fix-lod-{fixed_lod}")


def _parse_camera_path_index(index_value, frame_value, field_name):
    if int(index_value) >= 0:
        return int(index_value)
    frame_value = str(frame_value or "").strip()
    if not frame_value:
        return None
    frame_stem = os.path.splitext(os.path.basename(frame_value))[0]
    if not frame_stem.isdigit():
        raise ValueError(f"{field_name} must be an integer index or a zero-padded frame filename, got '{frame_value}'.")
    return int(frame_stem)


def _select_camera_path_source(scene, source_name):
    source_name = (source_name or "all").strip().lower()
    if source_name == "train":
        return scene.getTrainCameras()
    if source_name == "test":
        return scene.getTestCameras()
    if source_name == "all":
        return collect_camera_pool(scene)
    raise ValueError(f"Unknown camera_path_source: {source_name}")


def _filter_camera_path_image_type(cameras, image_type):
    image_type = (image_type or "all").strip().lower()
    if image_type == "all":
        return cameras
    if image_type not in {"street", "aerial"}:
        raise ValueError(f"Unknown camera_path_image_type: {image_type}")
    return [camera for camera in cameras if getattr(camera, "image_type", "") == image_type]


def _render_camera_path(scene, dataset, gaussians, pipe, render_motion_vectors, write_per_view_count, render_point_cloud):
    mode = getattr(dataset, "camera_path_mode", "")
    if not mode:
        return False

    all_cameras = collect_camera_pool(scene)
    if not all_cameras:
        raise ValueError("No cameras were loaded, cannot build a camera path.")

    if mode == "interpolate":
        start_name = getattr(dataset, "camera_path_start_view", "")
        end_name = getattr(dataset, "camera_path_end_view", "")
        if not start_name or not end_name:
            raise ValueError("camera_path_mode='interpolate' requires --camera_path_start_view and --camera_path_end_view.")
        start_cam = find_camera_by_name(all_cameras, start_name)
        end_cam = find_camera_by_name(all_cameras, end_name)
        path_views = build_interpolated_path(start_cam, end_cam, getattr(dataset, "camera_path_frames", 120))
    elif mode == "sequence":
        source_name = getattr(dataset, "camera_path_source", "all")
        image_type = getattr(dataset, "camera_path_image_type", "all")
        source_cameras = _select_camera_path_source(scene, source_name)
        source_cameras = _filter_camera_path_image_type(source_cameras, image_type)
        if not source_cameras:
            raise ValueError(
                f"camera_path_mode='sequence' found no cameras after filtering source='{source_name}' image_type='{image_type}'."
            )

        start_index = _parse_camera_path_index(
            getattr(dataset, "camera_path_start_index", -1),
            getattr(dataset, "camera_path_start_frame", ""),
            "--camera_path_start_index/--camera_path_start_frame",
        )
        end_index = _parse_camera_path_index(
            getattr(dataset, "camera_path_end_index", -1),
            getattr(dataset, "camera_path_end_frame", ""),
            "--camera_path_end_index/--camera_path_end_frame",
        )
        if start_index is None or end_index is None:
            raise ValueError(
                "camera_path_mode='sequence' requires --camera_path_start_index/--camera_path_start_frame "
                "and --camera_path_end_index/--camera_path_end_frame."
            )
        if start_index < 0 or end_index < 0:
            raise ValueError("camera_path_mode='sequence' requires non-negative start/end indices.")
        if start_index > end_index:
            raise ValueError("camera_path_mode='sequence' requires start index <= end index.")
        if end_index >= len(source_cameras):
            raise ValueError(
                f"camera_path_mode='sequence' end index {end_index} is out of range for {len(source_cameras)} filtered cameras."
            )

        key_cameras = source_cameras[start_index : end_index + 1]
        num_intermediate_views = int(getattr(dataset, "camera_path_intermediate_views", 1))
        path_views = build_sequence_interpolated_path(
            key_cameras,
            num_intermediate_views=num_intermediate_views,
        )
        print(
            f"[camera-path] source={source_name} image_type={image_type} key_frames={len(key_cameras)} "
            f"range=[{start_index}, {end_index}] inserted_per_pair={num_intermediate_views} total_frames={len(path_views)}"
        )
    elif mode == "orbit":
        ref_name = getattr(dataset, "camera_path_reference_view", "") or getattr(dataset, "camera_path_start_view", "")
        if not ref_name:
            ref_cam = all_cameras[0]
        else:
            ref_cam = find_camera_by_name(all_cameras, ref_name)
        center = getattr(dataset, "orbit_center", None)
        axis = getattr(dataset, "orbit_axis", None)
        radius = getattr(dataset, "orbit_radius", None)
        axis_mode = getattr(dataset, "orbit_axis_mode", "world")
        orbit_style = getattr(dataset, "orbit_style", "sphere")
        if center is None or radius is None:
            raise ValueError("camera_path_mode='orbit' requires --orbit_center and --orbit_radius.")
        resolved_axis = resolve_orbit_axis(ref_cam, all_cameras, axis=axis, axis_mode=axis_mode)
        print(
            f"[camera-path] orbit_axis_mode={axis_mode} orbit_style={orbit_style} "
            f"resolved_axis={np.round(resolved_axis, 4).tolist()} "
            f"reference_cam={np.round(ref_cam.camera_center.detach().cpu().numpy(), 4).tolist()} "
            f"lookat_center={np.round(np.asarray(center), 4).tolist()}"
        )
        path_views = build_orbit_path(
            ref_cam,
            cameras=all_cameras,
            center=center,
            axis=axis,
            radius=radius,
            polar_deg=getattr(dataset, "orbit_polar_deg", 60.0),
            num_frames=getattr(dataset, "camera_path_frames", 120),
            start_azimuth_deg=getattr(dataset, "orbit_start_azimuth_deg", 0.0),
            sweep_deg=getattr(dataset, "orbit_sweep_deg", 360.0),
            axis_mode=axis_mode,
            orbit_style=orbit_style,
        )
    else:
        raise ValueError(f"Unknown camera_path_mode: {mode}")

    raw_render_name = getattr(dataset, "camera_path_name", "")
    render_name = raw_render_name or f"path_{mode}"
    if mode == "sequence" and not raw_render_name:
        source_name = getattr(dataset, "camera_path_source", "all")
        image_type = getattr(dataset, "camera_path_image_type", "all")
        start_index = _parse_camera_path_index(
            getattr(dataset, "camera_path_start_index", -1),
            getattr(dataset, "camera_path_start_frame", ""),
            "--camera_path_start_index/--camera_path_start_frame",
        )
        end_index = _parse_camera_path_index(
            getattr(dataset, "camera_path_end_index", -1),
            getattr(dataset, "camera_path_end_frame", ""),
            "--camera_path_end_index/--camera_path_end_frame",
        )
        render_name = (
            f"path_sequence_{source_name}_{image_type}_{start_index:05d}_{end_index:05d}"
        )
    render_set(
        dataset.model_path,
        render_name,
        scene.loaded_iter,
        path_views,
        gaussians,
        pipe,
        scene.background,
        dataset.add_aerial,
        dataset.add_street,
        render_only=True,
        render_motion_vectors=render_motion_vectors,
        write_per_view_count=write_per_view_count,
        render_point_cloud=render_point_cloud,
    )

    if bool(getattr(dataset, "camera_path_save_video", False)):
        root = os.path.join(
            _resolve_render_output_root(dataset.model_path, getattr(pipe, "fix_lod", -1)),
            render_name,
            f"ours_{scene.loaded_iter}",
        )
        for branch in ["street", "aerial"]:
            render_dir = os.path.join(root, branch, "renders")
            if not os.path.isdir(render_dir):
                continue
            output_path = os.path.join(root, f"{branch}.mp4")
            if _write_video_from_render_dir(render_dir, output_path, getattr(dataset, "camera_path_video_fps", 30)):
                print(f"[camera-path] wrote video: {output_path}")
    return True

def render_set(model_path, name, iteration, views, gaussians, pipe, background, add_aerial, add_street, render_only=False, render_motion_vectors=False, write_per_view_count=False, show_fps=False, render_point_cloud=False):
    vis_normal = False
    vis_depth = False
    if gaussians.gs_attr == "2D" and not render_only:
        vis_normal = True
        vis_depth = True
    hybrid_enabled = _hybrid_enabled(pipe)
    save_hybrid_buffers = hybrid_enabled and bool(getattr(pipe, "hybrid_save_buffers", False))
    output_root = _resolve_render_output_root(model_path, getattr(pipe, "fix_lod", -1))
    point_cloud_lodmax = _resolve_lodmax(gaussians) if render_point_cloud else 0

    views = _filter_views_by_targets(views, TARGET_RENDER_IMAGE_PATHS)
    if TARGET_RENDER_MAX_VIEWS is not None:
        views = views[:TARGET_RENDER_MAX_VIEWS]
    print(f"[{name}] selected views: {len(views)}")
        
    if add_aerial:
        aerial_render_path = os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "renders")
        makedirs(aerial_render_path, exist_ok=True)
        if not render_only:
            aerial_error_path = os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "errors")
            aerial_gts_path = os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "gt")
            makedirs(aerial_error_path, exist_ok=True)
            makedirs(aerial_gts_path, exist_ok=True)
        
        if vis_normal:
            aerial_normal_path = os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "normal")
            makedirs(aerial_normal_path, exist_ok=True)
        if vis_depth:
            aerial_depth_path = os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "depth")
            makedirs(aerial_depth_path, exist_ok=True)
        if save_hybrid_buffers:
            aerial_hybrid_buffer_path = {
                "gs_depth": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_gs_depth"),
                "mesh_depth": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_mesh_depth"),
                "final_depth": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_final_depth"),
                "gs_alpha": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_gs_alpha"),
                "mesh_alpha": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_mesh_alpha"),
                "mesh_rgb": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_mesh_rgb"),
                "gs_rgb": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_gs_rgb"),
                "sky_rgb": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_sky_rgb"),
                "front_weight": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_front_weight"),
                "mesh_normal": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "hybrid_mesh_normal"),
            }
            for path in aerial_hybrid_buffer_path.values():
                makedirs(path, exist_ok=True)
        if render_point_cloud:
            aerial_point_cloud_path = {
                "lod0": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "point_cloud_lod0"),
                "lodmax": os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "point_cloud_lodmax"),
            }
            for path in aerial_point_cloud_path.values():
                makedirs(path, exist_ok=True)
    if add_street:
        street_render_path = os.path.join(output_root, name, "ours_{}".format(iteration), "street", "renders")
        makedirs(street_render_path, exist_ok=True)
        if not render_only:
            street_error_path = os.path.join(output_root, name, "ours_{}".format(iteration), "street", "errors")
            street_gts_path = os.path.join(output_root, name, "ours_{}".format(iteration), "street", "gt")
            makedirs(street_error_path, exist_ok=True)
            makedirs(street_gts_path, exist_ok=True)
        
        if vis_normal:
            street_normal_path = os.path.join(output_root, name, "ours_{}".format(iteration), "street", "normal")
            makedirs(street_normal_path, exist_ok=True)
        if vis_depth:
            street_depth_path = os.path.join(output_root, name, "ours_{}".format(iteration), "street", "depth")
            makedirs(street_depth_path, exist_ok=True)
        if save_hybrid_buffers:
            street_hybrid_buffer_path = {
                "gs_depth": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_gs_depth"),
                "mesh_depth": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_mesh_depth"),
                "final_depth": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_final_depth"),
                "gs_alpha": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_gs_alpha"),
                "mesh_alpha": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_mesh_alpha"),
                "mesh_rgb": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_mesh_rgb"),
                "gs_rgb": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_gs_rgb"),
                "sky_rgb": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_sky_rgb"),
                "front_weight": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_front_weight"),
                "mesh_normal": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "hybrid_mesh_normal"),
            }
            for path in street_hybrid_buffer_path.values():
                makedirs(path, exist_ok=True)
        if render_point_cloud:
            street_point_cloud_path = {
                "lod0": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "point_cloud_lod0"),
                "lodmax": os.path.join(output_root, name, "ours_{}".format(iteration), "street", "point_cloud_lodmax"),
            }
            for path in street_point_cloud_path.values():
                makedirs(path, exist_ok=True)

    modules = __import__('gaussian_renderer')
    render_fn = _select_render_fn(modules, pipe)

    street_t_list = []
    street_visible_count_list = []
    street_per_view_dict = {}
    street_point_cloud_dict = {}
    street_views = [view for view in views if view.image_type=="street"]
    for idx, view in enumerate(tqdm(street_views, desc="Street rendering progress")):
        if show_fps:
            torch.cuda.synchronize()
            t_start = time.time()
        render_pkg = render_fn(view, gaussians, pipe, background)
        # render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        if render_motion_vectors and idx + 1 < len(street_views):
            next_view = street_views[idx + 1]
            motion_pkg = getattr(modules, 'render_motion_vectors')(view, next_view, gaussians, pipe, background)
            motion_color = _flow_uv_to_color(motion_pkg["motion"], motion_pkg["motion_support"])
            motion_name = _motion_vec_filename(idx, idx + 1)
            imageio.imwrite(os.path.join(street_render_path, motion_name), motion_color)
        if show_fps:
            torch.cuda.synchronize()
            street_t_list.append(time.time() - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        alpha_mask = torch.ones((1, rendering.shape[1], rendering.shape[2]), device=rendering.device)

        if not render_only:
            view.ensure_image_tensors()
            gt = view.original_image.cuda()
            alpha_mask = view.alpha_mask.cuda()
            rendering = torch.cat([rendering, alpha_mask], dim=0)
            gt = torch.cat([gt, alpha_mask], dim=0)
            if gt.device != rendering.device:
                rendering = rendering.to(gt.device)
            errormap = (rendering - gt).abs()
        
        if vis_normal == True:
            normal_map = render_pkg['render_normals'][0].detach()
            vis_normal_map = visualize_normal(normal_map, view)
            vis_alpha_mask = ((alpha_mask * 255).byte()).permute(1, 2, 0).cpu().numpy()
            vis_normal_map = np.concatenate((vis_normal_map,vis_alpha_mask),axis=2)
            imageio.imwrite(os.path.join(street_normal_path, '{0:05d}'.format(idx) + ".png"), vis_normal_map)
        
        if vis_depth == True:
            depth_map = render_pkg["render_depth"]
            _save_depth_with_alpha(depth_map, alpha_mask, os.path.join(street_depth_path, '{0:05d}'.format(idx) + ".png"))

        if save_hybrid_buffers:
            _save_hybrid_buffers(render_pkg, alpha_mask, street_hybrid_buffer_path, idx)
        if hybrid_enabled and idx == 0:
            _save_mesh_instances(
                render_pkg,
                os.path.join(output_root, name, "ours_{}".format(iteration), "street"),
            )

        if render_only:
            torchvision.utils.save_image(rendering, os.path.join(street_render_path, '{0:05d}'.format(idx) + ".png"))
        else:
            save_rgba(rendering, os.path.join(street_render_path, '{0:05d}'.format(idx) + ".png"))
            save_rgba(errormap, os.path.join(street_error_path, '{0:05d}'.format(idx) + ".png"))
            save_rgba(gt, os.path.join(street_gts_path, '{0:05d}'.format(idx) + ".png"))
            view.release_image_tensors()
        street_visible_count_list.append(int(visible_count.item()))
        filename = '{0:05d}'.format(idx) + ".png"
        street_per_view_dict[filename] = int(visible_count.item())
        if render_point_cloud:
            street_point_cloud_dict[filename] = _render_point_cloud_lod_pair(
                view,
                gaussians,
                pipe,
                street_point_cloud_path,
                filename,
                point_cloud_lodmax,
            )
    
    if write_per_view_count and len(street_views) > 0:
        street_count_path = os.path.join(output_root, name, "ours_{}".format(iteration), "street", "per_view_count.json")
        with open(street_count_path, 'w') as fp:
            json.dump(street_per_view_dict, fp, indent=True)
        os.chmod(street_count_path, 0o666)
    if render_point_cloud and len(street_views) > 0:
        street_point_cloud_count_path = os.path.join(output_root, name, "ours_{}".format(iteration), "street", "point_cloud_counts.json")
        with open(street_point_cloud_count_path, 'w') as fp:
            json.dump(street_point_cloud_dict, fp, indent=True)
        os.chmod(street_point_cloud_count_path, 0o666)
    
    aerial_t_list = []
    aerial_visible_count_list = []
    aerial_per_view_dict = {}
    aerial_point_cloud_dict = {}
    aerial_views = [view for view in views if view.image_type=="aerial"]
    for idx, view in enumerate(tqdm(aerial_views, desc="Aerial rendering progress")):
        if show_fps:
            torch.cuda.synchronize()
            t_start = time.time()
        render_pkg = render_fn(view, gaussians, pipe, background)
        # render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        if render_motion_vectors and idx + 1 < len(aerial_views):
            next_view = aerial_views[idx + 1]
            motion_pkg = getattr(modules, 'render_motion_vectors')(view, next_view, gaussians, pipe, background)
            motion_color = _flow_uv_to_color(motion_pkg["motion"], motion_pkg["motion_support"])
            motion_name = _motion_vec_filename(idx, idx + 1)
            imageio.imwrite(os.path.join(aerial_render_path, motion_name), motion_color)
        if show_fps:
            torch.cuda.synchronize()
            aerial_t_list.append(time.time() - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        alpha_mask = torch.ones((1, rendering.shape[1], rendering.shape[2]), device=rendering.device)

        if not render_only:
            view.ensure_image_tensors()
            gt = view.original_image.cuda()
            alpha_mask = view.alpha_mask.cuda()
            rendering = torch.cat([rendering, alpha_mask], dim=0)
            gt = torch.cat([gt, alpha_mask], dim=0)
            if gt.device != rendering.device:
                rendering = rendering.to(gt.device)
            errormap = (rendering - gt).abs()
        
        if vis_normal == True:
            normal_map = render_pkg['render_normals'][0] 
            vis_normal_map = visualize_normal(normal_map, view)
            vis_alpha_mask = ((alpha_mask * 255).byte()).permute(1, 2, 0).cpu().numpy()
            vis_normal_map = np.concatenate((vis_normal_map,vis_alpha_mask),axis=2)
            imageio.imwrite(os.path.join(aerial_normal_path, '{0:05d}'.format(idx) + ".png"), vis_normal_map)
        
        if vis_depth == True:
            depth_map = render_pkg["render_depth"]
            _save_depth_with_alpha(depth_map, alpha_mask, os.path.join(aerial_depth_path, '{0:05d}'.format(idx) + ".png"))

        if save_hybrid_buffers:
            _save_hybrid_buffers(render_pkg, alpha_mask, aerial_hybrid_buffer_path, idx)
        if hybrid_enabled and idx == 0:
            _save_mesh_instances(
                render_pkg,
                os.path.join(output_root, name, "ours_{}".format(iteration), "aerial"),
            )

        if render_only:
            torchvision.utils.save_image(rendering, os.path.join(aerial_render_path, '{0:05d}'.format(idx) + ".png"))
        else:
            save_rgba(rendering, os.path.join(aerial_render_path, '{0:05d}'.format(idx) + ".png"))
            save_rgba(errormap, os.path.join(aerial_error_path, '{0:05d}'.format(idx) + ".png"))
            save_rgba(gt, os.path.join(aerial_gts_path, '{0:05d}'.format(idx) + ".png"))
            view.release_image_tensors()
        aerial_visible_count_list.append(int(visible_count.item()))
        filename = '{0:05d}'.format(idx) + ".png"
        aerial_per_view_dict[filename] = int(visible_count.item())
        if render_point_cloud:
            aerial_point_cloud_dict[filename] = _render_point_cloud_lod_pair(
                view,
                gaussians,
                pipe,
                aerial_point_cloud_path,
                filename,
                point_cloud_lodmax,
            )

    if write_per_view_count and len(aerial_views) > 0:
        aerial_count_path = os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "per_view_count.json")
        with open(aerial_count_path, 'w') as fp:
            json.dump(aerial_per_view_dict, fp, indent=True)
        os.chmod(aerial_count_path, 0o666)
    if render_point_cloud and len(aerial_views) > 0:
        aerial_point_cloud_count_path = os.path.join(output_root, name, "ours_{}".format(iteration), "aerial", "point_cloud_counts.json")
        with open(aerial_point_cloud_count_path, 'w') as fp:
            json.dump(aerial_point_cloud_dict, fp, indent=True)
        os.chmod(aerial_point_cloud_count_path, 0o666)

    if show_fps:
        aerial_valid_count = max(len(aerial_t_list) - 5, 0)
        street_valid_count = max(len(street_t_list) - 5, 0)
        aerial_time = sum(aerial_t_list[5:])
        street_time = sum(street_t_list[5:])

        aerial_fps = aerial_valid_count / aerial_time if aerial_time > 0 else 0.0
        street_fps = street_valid_count / street_time if street_time > 0 else 0.0

        total_valid_count = aerial_valid_count + street_valid_count
        total_time = aerial_time + street_time
        total_fps = total_valid_count / total_time if total_time > 0 else 0.0

        print("Aerial帧率: {}".format(aerial_fps))
        print("Street帧率: {}".format(street_fps))
        print("总帧率: {}".format(total_fps))

    
def render_sets(dataset, opt, pipe, iteration, skip_train, skip_test, ape_code, explicit, render_motion_vectors=False, write_per_view_count=False, show_fps=False, render_point_cloud=False):
    with torch.no_grad():
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
        fixed_lod = int(getattr(pipe, "fix_lod", -1))
        camera_path_mode = getattr(dataset, "camera_path_mode", "")
        xr_mode = getattr(dataset, "xr_mode", "")
        scene_skip_train = skip_train if not (camera_path_mode or xr_mode) else False
        scene_skip_test = skip_test if not (camera_path_mode or xr_mode) else False
        modules = __import__('scene')
        model_config = dataset.model_config
        model_config['kwargs']['ape_code'] = ape_code
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit, skip_train=scene_skip_train, skip_test=scene_skip_test)
        gaussians.eval()
        if fixed_lod >= 0:
            gaussians.fix_lod = fixed_lod
            print(f"[render] fixed LoD enabled: {fixed_lod}")

        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if xr_mode:
            render_fn = _select_render_fn(__import__('gaussian_renderer'), pipe)
            run_openxr_render_session(
                model_path=_resolve_render_output_root(dataset.model_path, fixed_lod),
                iteration=scene.loaded_iter,
                gaussians=gaussians,
                pipe=pipe,
                background=scene.background,
                render_fn=render_fn,
                xr_mode=xr_mode,
                xr_input=getattr(dataset, "xr_input", ""),
                xr_config_path=getattr(dataset, "xr_config", ""),
                xr_output_name=getattr(dataset, "xr_output_name", "openxr"),
                xr_output_layout=getattr(dataset, "xr_output_layout", "both"),
                xr_save_video=bool(getattr(dataset, "xr_save_video", False)),
                xr_video_fps=int(getattr(dataset, "xr_video_fps", 30)),
                xr_socket_host=getattr(dataset, "xr_socket_host", "127.0.0.1"),
                xr_socket_port=int(getattr(dataset, "xr_socket_port", 6110)),
                xr_max_frames=int(getattr(dataset, "xr_max_frames", -1)),
                xr_match_swapchain_resolution_scale=bool(
                    getattr(dataset, "xr_match_swapchain_resolution_scale", False)
                ),
            )
            return

        if _render_camera_path(scene, dataset, gaussians, pipe, render_motion_vectors, write_per_view_count, render_point_cloud):
            return
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, dataset.add_aerial, dataset.add_street, render_only=getattr(dataset, "render_only", False), render_motion_vectors=render_motion_vectors, write_per_view_count=write_per_view_count, show_fps=show_fps, render_point_cloud=render_point_cloud)

        if not skip_test:
            test_views = scene.getTestCameras()
            if len(TARGET_RENDER_IMAGE_PATHS) > 0 and skip_train:
                test_views = scene.getTrainCameras() + scene.getTestCameras()
                print("[test] using train+test camera pool for target-image rendering")
            render_set(dataset.model_path, "test", scene.loaded_iter, test_views, gaussians, pipe, scene.background, dataset.add_aerial, dataset.add_street, render_only=getattr(dataset, "render_only", False), render_motion_vectors=render_motion_vectors, write_per_view_count=write_per_view_count, show_fps=show_fps, render_point_cloud=render_point_cloud)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lazy_load_images", action="store_true")
    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--enable_hybrid_render", action="store_true")
    parser.add_argument(
        "--hybrid_render_car",
        action="store_true",
        help="Apply the default hybrid car scene preset. If no target view/XR mode is given, it also enables the default orbit-video camera path.",
    )
    parser.add_argument("--hybrid_scene_config", type=str, default="")
    parser.add_argument("--hybrid_mesh_path", type=str, default="")
    parser.add_argument("--hybrid_mesh_backend", type=str, default="")
    parser.add_argument("--hybrid_skybox_path", type=str, default="")
    parser.add_argument("--hybrid_save_buffers", action="store_true")
    parser.add_argument("--hybrid_mesh_scale", type=float, nargs="+", default=None)
    parser.add_argument("--hybrid_mesh_rotation_deg", type=float, nargs=3, default=None)
    parser.add_argument("--hybrid_mesh_translation", type=float, nargs=3, default=None)
    parser.add_argument("--hybrid_mesh_center_world", type=float, nargs=3, default=None)
    parser.add_argument("--hybrid_debug_cube", action="store_true")
    parser.add_argument("--hybrid_debug_cube_size", type=float, default=1.0)
    parser.add_argument("--hybrid_debug_cube_distance", type=float, default=4.0)
    parser.add_argument("--hybrid_debug_placement", type=str, default="")
    parser.add_argument("--hybrid_debug_scene_margin", type=float, default=-1.0)
    parser.add_argument("--hybrid_debug_min_distance", type=float, default=-1.0)
    parser.add_argument("--hybrid_auto_place_mesh", action="store_true")
    parser.add_argument("--hybrid_use_gs_env_light", action="store_true")
    parser.add_argument("--hybrid_mesh_env_resolution", type=int, default=-1)
    parser.add_argument("--hybrid_mesh_env_diffuse_strength", type=float, default=-1.0)
    parser.add_argument("--hybrid_mesh_env_ambient_strength", type=float, default=-1.0)
    parser.add_argument("--hybrid_verbose", action="store_true")
    parser.add_argument("--target_view", action="append", default=[])
    parser.add_argument("--max_views", type=int, default=-1)
    parser.add_argument("--render_motion_vectors", action="store_true")
    parser.add_argument("--write_per_view_count", action="store_true")
    parser.add_argument("--render_point_cloud", action="store_true")
    parser.add_argument("--showFPS", action="store_true")
    parser.add_argument("--camera_path_mode", type=str, default="")
    parser.add_argument("--camera_path_start_view", type=str, default="")
    parser.add_argument("--camera_path_end_view", type=str, default="")
    parser.add_argument("--camera_path_reference_view", type=str, default="")
    parser.add_argument("--camera_path_frames", type=int, default=120)
    parser.add_argument("--camera_path_name", type=str, default="")
    parser.add_argument("--camera_path_source", type=str, default="all")
    parser.add_argument("--camera_path_image_type", type=str, default="all")
    parser.add_argument("--camera_path_start_index", type=int, default=-1)
    parser.add_argument("--camera_path_end_index", type=int, default=-1)
    parser.add_argument("--camera_path_start_frame", type=str, default="")
    parser.add_argument("--camera_path_end_frame", type=str, default="")
    parser.add_argument("--camera_path_intermediate_views", type=int, default=1)
    parser.add_argument("--camera_path_save_video", action="store_true")
    parser.add_argument("--camera_path_video_fps", type=int, default=30)
    parser.add_argument("--orbit_center", type=float, nargs=3, default=None)
    parser.add_argument("--orbit_axis", type=float, nargs=3, default=None)
    parser.add_argument("--orbit_axis_mode", type=str, default="world")
    parser.add_argument("--orbit_style", type=str, default="sphere")
    parser.add_argument("--orbit_radius", type=float, default=None)
    parser.add_argument("--orbit_polar_deg", type=float, default=60.0)
    parser.add_argument("--orbit_start_azimuth_deg", type=float, default=0.0)
    parser.add_argument("--orbit_sweep_deg", type=float, default=360.0)
    parser.add_argument("--xr_mode", type=str, default="")
    parser.add_argument("--xr_input", type=str, default="")
    parser.add_argument("--xr_config", type=str, default="")
    parser.add_argument("--xr_output_name", type=str, default="openxr")
    parser.add_argument("--xr_output_layout", type=str, default="both")
    parser.add_argument("--xr_save_video", action="store_true")
    parser.add_argument("--xr_video_fps", type=int, default=30)
    parser.add_argument("--xr_socket_host", type=str, default="127.0.0.1")
    parser.add_argument("--xr_socket_port", type=int, default=6110)
    parser.add_argument("--xr_max_frames", type=int, default=-1)
    parser.add_argument("--xr_match_swapchain_resolution_scale", action="store_true")
    parser.add_argument("--fix-lod", type=int, default=-1)
    args = parser.parse_args(sys.argv[1:])
    _apply_hybrid_render_car_preset(args, sys.argv[1:])

    if args.num_workers > 0:
        os.environ["HGS_NUM_WORKERS"] = str(args.num_workers)

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        lp.model_path = args.model_path
        lp.lazy_load_images = args.lazy_load_images
        lp.render_only = args.render_only or bool(args.camera_path_mode) or bool(args.xr_mode)
        lp.target_views = args.target_view
        lp.camera_path_mode = args.camera_path_mode
        lp.camera_path_start_view = args.camera_path_start_view
        lp.camera_path_end_view = args.camera_path_end_view
        lp.camera_path_reference_view = args.camera_path_reference_view
        lp.camera_path_frames = args.camera_path_frames
        lp.camera_path_name = args.camera_path_name
        lp.camera_path_source = args.camera_path_source
        lp.camera_path_image_type = args.camera_path_image_type
        lp.camera_path_start_index = args.camera_path_start_index
        lp.camera_path_end_index = args.camera_path_end_index
        lp.camera_path_start_frame = args.camera_path_start_frame
        lp.camera_path_end_frame = args.camera_path_end_frame
        lp.camera_path_intermediate_views = args.camera_path_intermediate_views
        lp.camera_path_save_video = args.camera_path_save_video
        lp.camera_path_video_fps = args.camera_path_video_fps
        lp.orbit_center = args.orbit_center
        lp.orbit_axis = args.orbit_axis
        lp.orbit_axis_mode = args.orbit_axis_mode
        lp.orbit_style = args.orbit_style
        lp.orbit_radius = args.orbit_radius
        lp.orbit_polar_deg = args.orbit_polar_deg
        lp.orbit_start_azimuth_deg = args.orbit_start_azimuth_deg
        lp.orbit_sweep_deg = args.orbit_sweep_deg
        lp.xr_mode = args.xr_mode.strip().lower()
        lp.xr_input = args.xr_input
        lp.xr_config = args.xr_config
        lp.xr_output_name = args.xr_output_name
        lp.xr_output_layout = args.xr_output_layout.strip().lower()
        lp.xr_save_video = args.xr_save_video
        lp.xr_video_fps = args.xr_video_fps
        lp.xr_socket_host = args.xr_socket_host
        lp.xr_socket_port = args.xr_socket_port
        lp.xr_max_frames = args.xr_max_frames
        lp.xr_match_swapchain_resolution_scale = args.xr_match_swapchain_resolution_scale
        lp.fix_lod = args.fix_lod if args.fix_lod >= 0 else getattr(lp, "fix_lod", -1)
        lp.render_point_cloud = args.render_point_cloud
    if args.enable_hybrid_render:
        pp.enable_hybrid_render = True
    if args.hybrid_scene_config:
        pp.hybrid_scene_config = args.hybrid_scene_config
        pp.hybrid_scene_data = _load_yaml_if_exists(args.hybrid_scene_config)
    if args.hybrid_mesh_path:
        pp.hybrid_mesh_path = args.hybrid_mesh_path
    if args.hybrid_mesh_backend:
        pp.hybrid_mesh_backend = args.hybrid_mesh_backend
    if args.hybrid_skybox_path:
        pp.hybrid_skybox_path = args.hybrid_skybox_path
    if args.hybrid_save_buffers:
        pp.hybrid_save_buffers = True
    if args.hybrid_mesh_scale is not None:
        pp.hybrid_mesh_scale = args.hybrid_mesh_scale[0] if len(args.hybrid_mesh_scale) == 1 else args.hybrid_mesh_scale
    if args.hybrid_mesh_rotation_deg is not None:
        pp.hybrid_mesh_rotation_deg = args.hybrid_mesh_rotation_deg
    if args.hybrid_mesh_translation is not None:
        pp.hybrid_mesh_translation = args.hybrid_mesh_translation
    if args.hybrid_mesh_center_world is not None:
        pp.hybrid_mesh_center_world = args.hybrid_mesh_center_world
    if args.hybrid_debug_cube:
        pp.hybrid_debug_cube = True
        pp.hybrid_debug_cube_size = args.hybrid_debug_cube_size
        pp.hybrid_debug_cube_distance = args.hybrid_debug_cube_distance
    if args.hybrid_debug_placement:
        pp.hybrid_debug_placement = args.hybrid_debug_placement
    if args.hybrid_debug_scene_margin >= 0.0:
        pp.hybrid_debug_scene_margin = args.hybrid_debug_scene_margin
    if args.hybrid_debug_min_distance >= 0.0:
        pp.hybrid_debug_min_distance = args.hybrid_debug_min_distance
    if args.hybrid_auto_place_mesh:
        pp.hybrid_auto_place_mesh = True
    if args.hybrid_use_gs_env_light:
        pp.hybrid_use_gs_env_light = True
    if args.hybrid_mesh_env_resolution > 0:
        pp.hybrid_mesh_env_resolution = args.hybrid_mesh_env_resolution
    if args.hybrid_mesh_env_diffuse_strength >= 0.0:
        pp.hybrid_mesh_env_diffuse_strength = args.hybrid_mesh_env_diffuse_strength
    if args.hybrid_mesh_env_ambient_strength >= 0.0:
        pp.hybrid_mesh_env_ambient_strength = args.hybrid_mesh_env_ambient_strength
    if args.hybrid_verbose:
        pp.hybrid_verbose = True
    pp.fix_lod = args.fix_lod if args.fix_lod >= 0 else getattr(pp, "fix_lod", -1)
    if len(args.target_view) > 0:
        TARGET_RENDER_IMAGE_PATHS = set(args.target_view)
    elif args.camera_path_mode == "interpolate" and args.camera_path_start_view and args.camera_path_end_view:
        lp.target_views = [args.camera_path_start_view, args.camera_path_end_view]
    elif args.camera_path_mode == "orbit" and args.camera_path_reference_view:
        lp.target_views = [args.camera_path_reference_view]
    if args.max_views > 0:
        TARGET_RENDER_MAX_VIEWS = args.max_views
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(lp, op, pp, args.iteration, args.skip_train, args.skip_test, args.ape, args.explicit, render_motion_vectors=args.render_motion_vectors, write_per_view_count=args.write_per_view_count, show_fps=args.showFPS, render_point_cloud=args.render_point_cloud)
