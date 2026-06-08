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
import torch
import math
import os
import time
import gsplat
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs


def _render_profile_enabled():
    return os.environ.get("HGS_XR_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}


def _slow_render_threshold_ms():
    try:
        return float(os.environ.get("HGS_XR_SLOW_RENDER_MS", "150"))
    except ValueError:
        return 150.0


def _sync_cuda_for_profile():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _elapsed_ms(start_time):
    return (time.perf_counter() - start_time) * 1000.0


def _profile_stage(profile, name, start_time):
    if profile is None:
        return
    _sync_cuda_for_profile()
    profile["stages_ms"][name] = _elapsed_ms(start_time)


def _camera_center_list(viewpoint_camera):
    try:
        return [round(float(x), 4) for x in viewpoint_camera.camera_center.detach().cpu().tolist()]
    except Exception:
        return []


def _count_true(mask):
    if mask is None:
        return -1
    return int(mask.sum().item())


def _radii_stats(radii):
    visible = radii[radii > 0]
    if visible.numel() == 0:
        return 0, 0.0, 0.0
    return (
        int(visible.numel()),
        float(visible.max().item()),
        float(torch.quantile(visible.float(), 0.95).item()),
    )


def _maybe_log_slow_render(viewpoint_camera, pc, profile, visible_mask, selection_mask, radii):
    if profile is None:
        return
    total_ms = sum(profile["stages_ms"].values())
    if total_ms < _slow_render_threshold_ms():
        return

    visible_count, radii_max, radii_p95 = _radii_stats(radii)
    details = {
        "camera": getattr(viewpoint_camera, "image_name", ""),
        "image_type": getattr(viewpoint_camera, "image_type", ""),
        "size": f"{int(viewpoint_camera.image_width)}x{int(viewpoint_camera.image_height)}",
        "total_ms": round(total_ms, 2),
        "stages_ms": {key: round(value, 2) for key, value in profile["stages_ms"].items()},
        "lod_selected": _count_true(profile.get("lod_mask")),
        "prefilter_selected": _count_true(visible_mask),
        "generated_gaussians": int(profile.get("generated_gaussians", -1)),
        "selection_count": _count_true(selection_mask),
        "raster_visible": visible_count,
        "radii_max": round(radii_max, 2),
        "radii_p95": round(radii_p95, 2),
        "resolution_scale": float(getattr(viewpoint_camera, "resolution_scale", -1.0)),
        "camera_center": _camera_center_list(viewpoint_camera),
        "gs_attr": getattr(pc, "gs_attr", ""),
        "render_mode": getattr(pc, "render_mode", ""),
    }
    print("[render-slow] " + " ".join(f"{key}={value}" for key, value in details.items()), flush=True)

def render(viewpoint_camera, pc, pipe, bg_color):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    profile = {"stages_ms": {}} if _render_profile_enabled() else None
    if pc.explicit_gs:
        t0 = time.perf_counter() if profile is not None else None
        pc.set_gs_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        visible_mask = pc._gs_mask
        if profile is not None:
            profile["lod_mask"] = visible_mask
            _profile_stage(profile, "set_gs_mask", t0)

        t0 = time.perf_counter() if profile is not None else None
        xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_explicit_gaussians(visible_mask)
        if profile is not None:
            profile["generated_gaussians"] = int(xyz.shape[0])
            _profile_stage(profile, "generate_gaussians", t0)
    else:
        t0 = time.perf_counter() if profile is not None else None
        pc.set_anchor_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        if profile is not None:
            profile["lod_mask"] = pc._anchor_mask
            _profile_stage(profile, "set_anchor_mask", t0)

        t0 = time.perf_counter() if profile is not None else None
        visible_mask = prefilter_voxel(viewpoint_camera, pc).squeeze() if pipe.add_prefilter else pc._anchor_mask
        if profile is not None:
            _profile_stage(profile, "prefilter", t0)

        t0 = time.perf_counter() if profile is not None else None
        xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_neural_gaussians(viewpoint_camera, visible_mask)
        if profile is not None:
            profile["generated_gaussians"] = int(xyz.shape[0])
            _profile_stage(profile, "generate_gaussians", t0)

    # Set up rasterization configuration
    K = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    if pc.gs_attr == "3D":
        t0 = time.perf_counter() if profile is not None else None
        render_colors, render_alphas, info = gsplat.rasterization(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=color,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=sh_degree,
            render_mode=pc.render_mode,
        )
        if profile is not None:
            _profile_stage(profile, "raster", t0)
    elif pc.gs_attr == "2D":
        t0 = time.perf_counter() if profile is not None else None
        (render_colors, 
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,), info = \
        gsplat.rasterization_2dgs(
            means=xyz,  # [N, 3]
            quats=rot,  # [N, 4]
            scales=scaling,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=color,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=sh_degree,
            render_mode=pc.render_mode,
        )
        if profile is not None:
            _profile_stage(profile, "raster", t0)
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")

    # [1, H, W, 3] -> [3, H, W]
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    render_alphas = render_alphas[0].permute(2, 0, 1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
        "opacity": opacity,
        "render_depth": depth,
        "radii": radii,
        "render_alphas": render_alphas,
    }
    if profile is not None:
        return_dict["_render_profile"] = profile
        _maybe_log_slow_render(viewpoint_camera, pc, profile, visible_mask, selection_mask, radii)
    
    if pc.gs_attr == "2D":
        return_dict.update({
            "render_normals": render_normals,
            "render_normals_from_depth": render_normals_from_depth,
            "render_distort": render_distort,
        })

    return return_dict

def _project_gaussians_to_2d(means, quats, scales, viewpoint_camera, gs_attr):
    Ks = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    if gs_attr == "3D":
        proj_results = fully_fused_projection(
            means, None, quats, scales, viewmats, Ks,
            int(viewpoint_camera.image_width), int(viewpoint_camera.image_height),
            eps2d=0.3, packed=False, near_plane=0.01, far_plane=1e10,
            radius_clip=0.0, sparse_grad=False, calc_compensations=False,
        )
    elif gs_attr == "2D":
        C, N = viewmats.shape[0], means.shape[0]
        densifications = torch.zeros((C, N, 2), dtype=means.dtype, device="cuda")
        proj_results = fully_fused_projection_2dgs(
            means, quats, scales, viewmats, densifications, Ks,
            int(viewpoint_camera.image_width), int(viewpoint_camera.image_height),
            eps2d=0.3, packed=False, near_plane=0.01, far_plane=1e10,
            radius_clip=0.0, sparse_grad=False,
        )
    else:
        raise ValueError(f"Unknown gs_attr: {gs_attr}")

    radii, means2d, depths, conics, compensations = proj_results
    return radii.squeeze(0), means2d.squeeze(0)

def render_motion_vectors(viewpoint_camera_a, viewpoint_camera_b, pc, pipe, bg_color=None, normalize_by_alpha=True, eps=1e-6):
    if pc.explicit_gs:
        pc.set_gs_mask(viewpoint_camera_a.camera_center, viewpoint_camera_a.resolution_scale)
        visible_mask = pc._gs_mask
        xyz, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_explicit_gaussians(visible_mask)
    else:
        pc.set_anchor_mask(viewpoint_camera_a.camera_center, viewpoint_camera_a.resolution_scale)
        visible_mask = prefilter_voxel(viewpoint_camera_a, pc).squeeze() if pipe.add_prefilter else pc._anchor_mask
        xyz, offset, color, opacity, scaling, rot, sh_degree, selection_mask = pc.generate_neural_gaussians(viewpoint_camera_a, visible_mask)

    radii_a, means2d_a = _project_gaussians_to_2d(xyz, rot, scaling, viewpoint_camera_a, pc.gs_attr)
    radii_b, means2d_b = _project_gaussians_to_2d(xyz, rot, scaling, viewpoint_camera_b, pc.gs_attr)
    valid_pair = (radii_a > 0) & (radii_b > 0)

    flow = (means2d_b - means2d_a) * valid_pair[:, None].to(means2d_a.dtype)
    flow_payload = torch.cat([flow, valid_pair[:, None].to(flow.dtype)], dim=1)

    K_a = torch.tensor([
            [viewpoint_camera_a.fx, 0, viewpoint_camera_a.cx],
            [0, viewpoint_camera_a.fy, viewpoint_camera_a.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")
    viewmat_a = viewpoint_camera_a.world_view_transform.transpose(0, 1)

    flow_render, flow_alpha, info = gsplat.rasterization(
        means=xyz,
        quats=rot,
        scales=scaling,
        opacities=opacity.squeeze(-1),
        colors=flow_payload,
        viewmats=viewmat_a[None],
        Ks=K_a[None],
        backgrounds=torch.zeros((1, 3), dtype=flow.dtype, device=flow.device),
        width=int(viewpoint_camera_a.image_width),
        height=int(viewpoint_camera_a.image_height),
        packed=False,
        sh_degree=None,
        render_mode="RGB",
    )

    flow_rgb = flow_render[0].permute(2, 0, 1)
    alpha = flow_alpha[0].permute(2, 0, 1)
    motion_raw = flow_rgb[:2]
    motion = motion_raw / alpha.clamp_min(eps) if normalize_by_alpha else motion_raw

    return {
        "motion": motion,
        "motion_raw": motion_raw,
        "motion_support": flow_rgb[2:3],
        "render_alphas": alpha,
        "gaussian_valid_pair": valid_pair,
        "viewspace_points_a": means2d_a,
        "viewspace_points_b": means2d_b,
        "visibility_filter": info["radii"].squeeze(0) > 0,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
    }

def prefilter_voxel(viewpoint_camera, pc, return_stats=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    means = pc.get_anchor[pc._anchor_mask]
    scales = pc.get_scaling[pc._anchor_mask][:, :3]
    quats = pc.get_rotation[pc._anchor_mask]
    
    # Set up rasterization configuration
    Ks = torch.tensor([
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],dtype=torch.float32, device="cuda")[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    if pc.gs_attr == "3D":
        proj_results = fully_fused_projection(
            means,
            None,  # covars,
            quats,
            scales,
            viewmats,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
            calc_compensations=False,
        )
    elif pc.gs_attr == "2D":
        densifications = (
            torch.zeros((C, N, 2), dtype=means.dtype, device="cuda")
        )
        # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
        proj_results = fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            densifications,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
        )
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")
    
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None
    radii_flat = radii.squeeze(0)
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii_flat > 0

    if return_stats:
        visible_radii = radii_flat[radii_flat > 0].float()
        if visible_radii.numel() == 0:
            stats = {
                "raster_visible": 0,
                "radii_max": 0.0,
                "radii_p95": 0.0,
                "radii2_sum": 0.0,
            }
        else:
            stats = {
                "raster_visible": int(visible_radii.numel()),
                "radii_max": float(visible_radii.max().item()),
                "radii_p95": float(torch.quantile(visible_radii, 0.95).item()),
                "radii2_sum": float((visible_radii * visible_radii).sum().item()),
            }
        return visible_mask, stats

    return visible_mask
