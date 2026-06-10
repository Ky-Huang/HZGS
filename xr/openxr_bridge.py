import json
import math
import os

import numpy as np
import torch
import yaml

from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrixFromIntrinsics


OPENXR_CAMERA_FROM_RENDER_CAMERA = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def load_xr_session_config(path):
    if not path:
        return {}

    with open(path, "r") as f:
        if os.path.splitext(path)[1].lower() == ".json":
            data = json.load(f)
        else:
            data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"XR config must be a mapping, got {type(data).__name__}.")
    return data


def _require_mapping(payload, key):
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid '{key}' mapping in XR payload.")
    return value


def _to_vec(values, name, size):
    if len(values) != size:
        raise ValueError(f"{name} must have length {size}, got {len(values)}.")
    return np.asarray(values, dtype=np.float32)


def _to_matrix4x4(values, name):
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {arr.shape}.")
    return arr


def _quat_xyzw_to_rotmat(quat_xyzw):
    x, y, z, w = _to_vec(quat_xyzw, "orientation_xyzw", 4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        raise ValueError("orientation_xyzw must not be zero.")
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _pose_to_matrix(pose):
    position = pose.get("position")
    orientation = pose.get("orientation_xyzw")
    if position is None or orientation is None:
        raise ValueError("Each XR view pose must contain 'position' and 'orientation_xyzw'.")

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _quat_xyzw_to_rotmat(orientation)
    c2w[:3, 3] = _to_vec(position, "position", 3)
    return c2w


def _get_scene_from_tracking(config):
    matrix = config.get("scene_from_tracking")
    if matrix is None:
        return np.eye(4, dtype=np.float32)
    return _to_matrix4x4(matrix, "scene_from_tracking")


def _get_dimensions(view, config):
    image_rect = view.get("image_rect")
    if isinstance(image_rect, dict):
        width = int(image_rect.get("width", 0))
        height = int(image_rect.get("height", 0))
    else:
        default_rect = config.get("default_image_rect", {})
        width = int(default_rect.get("width", 0))
        height = int(default_rect.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("XR view must provide image_rect.width and image_rect.height, or set default_image_rect in config.")
    return width, height


def _fov_to_intrinsics(width, height, fov, znear):
    angle_left = float(fov["angle_left"])
    angle_right = float(fov["angle_right"])
    angle_up = float(fov["angle_up"])
    angle_down = float(fov["angle_down"])

    left = znear * math.tan(angle_left)
    right = znear * math.tan(angle_right)
    top = -znear * math.tan(angle_up)
    bottom = -znear * math.tan(angle_down)

    if right <= left:
        raise ValueError(f"Invalid OpenXR FOV: right ({right}) must be greater than left ({left}).")
    if bottom <= top:
        raise ValueError(f"Invalid OpenXR FOV: bottom ({bottom}) must be greater than top ({top}).")

    fx = width * znear / (right - left)
    fy = height * znear / (bottom - top)
    cx = -left * fx / znear
    cy = -top * fy / znear
    fovx = angle_right - angle_left
    fovy = angle_up - angle_down
    return fx, fy, cx, cy, fovx, fovy


def _get_view_pose_and_fov(frame, eye):
    views = frame.get("views")
    if isinstance(views, dict):
        view = views.get(eye)
    elif isinstance(views, list):
        view = next((item for item in views if str(item.get("eye", "")).lower() == eye), None)
    else:
        view = None

    if not isinstance(view, dict):
        raise ValueError(f"XR frame is missing the '{eye}' eye view.")
    pose = _require_mapping(view, "pose")
    fov = _require_mapping(view, "fov")
    return view, pose, fov


def get_openxr_view_dimensions(frame, eye, config):
    view, _, _ = _get_view_pose_and_fov(frame, str(eye).lower())
    return _get_dimensions(view, config)


def build_minicam_from_openxr_view(frame, eye, config, device="cuda", resolution_divisor=1.0):
    eye = str(eye).lower()
    view, pose, fov = _get_view_pose_and_fov(frame, eye)

    znear = float(view.get("near_z", config.get("near_z", 0.01)))
    zfar = float(view.get("far_z", config.get("far_z", 100.0)))
    width, height = _get_dimensions(view, config)
    resolution_divisor = float(resolution_divisor)
    if not math.isfinite(resolution_divisor) or resolution_divisor < 1.0:
        resolution_divisor = 1.0
    if resolution_divisor > 1.0:
        width = max(int(round(width / resolution_divisor)), 1)
        height = max(int(round(height / resolution_divisor)), 1)
    fx, fy, cx, cy, fovx, fovy = _fov_to_intrinsics(width, height, fov, znear)

    scene_from_tracking = _get_scene_from_tracking(config)
    tracking_from_camera_openxr = _pose_to_matrix(pose)
    c2w = scene_from_tracking @ tracking_from_camera_openxr @ OPENXR_CAMERA_FROM_RENDER_CAMERA

    world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).to(device=device, dtype=torch.float32)
    projection_matrix = getProjectionMatrixFromIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        znear=znear,
        zfar=zfar,
    ).to(device=device, dtype=torch.float32)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)

    eye_image_types = config.get("eye_image_types", {})
    default_image_type = config.get("default_image_type", "street")
    image_type = eye_image_types.get(eye, default_image_type)

    return MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        projection_matrix=projection_matrix,
        resolution_scale=float(config.get("resolution_scale", 1.0)) * resolution_divisor,
        image_name=f"{eye}_{int(frame.get('frame_id', 0)):05d}.png",
        image_path=f"{eye}_{int(frame.get('frame_id', 0)):05d}.png",
        image_type=image_type,
    )
