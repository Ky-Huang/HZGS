"""
Create an XR config by aligning a captured OpenXR pose to a known scene camera.

Example:
  conda run -n horizon_gs_py_312_pt271_cu126 python tools/make_xr_config_from_anchor.py \
    --source_path data/real/road_subset \
    --source_image_path data/real/road_subset/images/street_cam1/TIMELAPSE_0371.JPG \
    --xr_frame outputs/horizongs/real/road_subset/fine/xr_anchor_capture_frame100/ours_40000/xr_input_frames.jsonl \
    --base_config config/xr/openxr_road_anchor_frame100.yaml \
    --output config/xr/openxr_road_anchor_new.yaml

Use --cameras_json and --camera_name instead of --source_path and
--source_image_path when the anchor camera comes from cameras.json.
The generated config aligns both start position and start orientation.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_extrinsics_text


OPENXR_CAMERA_FROM_RENDER_CAMERA = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _load_cameras(path):
    with open(path, "r", encoding="utf-8") as f:
        cameras = json.load(f)
    if not isinstance(cameras, list):
        raise ValueError("cameras.json must contain a list.")
    return cameras


def _find_camera(cameras, name):
    target = os.path.splitext(os.path.basename(name))[0].lower()
    matches = []
    for camera in cameras:
        camera_name = os.path.splitext(os.path.basename(camera["img_name"]))[0].lower()
        if camera_name == target:
            matches.append(camera)
    if not matches:
        raise ValueError(f"Camera '{name}' not found in cameras.json.")
    if len(matches) > 1:
        print(f"[calibrate] camera '{name}' matched {len(matches)} entries; using the first one.")
    return matches[0]


def _load_frame(path, frame_index):
    if frame_index < 0:
        raise ValueError("--frame_index must be non-negative.")

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            current = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if current == frame_index:
                    return json.loads(line)
                current += 1
            raise ValueError(f"Frame index {frame_index} not found in {path}.")

        payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
            if frame_index >= len(payload["frames"]):
                raise ValueError(f"Frame index {frame_index} not found in {path}.")
            return payload["frames"][frame_index]
        if isinstance(payload, list):
            if frame_index >= len(payload):
                raise ValueError(f"Frame index {frame_index} not found in {path}.")
            return payload[frame_index]
        if isinstance(payload, dict) and "views" in payload:
            if frame_index != 0:
                raise ValueError("Single-frame JSON input only supports --frame_index 0.")
            return payload
    raise ValueError(f"Unsupported XR frame file: {path}")


def _quat_xyzw_to_rotmat(quat_xyzw):
    x, y, z, w = [float(v) for v in quat_xyzw]
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
        dtype=np.float64,
    )


def _pose_to_matrix(pose):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_xyzw_to_rotmat(pose["orientation_xyzw"])
    matrix[:3, 3] = np.asarray(pose["position"], dtype=np.float64)
    return matrix


def _get_view(frame, eye):
    views = frame.get("views")
    if isinstance(views, dict):
        view = views.get(eye)
    elif isinstance(views, list):
        view = next((item for item in views if str(item.get("eye", "")).lower() == eye), None)
    else:
        view = None
    if not isinstance(view, dict):
        raise ValueError(f"XR frame is missing the '{eye}' eye view.")
    return view


def _tracking_from_anchor_view(frame, anchor_view):
    left = _get_view(frame, "left")
    right = _get_view(frame, "right")
    if anchor_view == "left":
        return _pose_to_matrix(left["pose"])
    if anchor_view == "right":
        return _pose_to_matrix(right["pose"])

    left_pose = left["pose"]
    right_pose = right["pose"]
    center_pose = {
        "position": (
            0.5
            * (
                np.asarray(left_pose["position"], dtype=np.float64)
                + np.asarray(right_pose["position"], dtype=np.float64)
            )
        ).tolist(),
        "orientation_xyzw": left_pose["orientation_xyzw"],
    }
    return _pose_to_matrix(center_pose)


def _camera_to_c2w_render(camera):
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.asarray(camera["rotation"], dtype=np.float64)
    c2w[:3, 3] = np.asarray(camera["position"], dtype=np.float64)
    return c2w


def _normalize_relpath(path):
    return os.path.normpath(path).replace("\\", "/").lower()


def _load_colmap_extrinsics(source_path):
    sparse_dir = os.path.join(source_path, "sparse", "0")
    binary_path = os.path.join(sparse_dir, "images.bin")
    text_path = os.path.join(sparse_dir, "images.txt")
    if os.path.exists(binary_path):
        return read_extrinsics_binary(binary_path)
    return read_extrinsics_text(text_path)


def _source_image_to_colmap_name(source_path, source_image_path):
    abs_source = os.path.abspath(source_path)
    abs_image = os.path.abspath(source_image_path)
    images_root = os.path.join(abs_source, "images")
    try:
        rel = os.path.relpath(abs_image, images_root)
    except ValueError:
        rel = source_image_path
    return _normalize_relpath(rel)


def _source_image_to_c2w_render(source_path, source_image_path):
    target_name = _source_image_to_colmap_name(source_path, source_image_path)
    extrinsics = _load_colmap_extrinsics(source_path)
    matches = [
        extr
        for extr in extrinsics.values()
        if _normalize_relpath(extr.name) == target_name
    ]
    if not matches:
        raise ValueError(f"Source image '{target_name}' not found in COLMAP sparse images.")
    if len(matches) > 1:
        raise ValueError(f"Source image '{target_name}' matched multiple COLMAP images.")

    extr = matches[0]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec2rotmat(extr.qvec)
    w2c[:3, 3] = np.asarray(extr.tvec, dtype=np.float64)
    return np.linalg.inv(w2c)


def _load_base_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _matrix_to_list(matrix):
    return [[float(f"{value:.9g}") for value in row] for row in matrix.tolist()]


def main():
    parser = argparse.ArgumentParser(
        description="Create an XR config that aligns the first captured OpenXR pose to a known HorizonGS camera."
    )
    parser.add_argument("--cameras_json", default="", type=str)
    parser.add_argument("--camera_name", default="", type=str)
    parser.add_argument("--source_path", default="", type=str)
    parser.add_argument("--source_image_path", default="", type=str)
    parser.add_argument("--xr_frame", required=True, type=str, help="A JSON/JSONL file containing captured OpenXR frames.")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--base_config", default="config/xr/openxr_replay_example.yaml", type=str)
    parser.add_argument("--anchor_view", default="center", choices=["center", "left", "right"])
    parser.add_argument("--frame_index", default=0, type=int)
    args = parser.parse_args()

    frame = _load_frame(args.xr_frame, args.frame_index)

    if args.source_image_path:
        if not args.source_path:
            raise ValueError("--source_image_path requires --source_path.")
        desired_scene_from_render_camera = _source_image_to_c2w_render(args.source_path, args.source_image_path)
        anchor_label = args.source_image_path
        anchor_position = desired_scene_from_render_camera[:3, 3].tolist()
    else:
        if not args.cameras_json or not args.camera_name:
            raise ValueError("Provide either --source_path/--source_image_path or --cameras_json/--camera_name.")
        cameras = _load_cameras(args.cameras_json)
        camera = _find_camera(cameras, args.camera_name)
        desired_scene_from_render_camera = _camera_to_c2w_render(camera)
        anchor_label = camera["img_name"]
        anchor_position = camera["position"]

    tracking_from_openxr_camera = _tracking_from_anchor_view(frame, args.anchor_view)
    scene_from_tracking = (
        desired_scene_from_render_camera
        @ OPENXR_CAMERA_FROM_RENDER_CAMERA
        @ np.linalg.inv(tracking_from_openxr_camera)
    )

    config = _load_base_config(args.base_config)
    config["scene_from_tracking"] = _matrix_to_list(scene_from_tracking)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"[calibrate] wrote {args.output}")
    print(f"[calibrate] anchor camera: {anchor_label}")
    print(f"[calibrate] anchor position: {anchor_position}")
    print(f"[calibrate] anchor view: {args.anchor_view}")
    print(f"[calibrate] frame index: {args.frame_index}")


if __name__ == "__main__":
    main()
