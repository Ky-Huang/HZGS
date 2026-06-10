import json
import os
import socket

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from xr.frame_sources import SocketFrameSource, load_xr_frames
from xr.openxr_bridge import build_minicam_from_openxr_view, get_openxr_view_dimensions, load_xr_session_config


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
        os.path.join(render_dir, name)
        for name in os.listdir(render_dir)
        if name.lower().endswith(".png")
    )
    if not frame_paths:
        return False

    writer = imageio.get_writer(
        output_path,
        fps=int(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame_path in frame_paths:
            frame = _pad_frame_to_block(_read_rgb_frame(frame_path), 16)
            writer.append_data(frame)
    finally:
        writer.close()
    return True


def _profile_enabled_from_env():
    return os.environ.get("HGS_XR_PROFILE", "").strip().lower() in {"1", "true", "yes", "on"}


def _slow_render_threshold_ms():
    try:
        return float(os.environ.get("HGS_XR_SLOW_RENDER_MS", "150"))
    except ValueError:
        return 150.0


def _slow_frame_log_path(model_path):
    path = os.environ.get("HGS_XR_SLOW_FRAMES_PATH", "").strip()
    if path:
        return path
    return os.path.join(model_path, "xr_slow_frames.jsonl")


def _match_resolution_scale_to_swapchain(config, frame, enabled):
    if not enabled:
        return False
    try:
        swapchain_scale = float(frame.get("swapchain_scale", 0.0))
    except (TypeError, ValueError):
        return False
    if swapchain_scale <= 0.0:
        return False

    matched_resolution_scale = 1.0 / swapchain_scale
    previous = float(config.get("resolution_scale", 1.0))
    config["resolution_scale"] = matched_resolution_scale
    return abs(previous - matched_resolution_scale) > 1e-6


def _preflight_log_enabled_from_env():
    return os.environ.get("HGS_XR_PREFLIGHT_LOG", "").strip().lower() in {"1", "true", "yes", "on"}


def _preflight_guard_enabled_from_env():
    return os.environ.get("HGS_XR_PREFLIGHT_GUARD", "").strip().lower() in {"1", "true", "yes", "on"}


def _preflight_log_every_from_env():
    try:
        return max(int(os.environ.get("HGS_XR_PREFLIGHT_LOG_EVERY", "1")), 1)
    except ValueError:
        return 1


def _preflight_int_limit_from_env(name, default):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = int(default)
    return value if value > 0 else None


def _preflight_float_limit_from_env(name, default):
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = float(default)
    return value if value > 0.0 else None


def _preflight_guard_cooldown_seconds_from_env():
    try:
        cooldown_ms = float(os.environ.get("HGS_XR_PREFLIGHT_GUARD_COOLDOWN_MS", "750"))
    except ValueError:
        cooldown_ms = 750.0
    return max(cooldown_ms, 0.0) / 1000.0


def _format_preflight_value(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace(" ", "_")


def _collect_preflight_eye_stats(eye, viewpoint_camera, gaussians, pipe):
    import time

    from gaussian_renderer.render import prefilter_voxel

    t0 = time.perf_counter()
    stats = {
        "eye": eye,
        "size": f"{int(viewpoint_camera.image_width)}x{int(viewpoint_camera.image_height)}",
        "resolution_scale": float(getattr(viewpoint_camera, "resolution_scale", 1.0)),
        "image_type": getattr(viewpoint_camera, "image_type", ""),
        "add_prefilter": int(bool(getattr(pipe, "add_prefilter", False))),
    }

    if getattr(gaussians, "explicit_gs", False):
        gaussians.set_gs_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        selected_count = int(gaussians._gs_mask.sum().item())
        stats.update(
            {
                "model": "explicit",
                "total_gaussians": int(getattr(gaussians, "_xyz").shape[0]),
                "lod_selected": selected_count,
                "prefilter_selected": selected_count,
                "estimated_gaussians": selected_count,
            }
        )
    else:
        gaussians.set_anchor_mask(viewpoint_camera.camera_center, viewpoint_camera.resolution_scale)
        lod_selected = int(gaussians._anchor_mask.sum().item())
        n_offsets = int(getattr(gaussians, "n_offsets", 1))
        prefilter_selected = lod_selected
        prefilter_stats = {}

        if bool(getattr(pipe, "add_prefilter", False)) and lod_selected > 0:
            visible_mask, prefilter_stats = prefilter_voxel(viewpoint_camera, gaussians, return_stats=True)
            prefilter_selected = int(visible_mask.sum().item())

        stats.update(
            {
                "model": "anchor",
                "total_anchors": int(gaussians.get_anchor.shape[0]),
                "n_offsets": n_offsets,
                "lod_selected": lod_selected,
                "prefilter_selected": prefilter_selected,
                "estimated_gaussians": prefilter_selected * n_offsets,
            }
        )
        if prefilter_stats:
            stats.update(
                {
                    "anchor_raster_visible": int(prefilter_stats.get("raster_visible", 0)),
                    "anchor_radii_max": float(prefilter_stats.get("radii_max", 0.0)),
                    "anchor_radii_p95": float(prefilter_stats.get("radii_p95", 0.0)),
                    "anchor_radii2_sum": float(prefilter_stats.get("radii2_sum", 0.0)),
                }
            )

    stats["preflight_ms"] = (time.perf_counter() - t0) * 1000.0
    return stats


def _log_preflight_frame(frame, left_cam, right_cam, gaussians, pipe):
    frame_id = int(frame.get("frame_id", 0))
    every = _preflight_log_every_from_env()
    if every > 1 and frame_id % every != 0:
        return

    field_order = [
        "frame_id",
        "eye",
        "model",
        "size",
        "resolution_scale",
        "image_type",
        "add_prefilter",
        "total_anchors",
        "total_gaussians",
        "n_offsets",
        "lod_selected",
        "prefilter_selected",
        "estimated_gaussians",
        "anchor_raster_visible",
        "anchor_radii_p95",
        "anchor_radii_max",
        "anchor_radii2_sum",
        "preflight_ms",
        "error",
    ]

    for eye, camera in (("left", left_cam), ("right", right_cam)):
        try:
            stats = _collect_preflight_eye_stats(eye, camera, gaussians, pipe)
            stats["frame_id"] = frame_id
        except Exception as exc:
            stats = {
                "frame_id": frame_id,
                "eye": eye,
                "error": f"{type(exc).__name__}:{str(exc)}",
            }
        parts = []
        for key in field_order:
            if key in stats:
                parts.append(f"{key}={_format_preflight_value(stats[key])}")
        print("[xr-preflight] " + " ".join(parts), flush=True)


def _collect_preflight_frame_stats(left_cam, right_cam, gaussians, pipe):
    stats_list = []
    for eye, camera in (("left", left_cam), ("right", right_cam)):
        try:
            stats_list.append(_collect_preflight_eye_stats(eye, camera, gaussians, pipe))
        except Exception as exc:
            stats_list.append(
                {
                    "eye": eye,
                    "error": f"{type(exc).__name__}:{str(exc)}",
                }
            )
    return stats_list


def _log_preflight_stats(frame, stats_list):
    frame_id = int(frame.get("frame_id", 0))
    every = _preflight_log_every_from_env()
    if every > 1 and frame_id % every != 0:
        return

    field_order = [
        "frame_id",
        "eye",
        "model",
        "size",
        "resolution_scale",
        "image_type",
        "add_prefilter",
        "total_anchors",
        "total_gaussians",
        "n_offsets",
        "lod_selected",
        "prefilter_selected",
        "estimated_gaussians",
        "anchor_raster_visible",
        "anchor_radii_p95",
        "anchor_radii_max",
        "anchor_radii2_sum",
        "preflight_ms",
        "error",
    ]
    for stats in stats_list:
        row = dict(stats)
        row["frame_id"] = frame_id
        parts = []
        for key in field_order:
            if key in row:
                parts.append(f"{key}={_format_preflight_value(row[key])}")
        print("[xr-preflight] " + " ".join(parts), flush=True)


def _preflight_guard_reason(stats_list):
    if not _preflight_guard_enabled_from_env():
        return ""

    max_prefilter_selected = _preflight_int_limit_from_env(
        "HGS_XR_PREFLIGHT_MAX_PREFILTER_SELECTED",
        850000,
    )
    max_estimated_gaussians = _preflight_int_limit_from_env(
        "HGS_XR_PREFLIGHT_MAX_ESTIMATED_GAUSSIANS",
        8500000,
    )
    max_anchor_radii2_sum = _preflight_float_limit_from_env(
        "HGS_XR_PREFLIGHT_MAX_ANCHOR_RADII2_SUM",
        0.0,
    )
    max_anchor_radii = _preflight_float_limit_from_env(
        "HGS_XR_PREFLIGHT_MAX_ANCHOR_RADII",
        0.0,
    )

    reasons = []
    for stats in stats_list:
        eye = stats.get("eye", "?")
        checks = [
            ("prefilter_selected", max_prefilter_selected),
            ("estimated_gaussians", max_estimated_gaussians),
            ("anchor_radii2_sum", max_anchor_radii2_sum),
            ("anchor_radii_max", max_anchor_radii),
        ]
        for key, limit in checks:
            if limit is None or key not in stats:
                continue
            value = stats[key]
            if value > limit:
                reasons.append(
                    f"{eye}.{key}={_format_preflight_value(value)}>{_format_preflight_value(limit)}"
                )
    return ",".join(reasons)


def _preflight_guard_config_summary():
    if not _preflight_guard_enabled_from_env():
        return "disabled"
    values = {
        "max_prefilter_selected": _preflight_int_limit_from_env(
            "HGS_XR_PREFLIGHT_MAX_PREFILTER_SELECTED",
            850000,
        ),
        "max_estimated_gaussians": _preflight_int_limit_from_env(
            "HGS_XR_PREFLIGHT_MAX_ESTIMATED_GAUSSIANS",
            8500000,
        ),
        "max_anchor_radii2_sum": _preflight_float_limit_from_env(
            "HGS_XR_PREFLIGHT_MAX_ANCHOR_RADII2_SUM",
            0.0,
        ),
        "max_anchor_radii": _preflight_float_limit_from_env(
            "HGS_XR_PREFLIGHT_MAX_ANCHOR_RADII",
            0.0,
        ),
        "cooldown_ms": _preflight_guard_cooldown_seconds_from_env() * 1000.0,
    }
    return "enabled " + " ".join(
        f"{key}={_format_preflight_value(value)}"
        for key, value in values.items()
        if value is not None
    )


def _fallback_rgb_for_camera(viewpoint_camera, background, cached_rgb=None):
    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    if cached_rgb is not None and tuple(cached_rgb.shape[-2:]) == (height, width):
        return cached_rgb
    bg = torch.clamp(background.detach(), 0.0, 1.0)
    if bg.ndim == 1:
        bg = bg[:, None, None]
    return bg[:3].expand(3, height, width).contiguous()


def _render_stereo_cameras(left_cam, right_cam, gaussians, pipe, background, render_fn, profile_meter=None):
    profiler = profile_meter if profile_meter is not None and profile_meter.enabled else None
    if profiler is not None:
        t0 = profiler.now()
    left_pkg = render_fn(left_cam, gaussians, pipe, background)
    left_rgb = torch.clamp(left_pkg["render"], 0.0, 1.0)
    if profiler is not None:
        profiler.sync_cuda()
        elapsed_ms = profiler.elapsed_ms(t0)
        profiler.record("left_render", elapsed_ms)
        profiler.note_slow_render("left", elapsed_ms)

    if profiler is not None:
        t0 = profiler.now()
    right_pkg = render_fn(right_cam, gaussians, pipe, background)
    right_rgb = torch.clamp(right_pkg["render"], 0.0, 1.0)
    if profiler is not None:
        profiler.sync_cuda()
        elapsed_ms = profiler.elapsed_ms(t0)
        profiler.record("right_render", elapsed_ms)
        profiler.note_slow_render("right", elapsed_ms)
    return left_rgb, right_rgb


def _render_stereo_frame(frame, config, gaussians, pipe, background, render_fn, profile_meter=None):
    profiler = profile_meter if profile_meter is not None and profile_meter.enabled else None
    if profiler is not None:
        t0 = profiler.now()
    left_cam = build_minicam_from_openxr_view(frame, "left", config)
    right_cam = build_minicam_from_openxr_view(frame, "right", config)
    if profiler is not None:
        profiler.record("camera", profiler.elapsed_ms(t0))

    if _preflight_log_enabled_from_env():
        _log_preflight_frame(frame, left_cam, right_cam, gaussians, pipe)

    if profiler is not None:
        t0 = profiler.now()
    left_pkg = render_fn(left_cam, gaussians, pipe, background)
    left_rgb = torch.clamp(left_pkg["render"], 0.0, 1.0)
    if profiler is not None:
        profiler.sync_cuda()
        elapsed_ms = profiler.elapsed_ms(t0)
        profiler.record("left_render", elapsed_ms)
        profiler.note_slow_render("left", elapsed_ms)

    if profiler is not None:
        t0 = profiler.now()
    right_pkg = render_fn(right_cam, gaussians, pipe, background)
    right_rgb = torch.clamp(right_pkg["render"], 0.0, 1.0)
    if profiler is not None:
        profiler.sync_cuda()
        elapsed_ms = profiler.elapsed_ms(t0)
        profiler.record("right_render", elapsed_ms)
        profiler.note_slow_render("right", elapsed_ms)
    return left_cam, right_cam, left_rgb, right_rgb


def _validate_stream_upsample_scale(value):
    try:
        scale = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid XR stream upsample scale: {value!r}.")
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("--xr_stream_upsample_scale must be finite and >= 1.0.")
    return scale


def _validate_stream_upsample_mode(value):
    mode = str(value or "bilinear").strip().lower()
    valid_modes = {"nearest", "bilinear", "bicubic"}
    if mode not in valid_modes:
        raise ValueError(
            f"--xr_stream_upsample_mode must be one of {sorted(valid_modes)}, got {value!r}."
        )
    return mode


def _resize_tensor_image(image, output_size=None, mode="bilinear"):
    if output_size is None:
        return image
    output_height, output_width = int(output_size[0]), int(output_size[1])
    if output_width <= 0 or output_height <= 0:
        raise ValueError("XR stream output size must be positive.")
    if tuple(image.shape[-2:]) == (output_height, output_width):
        return image

    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(
        image.unsqueeze(0),
        size=(output_height, output_width),
        **kwargs,
    )[0]


def _tensor_to_rgba8_bytes(image, output_size=None, upsample_mode="bilinear"):
    image = torch.clamp(image.detach(), 0.0, 1.0)
    image = _resize_tensor_image(image, output_size=output_size, mode=upsample_mode)
    image = torch.clamp(image, 0.0, 1.0)
    if image.shape[0] == 3:
        alpha = torch.ones((1, image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)
        image = torch.cat([image, alpha], dim=0)
    elif image.shape[0] != 4:
        raise ValueError(f"Expected 3 or 4 image channels, got {image.shape[0]}.")

    image_u8 = (image.permute(1, 2, 0).contiguous() * 255.0).byte().cpu().numpy()
    height, width = image_u8.shape[:2]
    return image_u8.tobytes(), width, height


class _StreamFpsMeter:
    def __init__(self, log_interval=1.0):
        import time

        self._time = time
        self.log_interval = float(log_interval)
        self.start_time = None
        self.last_frame_time = None
        self.last_log_time = self._time.perf_counter()
        self.total_frames = 0
        self.current_fps = 0.0

    def mark_frame_sent(self):
        now = self._time.perf_counter()
        if self.start_time is None:
            self.start_time = now
        if self.last_frame_time is not None:
            elapsed = now - self.last_frame_time
            if elapsed > 1e-6:
                instant_fps = 1.0 / elapsed
                if self.current_fps <= 0.0:
                    self.current_fps = instant_fps
                else:
                    self.current_fps = self.current_fps * 0.85 + instant_fps * 0.15
        self.last_frame_time = now
        self.total_frames += 1
        return now

    def average_fps(self, now=None):
        if self.start_time is None or self.total_frames <= 1:
            return 0.0
        now = self._time.perf_counter() if now is None else now
        elapsed = now - self.start_time
        if elapsed <= 1e-6:
            return 0.0
        return float(self.total_frames - 1) / elapsed

    def should_log(self, now):
        if now - self.last_log_time < self.log_interval:
            return False
        self.last_log_time = now
        return True


class _StreamProfileMeter:
    def __init__(self, enabled=False):
        import time

        self.enabled = bool(enabled)
        self._time = time
        self._totals = {}
        self._frames = 0
        self._slow_render_threshold_ms = _slow_render_threshold_ms()
        self._slow_render_events = []

    def now(self):
        return self._time.perf_counter()

    def elapsed_ms(self, start_time):
        return (self.now() - start_time) * 1000.0

    def sync_cuda(self):
        if self.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()

    def record(self, name, elapsed_ms):
        if not self.enabled:
            return
        self._totals[name] = self._totals.get(name, 0.0) + float(elapsed_ms)

    def note_slow_render(self, eye, elapsed_ms):
        if not self.enabled or elapsed_ms < self._slow_render_threshold_ms:
            return
        self._slow_render_events.append(
            {
                "eye": str(eye),
                "elapsed_ms": round(float(elapsed_ms), 3),
                "threshold_ms": round(float(self._slow_render_threshold_ms), 3),
            }
        )

    def take_slow_render_events(self):
        events = self._slow_render_events
        self._slow_render_events = []
        return events

    def mark_frame(self):
        if self.enabled:
            self._frames += 1

    def summary(self):
        if not self.enabled or self._frames <= 0:
            return ""
        names = [
            ("wait", "wait_pose"),
            ("camera", "camera"),
            ("left", "left_render"),
            ("right", "right_render"),
            ("pack", "pack_rgba"),
            ("send", "send_socket"),
            ("busy", "busy_total"),
        ]
        parts = []
        for label, key in names:
            value = self._totals.get(key, 0.0) / self._frames
            parts.append(f"{label}={value:.2f}")
        return "profile_ms " + " ".join(parts)

    def reset_interval(self):
        if self.enabled:
            self._totals.clear()
            self._frames = 0


def _sendall(conn, payload):
    conn.sendall(payload)


def _run_openxr_stream_session(
    model_path,
    gaussians,
    pipe,
    background,
    render_fn,
    xr_config_path,
    xr_socket_host,
    xr_socket_port,
    xr_max_frames,
    xr_match_swapchain_resolution_scale,
    xr_stream_upsample_scale,
    xr_stream_upsample_mode,
):
    config = load_xr_session_config(xr_config_path)
    xr_stream_upsample_scale = _validate_stream_upsample_scale(xr_stream_upsample_scale)
    xr_stream_upsample_mode = _validate_stream_upsample_mode(xr_stream_upsample_mode)
    reported_matched_resolution_scale = False
    reported_missing_swapchain_scale = False
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((xr_socket_host, int(xr_socket_port)))
    listener.listen(1)
    print(f"[openxr-stream] waiting for client on {xr_socket_host}:{xr_socket_port}")
    try:
        conn, addr = listener.accept()
        print(f"[openxr-stream] connected by {addr}")
        print(f"[openxr-stream] preflight_guard={_preflight_guard_config_summary()}", flush=True)
        if xr_stream_upsample_scale > 1.0:
            print(
                "[openxr-stream] stream upsample enabled: "
                f"render_size=swapchain/{xr_stream_upsample_scale:g} "
                f"output_size=swapchain mode={xr_stream_upsample_mode}",
                flush=True,
            )
        rendered_count = 0
        fps_meter = _StreamFpsMeter()
        profile_meter = _StreamProfileMeter(_profile_enabled_from_env())
        skipped_count = 0
        last_left_rgb = None
        last_right_rgb = None
        guard_cooldown_until = 0.0
        last_guard_reason = ""
        slow_frame_log = None
        if profile_meter.enabled:
            print("[openxr-stream] profiling enabled via HGS_XR_PROFILE=1")
            slow_frame_path = _slow_frame_log_path(model_path)
            slow_frame_dir = os.path.dirname(slow_frame_path)
            if slow_frame_dir:
                os.makedirs(slow_frame_dir, exist_ok=True)
            slow_frame_log = open(slow_frame_path, "w", encoding="utf-8")
            print(f"[openxr-stream] slow frame log: {slow_frame_path}")
        with conn:
            reader = conn.makefile("rb")
            while True:
                if profile_meter.enabled:
                    t_wait = profile_meter.now()
                line = reader.readline()
                if profile_meter.enabled:
                    profile_meter.record("wait_pose", profile_meter.elapsed_ms(t_wait))
                if not line:
                    break
                payload = json.loads(line.decode("utf-8"))
                if isinstance(payload, dict) and payload.get("type") == "eos":
                    break
                if xr_max_frames > 0 and rendered_count >= xr_max_frames:
                    break
                if _match_resolution_scale_to_swapchain(config, payload, xr_match_swapchain_resolution_scale):
                    if not reported_matched_resolution_scale:
                        print(
                            "[openxr-stream] matched resolution_scale to "
                            f"swapchain_scale={float(payload['swapchain_scale']):.3f}: "
                            f"resolution_scale={float(config['resolution_scale']):.3f}",
                            flush=True,
                        )
                        reported_matched_resolution_scale = True
                elif xr_match_swapchain_resolution_scale and "swapchain_scale" not in payload:
                    if not reported_missing_swapchain_scale:
                        print(
                            "[openxr-stream] --xr_match_swapchain_resolution_scale is enabled, "
                            "but the OpenXR payload has no swapchain_scale. Rebuild openxr_cuda_demo.",
                            flush=True,
                        )
                        reported_missing_swapchain_scale = True

                frame_id = int(payload.get("frame_id", rendered_count))
                if profile_meter.enabled:
                    t_busy = profile_meter.now()
                    t_camera = profile_meter.now()
                left_output_width, left_output_height = get_openxr_view_dimensions(payload, "left", config)
                right_output_width, right_output_height = get_openxr_view_dimensions(payload, "right", config)
                left_cam = build_minicam_from_openxr_view(
                    payload,
                    "left",
                    config,
                    resolution_divisor=xr_stream_upsample_scale,
                )
                right_cam = build_minicam_from_openxr_view(
                    payload,
                    "right",
                    config,
                    resolution_divisor=xr_stream_upsample_scale,
                )
                if profile_meter.enabled:
                    profile_meter.record("camera", profile_meter.elapsed_ms(t_camera))

                preflight_stats = None
                cooldown_now = profile_meter.now()
                guard_cooldown_remaining = max(0.0, guard_cooldown_until - cooldown_now)
                guard_in_cooldown = _preflight_guard_enabled_from_env() and guard_cooldown_remaining > 0.0
                if guard_in_cooldown:
                    skip_reason = (
                        f"cooldown_remaining_ms={guard_cooldown_remaining * 1000.0:.1f}"
                        + (f",last_reason={last_guard_reason}" if last_guard_reason else "")
                    )
                elif _preflight_log_enabled_from_env() or _preflight_guard_enabled_from_env():
                    preflight_stats = _collect_preflight_frame_stats(left_cam, right_cam, gaussians, pipe)
                    if _preflight_log_enabled_from_env():
                        _log_preflight_stats(payload, preflight_stats)
                    skip_reason = _preflight_guard_reason(preflight_stats or [])
                    if skip_reason:
                        last_guard_reason = skip_reason
                        cooldown_seconds = _preflight_guard_cooldown_seconds_from_env()
                        if cooldown_seconds > 0.0:
                            guard_cooldown_until = profile_meter.now() + cooldown_seconds
                else:
                    skip_reason = ""
                if skip_reason:
                    skipped_count += 1
                    left_rgb = _fallback_rgb_for_camera(left_cam, background, last_left_rgb)
                    right_rgb = _fallback_rgb_for_camera(right_cam, background, last_right_rgb)
                    action = "cooldown_reuse" if guard_in_cooldown else "reuse_or_background"
                    print(
                        f"[xr-guard] frame_id={frame_id} action={action} "
                        f"skipped={skipped_count} reason={skip_reason}",
                        flush=True,
                    )
                else:
                    left_rgb, right_rgb = _render_stereo_cameras(
                        left_cam,
                        right_cam,
                        gaussians,
                        pipe,
                        background,
                        render_fn,
                        profile_meter,
                    )
                    last_left_rgb = left_rgb.detach()
                    last_right_rgb = right_rgb.detach()
                slow_render_events = profile_meter.take_slow_render_events()
                if slow_frame_log is not None and slow_render_events:
                    slow_frame_log.write(
                        json.dumps(
                            {
                                "frame_id": frame_id,
                                "events": slow_render_events,
                                "payload": payload,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    slow_frame_log.flush()
                if profile_meter.enabled:
                    t_pack = profile_meter.now()
                left_bytes, left_width, left_height = _tensor_to_rgba8_bytes(
                    left_rgb,
                    output_size=(left_output_height, left_output_width),
                    upsample_mode=xr_stream_upsample_mode,
                )
                right_bytes, right_width, right_height = _tensor_to_rgba8_bytes(
                    right_rgb,
                    output_size=(right_output_height, right_output_width),
                    upsample_mode=xr_stream_upsample_mode,
                )
                if profile_meter.enabled:
                    profile_meter.record("pack_rgba", profile_meter.elapsed_ms(t_pack))
                if left_width != right_width or left_height != right_height:
                    raise ValueError("Left and right stream images must have matching dimensions.")

                header = (
                    f"HGSFRAME {frame_id} {left_width} {left_height} 4 "
                    f"{len(left_bytes)} {len(right_bytes)}\n"
                ).encode("ascii")
                if profile_meter.enabled:
                    t_send = profile_meter.now()
                _sendall(conn, header)
                _sendall(conn, left_bytes)
                _sendall(conn, right_bytes)
                if profile_meter.enabled:
                    profile_meter.record("send_socket", profile_meter.elapsed_ms(t_send))
                    profile_meter.record("busy_total", profile_meter.elapsed_ms(t_busy))
                    profile_meter.mark_frame()
                rendered_count += 1
                now = fps_meter.mark_frame_sent()
                if fps_meter.should_log(now):
                    profile_summary = profile_meter.summary()
                    profile_suffix = f" {profile_summary}" if profile_summary else ""
                    print(
                        f"[openxr-stream] fps={fps_meter.current_fps:.1f} "
                        f"avg={fps_meter.average_fps(now):.1f} sent={rendered_count} "
                        f"last_frame={frame_id} ({left_width}x{left_height})"
                        f"{profile_suffix}",
                        flush=True,
                    )
                    profile_meter.reset_interval()
    finally:
        if "slow_frame_log" in locals() and slow_frame_log is not None:
            slow_frame_log.close()
        listener.close()

    if "fps_meter" in locals() and fps_meter.total_frames > 0:
        print(
            f"[openxr-stream] session ended, sent={fps_meter.total_frames} "
            f"avg_fps={fps_meter.average_fps():.1f}"
        )
    else:
        print("[openxr-stream] session ended")
    return True


def _iter_frames(xr_mode, xr_input, xr_socket_host, xr_socket_port):
    if xr_mode == "openxr_replay":
        for frame in load_xr_frames(xr_input):
            yield frame
        return

    if xr_mode == "openxr_socket":
        with SocketFrameSource(xr_socket_host, xr_socket_port) as source:
            for frame in source:
                yield frame
        return

    raise ValueError(f"Unsupported XR mode: {xr_mode}")


def run_openxr_render_session(
    model_path,
    iteration,
    gaussians,
    pipe,
    background,
    render_fn,
    xr_mode,
    xr_input="",
    xr_config_path="",
    xr_output_name="openxr",
    xr_output_layout="both",
    xr_save_video=False,
    xr_video_fps=30,
    xr_socket_host="127.0.0.1",
    xr_socket_port=6110,
    xr_max_frames=-1,
    xr_match_swapchain_resolution_scale=False,
    xr_stream_upsample_scale=1.0,
    xr_stream_upsample_mode="bilinear",
):
    if xr_mode == "openxr_stream":
        return _run_openxr_stream_session(
            model_path=model_path,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            render_fn=render_fn,
            xr_config_path=xr_config_path,
            xr_socket_host=xr_socket_host,
            xr_socket_port=xr_socket_port,
            xr_max_frames=xr_max_frames,
            xr_match_swapchain_resolution_scale=xr_match_swapchain_resolution_scale,
            xr_stream_upsample_scale=xr_stream_upsample_scale,
            xr_stream_upsample_mode=xr_stream_upsample_mode,
        )

    config = load_xr_session_config(xr_config_path)
    reported_matched_resolution_scale = False
    reported_missing_swapchain_scale = False
    output_root = os.path.join(model_path, xr_output_name, f"ours_{iteration}")
    left_dir = os.path.join(output_root, "left_eye")
    right_dir = os.path.join(output_root, "right_eye")
    sbs_dir = os.path.join(output_root, "side_by_side")
    raw_frames_path = os.path.join(output_root, "xr_input_frames.jsonl")
    for path in [left_dir, right_dir]:
        os.makedirs(path, exist_ok=True)
    if xr_output_layout in {"side_by_side", "both"}:
        os.makedirs(sbs_dir, exist_ok=True)

    frame_records = []
    iterator = _iter_frames(xr_mode, xr_input, xr_socket_host, xr_socket_port)
    total = None if xr_mode == "openxr_socket" else max(int(xr_max_frames), 0) if xr_max_frames > 0 else None
    progress = tqdm(iterator, total=total, desc="OpenXR rendering")
    with open(raw_frames_path, "w", encoding="utf-8") as raw_frame_log:
        for frame_idx, frame in enumerate(progress):
            if xr_max_frames > 0 and frame_idx >= xr_max_frames:
                break

            raw_frame_log.write(json.dumps(frame, separators=(",", ":")) + "\n")
            raw_frame_log.flush()
            if _match_resolution_scale_to_swapchain(config, frame, xr_match_swapchain_resolution_scale):
                if not reported_matched_resolution_scale:
                    print(
                        "[openxr] matched resolution_scale to "
                        f"swapchain_scale={float(frame['swapchain_scale']):.3f}: "
                        f"resolution_scale={float(config['resolution_scale']):.3f}",
                        flush=True,
                    )
                    reported_matched_resolution_scale = True
            elif xr_match_swapchain_resolution_scale and "swapchain_scale" not in frame:
                if not reported_missing_swapchain_scale:
                    print(
                        "[openxr] --xr_match_swapchain_resolution_scale is enabled, "
                        "but the XR frame has no swapchain_scale.",
                        flush=True,
                    )
                    reported_missing_swapchain_scale = True

            frame_id = int(frame.get("frame_id", frame_idx))
            left_cam, right_cam, left_rgb, right_rgb = _render_stereo_frame(
                frame,
                config,
                gaussians,
                pipe,
                background,
                render_fn,
            )

            left_path = os.path.join(left_dir, f"{frame_id:05d}.png")
            right_path = os.path.join(right_dir, f"{frame_id:05d}.png")
            torchvision.utils.save_image(left_rgb, left_path)
            torchvision.utils.save_image(right_rgb, right_path)

            if xr_output_layout in {"side_by_side", "both"}:
                sbs_path = os.path.join(sbs_dir, f"{frame_id:05d}.png")
                torchvision.utils.save_image(torch.cat([left_rgb, right_rgb], dim=2), sbs_path)

            frame_records.append(
                {
                    "frame_id": frame_id,
                    "timestamp_ns": frame.get("timestamp_ns"),
                    "left_path": os.path.relpath(left_path, output_root),
                    "right_path": os.path.relpath(right_path, output_root),
                    "left_camera": {
                        "center": [float(x) for x in left_cam.camera_center.detach().cpu().tolist()],
                        "fx": float(left_cam.fx),
                        "fy": float(left_cam.fy),
                        "cx": float(left_cam.cx),
                        "cy": float(left_cam.cy),
                        "width": int(left_cam.image_width),
                        "height": int(left_cam.image_height),
                    },
                    "right_camera": {
                        "center": [float(x) for x in right_cam.camera_center.detach().cpu().tolist()],
                        "fx": float(right_cam.fx),
                        "fy": float(right_cam.fy),
                        "cx": float(right_cam.cx),
                        "cy": float(right_cam.cy),
                        "width": int(right_cam.image_width),
                        "height": int(right_cam.image_height),
                    },
                }
            )

    manifest = {
        "schema_version": 1,
        "xr_mode": xr_mode,
        "xr_input": xr_input,
        "xr_config_path": xr_config_path,
        "raw_input_frames_path": os.path.relpath(raw_frames_path, output_root),
        "output_layout": xr_output_layout,
        "frame_count": len(frame_records),
        "frames": frame_records,
    }
    with open(os.path.join(output_root, "xr_session_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if xr_save_video:
        if _write_video_from_render_dir(left_dir, os.path.join(output_root, "left_eye.mp4"), xr_video_fps):
            print(f"[openxr] wrote video: {os.path.join(output_root, 'left_eye.mp4')}")
        if _write_video_from_render_dir(right_dir, os.path.join(output_root, "right_eye.mp4"), xr_video_fps):
            print(f"[openxr] wrote video: {os.path.join(output_root, 'right_eye.mp4')}")
        if xr_output_layout in {"side_by_side", "both"}:
            if _write_video_from_render_dir(sbs_dir, os.path.join(output_root, "side_by_side.mp4"), xr_video_fps):
                print(f"[openxr] wrote video: {os.path.join(output_root, 'side_by_side.mp4')}")

    print(f"[openxr] rendered {len(frame_records)} stereo frames into {output_root}")
    return True
