import json
import os
import socket

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from xr.frame_sources import SocketFrameSource, load_xr_frames
from xr.openxr_bridge import build_minicam_from_openxr_view, load_xr_session_config


_OVERLAY_FONT_CACHE = {}


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


def _render_stereo_frame(frame, config, gaussians, pipe, background, render_fn):
    left_cam = build_minicam_from_openxr_view(frame, "left", config)
    right_cam = build_minicam_from_openxr_view(frame, "right", config)

    left_pkg = render_fn(left_cam, gaussians, pipe, background)
    right_pkg = render_fn(right_cam, gaussians, pipe, background)
    left_rgb = torch.clamp(left_pkg["render"], 0.0, 1.0)
    right_rgb = torch.clamp(right_pkg["render"], 0.0, 1.0)
    return left_cam, right_cam, left_rgb, right_rgb


def _load_overlay_font(image_height):
    from PIL import ImageFont

    font_size = max(16, min(56, int(image_height) // 20))
    if font_size in _OVERLAY_FONT_CACHE:
        return _OVERLAY_FONT_CACHE[font_size]

    font = None
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except OSError:
            pass
    if font is None:
        font = ImageFont.load_default()

    _OVERLAY_FONT_CACHE[font_size] = font
    return font


def _draw_rgba_text_overlay(image_u8, text):
    if not text:
        return image_u8

    from PIL import ImageDraw

    height, width = image_u8.shape[:2]
    image = Image.fromarray(image_u8)
    draw = ImageDraw.Draw(image)
    font = _load_overlay_font(height)
    padding = max(4, height // 90)
    margin_x = max(8, width // 32)
    margin_y = max(8, height // 32)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except AttributeError:
        text_width, text_height = draw.textsize(text, font=font)
        bbox = (0, 0, text_width, text_height)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    rect = (
        margin_x,
        margin_y,
        margin_x + text_width + padding * 2,
        margin_y + text_height + padding * 2,
    )
    draw.rectangle(rect, fill=(0, 0, 0, 255))
    draw.text(
        (margin_x + padding - bbox[0], margin_y + padding - bbox[1]),
        text,
        font=font,
        fill=(255, 255, 255, 255),
    )
    return np.array(image, dtype=np.uint8)


def _tensor_to_rgba8_bytes(image, overlay_text=None):
    image = torch.clamp(image.detach(), 0.0, 1.0)
    if image.shape[0] == 3:
        alpha = torch.ones((1, image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)
        image = torch.cat([image, alpha], dim=0)
    elif image.shape[0] != 4:
        raise ValueError(f"Expected 3 or 4 image channels, got {image.shape[0]}.")

    image_u8 = (image.permute(1, 2, 0).contiguous() * 255.0).byte().cpu().numpy()
    image_u8 = _draw_rgba_text_overlay(image_u8, overlay_text)
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

    def overlay_text(self):
        if self.current_fps <= 0.0:
            return "FPS --.-"
        return f"FPS {self.current_fps:5.1f}"

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


def _sendall(conn, payload):
    conn.sendall(payload)


def _run_openxr_stream_session(
    gaussians,
    pipe,
    background,
    render_fn,
    xr_config_path,
    xr_socket_host,
    xr_socket_port,
    xr_max_frames,
):
    config = load_xr_session_config(xr_config_path)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((xr_socket_host, int(xr_socket_port)))
    listener.listen(1)
    print(f"[openxr-stream] waiting for client on {xr_socket_host}:{xr_socket_port}")
    try:
        conn, addr = listener.accept()
        print(f"[openxr-stream] connected by {addr}")
        rendered_count = 0
        fps_meter = _StreamFpsMeter()
        with conn:
            reader = conn.makefile("rb")
            while True:
                line = reader.readline()
                if not line:
                    break
                payload = json.loads(line.decode("utf-8"))
                if isinstance(payload, dict) and payload.get("type") == "eos":
                    break
                if xr_max_frames > 0 and rendered_count >= xr_max_frames:
                    break

                frame_id = int(payload.get("frame_id", rendered_count))
                _, _, left_rgb, right_rgb = _render_stereo_frame(
                    payload,
                    config,
                    gaussians,
                    pipe,
                    background,
                    render_fn,
                )
                overlay_text = fps_meter.overlay_text()
                left_bytes, left_width, left_height = _tensor_to_rgba8_bytes(left_rgb, overlay_text)
                right_bytes, right_width, right_height = _tensor_to_rgba8_bytes(right_rgb, overlay_text)
                if left_width != right_width or left_height != right_height:
                    raise ValueError("Left and right stream images must have matching dimensions.")

                header = (
                    f"HGSFRAME {frame_id} {left_width} {left_height} 4 "
                    f"{len(left_bytes)} {len(right_bytes)}\n"
                ).encode("ascii")
                _sendall(conn, header)
                _sendall(conn, left_bytes)
                _sendall(conn, right_bytes)
                rendered_count += 1
                now = fps_meter.mark_frame_sent()
                if fps_meter.should_log(now):
                    print(
                        f"[openxr-stream] fps={fps_meter.current_fps:.1f} "
                        f"avg={fps_meter.average_fps(now):.1f} sent={rendered_count} "
                        f"last_frame={frame_id} ({left_width}x{left_height})",
                        flush=True,
                    )
    finally:
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
):
    if xr_mode == "openxr_stream":
        return _run_openxr_stream_session(
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            render_fn=render_fn,
            xr_config_path=xr_config_path,
            xr_socket_host=xr_socket_host,
            xr_socket_port=xr_socket_port,
            xr_max_frames=xr_max_frames,
        )

    config = load_xr_session_config(xr_config_path)
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
