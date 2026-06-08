import argparse
import ast
import json
import re
from collections import defaultdict


LINE_RE = re.compile(r"^\[(?P<tag>[^\]]+)\]\s+(?P<body>.*?)(?:\s+\[\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\])?\s*$")
KEY_RE = re.compile(r"(?<!\S)(?P<key>[A-Za-z_][A-Za-z0-9_]*)=")


def _parse_value(raw):
    raw = raw.strip()
    if not raw:
        return raw
    if raw[0] in "[{(":
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw
    try:
        if any(c in raw for c in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_body(body):
    matches = list(KEY_RE.finditer(body))
    parsed = {}
    for idx, match in enumerate(matches):
        key = match.group("key")
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        parsed[key] = _parse_value(body[value_start:value_end].strip())
    return parsed


def parse_log(path):
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            match = LINE_RE.match(line.strip())
            if not match:
                continue
            records.append(
                {
                    "line": line_no,
                    "tag": match.group("tag"),
                    **_parse_body(match.group("body")),
                }
            )
    return records


def _numeric_values(records, key):
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _summary(values):
    if not values:
        return None
    values = sorted(values)
    n = len(values)

    def percentile(p):
        if n == 1:
            return values[0]
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return values[lo] * (1.0 - frac) + values[hi] * frac

    return {
        "count": n,
        "min": round(values[0], 3),
        "mean": round(sum(values) / n, 3),
        "p50": round(percentile(0.50), 3),
        "p95": round(percentile(0.95), 3),
        "max": round(values[-1], 3),
    }


def _top(records, key, limit):
    sortable = [record for record in records if isinstance(record.get(key), (int, float))]
    sortable.sort(key=lambda item: float(item[key]), reverse=True)
    keep = []
    for record in sortable[:limit]:
        keep.append(
            {
                "line": record.get("line"),
                "frame_id": record.get("frame_id"),
                "eye": record.get("eye"),
                key: record.get(key),
                "prefilter_selected": record.get("prefilter_selected"),
                "estimated_gaussians": record.get("estimated_gaussians"),
                "preflight_ms": record.get("preflight_ms"),
                "total_ms": record.get("total_ms"),
            }
        )
    return keep


def build_report(records, top_n=8):
    by_tag = defaultdict(list)
    for record in records:
        by_tag[record["tag"]].append(record)

    preflight = by_tag.get("xr-preflight", [])
    slow = by_tag.get("render-slow", [])
    guard = by_tag.get("xr-guard", [])
    stream = by_tag.get("openxr-stream", [])

    by_eye = defaultdict(list)
    for record in preflight:
        by_eye[str(record.get("eye", ""))].append(record)

    preflight_summary = {}
    for eye, eye_records in sorted(by_eye.items()):
        preflight_summary[eye] = {
            key: _summary(_numeric_values(eye_records, key))
            for key in [
                "lod_selected",
                "prefilter_selected",
                "estimated_gaussians",
                "anchor_raster_visible",
                "anchor_radii_p95",
                "anchor_radii_max",
                "anchor_radii2_sum",
                "preflight_ms",
            ]
        }

    stage_totals = defaultdict(list)
    for record in slow:
        stages = record.get("stages_ms")
        if isinstance(stages, dict):
            for key, value in stages.items():
                if isinstance(value, (int, float)):
                    stage_totals[key].append(float(value))

    guard_thresholds = defaultdict(int)
    for record in guard:
        reason = str(record.get("reason", ""))
        for part in reason.split(","):
            metric = part.split(">", 1)[0]
            if "." in metric:
                metric = metric.split(".", 1)[1]
            if "=" in metric:
                metric = metric.split("=", 1)[0]
            if metric:
                guard_thresholds[metric] += 1

    return {
        "records": len(records),
        "counts": {
            "xr_preflight": len(preflight),
            "render_slow": len(slow),
            "xr_guard": len(guard),
            "openxr_stream": len(stream),
        },
        "frames": {
            "first": min((int(r["frame_id"]) for r in preflight if isinstance(r.get("frame_id"), int)), default=None),
            "last": max((int(r["frame_id"]) for r in preflight if isinstance(r.get("frame_id"), int)), default=None),
            "unique_preflight": len({r.get("frame_id") for r in preflight if r.get("frame_id") is not None}),
        },
        "preflight_by_eye": preflight_summary,
        "slow_render": {
            "total_ms": _summary(_numeric_values(slow, "total_ms")),
            "generated_gaussians": _summary(_numeric_values(slow, "generated_gaussians")),
            "prefilter_selected": _summary(_numeric_values(slow, "prefilter_selected")),
            "stage_ms": {key: _summary(values) for key, values in sorted(stage_totals.items())},
            "top_total_ms": _top(slow, "total_ms", top_n),
        },
        "guard": {
            "count": len(guard),
            "first_frame": guard[0].get("frame_id") if guard else None,
            "last_frame": guard[-1].get("frame_id") if guard else None,
            "threshold_hits": dict(sorted(guard_thresholds.items())),
        },
        "top_prefilter_selected": _top(preflight, "prefilter_selected", top_n),
        "top_preflight_ms": _top(preflight, "preflight_ms", top_n),
    }


def _format_summary_block(title, stats):
    print(title)
    if stats is None:
        print("  no numeric samples")
        return
    print(
        "  count={count} min={min} mean={mean} p50={p50} p95={p95} max={max}".format(
            **stats
        )
    )


def print_report(report):
    print("XR log summary")
    print(f"  records={report['records']} counts={report['counts']}")
    print(f"  frames={report['frames']}")
    print()

    for eye, stats_by_key in report["preflight_by_eye"].items():
        print(f"preflight eye={eye}")
        for key, stats in stats_by_key.items():
            _format_summary_block(f"  {key}", stats)
        print()

    slow = report["slow_render"]
    _format_summary_block("slow render total_ms", slow["total_ms"])
    _format_summary_block("slow render prefilter_selected", slow["prefilter_selected"])
    _format_summary_block("slow render generated_gaussians", slow["generated_gaussians"])
    print("slow render stage_ms")
    for key, stats in slow["stage_ms"].items():
        _format_summary_block(f"  {key}", stats)
    print()

    guard = report["guard"]
    print(
        f"guard count={guard['count']} first_frame={guard['first_frame']} "
        f"last_frame={guard['last_frame']} threshold_hits={guard['threshold_hits']}"
    )
    print()

    print("top prefilter_selected")
    for item in report["top_prefilter_selected"]:
        print(f"  {item}")
    print()
    print("top preflight_ms")
    for item in report["top_preflight_ms"]:
        print(f"  {item}")
    print()
    print("top slow total_ms")
    for item in slow["top_total_ms"]:
        print(f"  {item}")


def main():
    parser = argparse.ArgumentParser(description="Summarize HorizonGS XR profiling log lines.")
    parser.add_argument("log_path")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(parse_log(args.log_path), top_n=args.top)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
