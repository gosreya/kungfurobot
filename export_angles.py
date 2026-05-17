#!/usr/bin/env python3
"""Compute joint angles from exported MediaPipe pose landmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ANGLE_DEFINITIONS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_ankle": ("left_knee", "left_ankle", "left_foot_index"),
    "right_ankle": ("right_knee", "right_ankle", "right_foot_index"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose_json", type=Path, help="Pose JSON from run_pose.py")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the pose JSON directory.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.4,
        help="Require each landmark in an angle to have at least this visibility.",
    )
    return parser.parse_args()


def angle_degrees(a: dict, b: dict, c: dict) -> float:
    ba = (float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]), float(a["z"]) - float(b["z"]))
    bc = (float(c["x"]) - float(b["x"]), float(c["y"]) - float(b["y"]), float(c["z"]) - float(b["z"]))

    dot = sum(ba_i * bc_i for ba_i, bc_i in zip(ba, bc))
    ba_mag = math.sqrt(sum(component * component for component in ba))
    bc_mag = math.sqrt(sum(component * component for component in bc))
    if ba_mag == 0 or bc_mag == 0:
        return math.nan

    cosine = max(-1.0, min(1.0, dot / (ba_mag * bc_mag)))
    return math.degrees(math.acos(cosine))


def landmark_map(frame: dict) -> dict[str, dict]:
    return {landmark["name"]: landmark for landmark in frame.get("landmarks", [])}


def visible_enough(points: tuple[dict, dict, dict], min_visibility: float) -> bool:
    return all(float(point.get("visibility") or 0.0) >= min_visibility for point in points)


def compute_frame_angles(frame: dict, min_visibility: float) -> dict[str, float | int | None]:
    landmarks = landmark_map(frame)
    row: dict[str, float | int | None] = {
        "frame": frame["frame"],
        "timestamp_ms": frame["timestamp_ms"],
    }

    for angle_name, names in ANGLE_DEFINITIONS.items():
        try:
            points = tuple(landmarks[name] for name in names)
        except KeyError:
            row[angle_name] = None
            continue

        if not visible_enough(points, min_visibility):
            row[angle_name] = None
            continue

        angle = angle_degrees(*points)
        row[angle_name] = None if math.isnan(angle) else round(angle, 3)

    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["frame", "timestamp_ms", *ANGLE_DEFINITIONS.keys()]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    pose_path = args.pose_json.expanduser().resolve()
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose JSON not found: {pose_path}")

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else pose_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{pose_path.stem}_angles.csv"
    json_path = out_dir / f"{pose_path.stem}_angles.json"

    with pose_path.open() as handle:
        pose_data = json.load(handle)

    rows = [
        compute_frame_angles(frame, args.min_visibility)
        for frame in pose_data.get("frames", [])
    ]
    payload = {
        "source_pose_json": str(pose_path),
        "angle_definitions": ANGLE_DEFINITIONS,
        "min_visibility": args.min_visibility,
        "frames": rows,
    }

    write_csv(csv_path, rows)
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
