#!/usr/bin/env python3
"""Create a normalized motion recording from MediaPipe pose landmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2

from export_angles import ANGLE_DEFINITIONS, compute_frame_angles


POSE_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
)


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
        help="Visibility threshold used for angle calculation and preview drawing.",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=900,
        help="Width and height of the square skeleton preview video.",
    )
    return parser.parse_args()


def landmarks_by_name(frame: dict) -> dict[str, dict]:
    return {landmark["name"]: landmark for landmark in frame.get("landmarks", [])}


def midpoint(a: dict, b: dict) -> dict[str, float]:
    return {
        "x": (float(a["x"]) + float(b["x"])) / 2.0,
        "y": (float(a["y"]) + float(b["y"])) / 2.0,
        "z": (float(a["z"]) + float(b["z"])) / 2.0,
    }


def distance_2d(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def scale_for_pose(landmarks: dict[str, dict]) -> float:
    candidates = []
    if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
        candidates.append(distance_2d(landmarks["left_shoulder"], landmarks["right_shoulder"]))
    if "left_hip" in landmarks and "right_hip" in landmarks:
        candidates.append(distance_2d(landmarks["left_hip"], landmarks["right_hip"]))
    candidates = [value for value in candidates if value > 0]
    return max(candidates) if candidates else 1.0


def normalize_landmarks(frame: dict) -> dict[str, dict[str, float]]:
    landmarks = landmarks_by_name(frame)
    if "left_hip" in landmarks and "right_hip" in landmarks:
        origin = midpoint(landmarks["left_hip"], landmarks["right_hip"])
    elif "left_shoulder" in landmarks and "right_shoulder" in landmarks:
        origin = midpoint(landmarks["left_shoulder"], landmarks["right_shoulder"])
    else:
        origin = {"x": 0.5, "y": 0.5, "z": 0.0}

    scale = scale_for_pose(landmarks)
    normalized = {}
    for name, landmark in landmarks.items():
        normalized[name] = {
            "x": round((float(landmark["x"]) - origin["x"]) / scale, 6),
            "y": round((float(landmark["y"]) - origin["y"]) / scale, 6),
            "z": round((float(landmark["z"]) - origin["z"]) / scale, 6),
            "visibility": round(float(landmark.get("visibility") or 0.0), 6),
        }
    return normalized


def make_motion_frame(frame: dict, min_visibility: float) -> dict:
    angle_row = compute_frame_angles(frame, min_visibility)
    angle_row.pop("frame", None)
    angle_row.pop("timestamp_ms", None)
    return {
        "frame": frame["frame"],
        "timestamp_ms": frame["timestamp_ms"],
        "root": "mid_hip",
        "scale": "shoulder_width_or_hip_width",
        "landmarks": normalize_landmarks(frame),
        "angles_degrees": angle_row,
    }


def write_motion_csv(path: Path, frames: list[dict]) -> None:
    landmark_names = sorted({name for frame in frames for name in frame["landmarks"]})
    fields = ["frame", "timestamp_ms"]
    fields.extend(ANGLE_DEFINITIONS.keys())
    for name in landmark_names:
        fields.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            row = {
                "frame": frame["frame"],
                "timestamp_ms": frame["timestamp_ms"],
                **frame["angles_degrees"],
            }
            for name in landmark_names:
                landmark = frame["landmarks"].get(name)
                if not landmark:
                    continue
                row[f"{name}_x"] = landmark["x"]
                row[f"{name}_y"] = landmark["y"]
                row[f"{name}_z"] = landmark["z"]
                row[f"{name}_visibility"] = landmark["visibility"]
            writer.writerow(row)


def draw_preview_frame(frame: dict, size: int, min_visibility: float):
    canvas = 255 * cv2.UMat(size, size, cv2.CV_8UC3).get()
    center = size // 2
    scale = size * 0.2
    points = {}

    for name, landmark in frame["landmarks"].items():
        if float(landmark["visibility"]) < min_visibility:
            continue
        x = int(center + float(landmark["x"]) * scale)
        y = int(center + float(landmark["y"]) * scale)
        points[name] = (x, y)

    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(canvas, points[start], points[end], (35, 95, 190), 4)

    for point in points.values():
        cv2.circle(canvas, point, 5, (40, 155, 75), -1)

    label = f"frame {frame['frame']}  {frame['timestamp_ms']} ms"
    cv2.putText(canvas, label, (24, size - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    return canvas


def write_preview(path: Path, frames: list[dict], fps: float, size: int, min_visibility: float) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps or 30.0,
        (size, size),
    )
    for frame in frames:
        writer.write(draw_preview_frame(frame, size, min_visibility))
    writer.release()


def main() -> int:
    args = parse_args()
    pose_path = args.pose_json.expanduser().resolve()
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose JSON not found: {pose_path}")

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else pose_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with pose_path.open() as handle:
        pose_data = json.load(handle)

    fps = float(pose_data.get("metadata", {}).get("fps") or 30.0)
    motion_frames = [
        make_motion_frame(frame, args.min_visibility)
        for frame in pose_data.get("frames", [])
    ]

    stem = pose_path.stem
    json_path = out_dir / f"{stem}_motion.json"
    csv_path = out_dir / f"{stem}_motion.csv"
    preview_path = out_dir / f"{stem}_motion_preview.mp4"

    payload = {
        "source_pose_json": str(pose_path),
        "coordinate_system": {
            "origin": "midpoint between left_hip and right_hip",
            "scale": "shoulder width when available, otherwise hip width",
            "x": "positive right in image space",
            "y": "positive down in image space",
            "z": "MediaPipe relative depth, normalized by same scale",
        },
        "angle_definitions": ANGLE_DEFINITIONS,
        "frames": motion_frames,
    }

    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    write_motion_csv(csv_path, motion_frames)
    write_preview(preview_path, motion_frames, fps, args.preview_size, args.min_visibility)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
