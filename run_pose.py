#!/usr/bin/env python3
"""Run MediaPipe pose estimation on a video and export landmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib-cache")
)

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


POSE_LANDMARKS = [landmark.name.lower() for landmark in mp.solutions.pose.PoseLandmark]
POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("pose_output"),
        help="Directory for CSV/JSON outputs",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write an annotated preview MP4 next to the landmark outputs",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent
        / "models"
        / "pose_landmarker_lite.task",
        help="MediaPipe Pose Landmarker .task model path",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="MediaPipe pose model complexity: 0 fastest, 2 most accurate",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for initial pose detection",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for pose tracking",
    )
    return parser.parse_args()


def normalized_landmarks_to_dict(landmarks) -> list[dict[str, float | int | str]]:
    if landmarks is None:
        return []

    rows = []
    for index, landmark in enumerate(landmarks):
        rows.append(
            {
                "index": index,
                "name": POSE_LANDMARKS[index],
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }
        )
    return rows


def write_csv(path: Path, frames: list[dict]) -> None:
    fieldnames = [
        "frame",
        "timestamp_ms",
        "landmark_index",
        "landmark_name",
        "x",
        "y",
        "z",
        "visibility",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            for landmark in frame["landmarks"]:
                writer.writerow(
                    {
                        "frame": frame["frame"],
                        "timestamp_ms": frame["timestamp_ms"],
                        "landmark_index": landmark["index"],
                        "landmark_name": landmark["name"],
                        "x": landmark["x"],
                        "y": landmark["y"],
                        "z": landmark["z"],
                        "visibility": landmark["visibility"],
                    }
                )


def draw_landmarks(frame, landmarks: list[dict[str, float | int | str]]) -> None:
    height, width = frame.shape[:2]
    points = {}
    for landmark in landmarks:
        x = int(float(landmark["x"]) * width)
        y = int(float(landmark["y"]) * height)
        visibility = float(landmark.get("visibility") or 0.0)
        if visibility < 0.25:
            continue
        points[int(landmark["index"])] = (x, y)
        cv2.circle(frame, (x, y), 4, (80, 220, 120), -1)

    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (60, 160, 255), 2)


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Download pose_landmarker_lite.task first."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem.replace(" ", "_")
    json_path = args.out_dir / f"{stem}_pose.json"
    csv_path = args.out_dir / f"{stem}_pose.csv"
    preview_path = args.out_dir / f"{stem}_pose_preview.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))

    frames: list[dict] = []

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

            landmarks = []
            if result.pose_landmarks:
                landmarks = normalized_landmarks_to_dict(result.pose_landmarks[0])
            frames.append(
                {
                    "frame": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "landmarks": landmarks,
                }
            )

            if writer is not None:
                annotated = frame.copy()
                draw_landmarks(annotated, landmarks)
                writer.write(annotated)

            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Processed {frame_index}/{total_frames or '?'} frames")

    cap.release()
    if writer is not None:
        writer.release()

    metadata = {
        "video": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames_reported": total_frames,
        "frames_processed": len(frames),
        "landmark_model": "mediapipe.tasks.vision.PoseLandmarker",
        "model_path": str(model_path),
    }
    with json_path.open("w") as handle:
        json.dump({"metadata": metadata, "frames": frames}, handle, indent=2)
    write_csv(csv_path, frames)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    if writer is not None:
        print(f"Wrote {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
