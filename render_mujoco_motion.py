#!/usr/bin/env python3
"""Render a normalized pose-motion JSON as MuJoCo mocap markers."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco


DEFAULT_LANDMARKS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_json", type=Path, help="Motion JSON from record_motion.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mujoco_motion_replay.mp4"),
        help="Output MP4 path",
    )
    parser.add_argument("--width", type=int, default=960, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    parser.add_argument(
        "--world-scale",
        type=float,
        default=0.9,
        help="Scale normalized pose coordinates into MuJoCo world units",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.35,
        help="Hide landmarks below this visibility by moving them below the floor",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


def landmark_to_world(landmark: dict, world_scale: float) -> tuple[float, float, float]:
    x = float(landmark["x"]) * world_scale
    y = -float(landmark["z"]) * world_scale * 0.35
    z = 1.25 - float(landmark["y"]) * world_scale
    return x, y, z


def rgba_for_landmark(name: str) -> str:
    if name.startswith("left_"):
        return "0.15 0.45 0.95 1"
    if name.startswith("right_"):
        return "0.95 0.35 0.15 1"
    return "0.2 0.2 0.2 1"


def build_mjcf(landmark_names: list[str]) -> str:
    mujoco_node = ET.Element("mujoco", model="kungfu_mocap")
    ET.SubElement(mujoco_node, "compiler", angle="degree")
    ET.SubElement(mujoco_node, "option", timestep="0.0166667")

    asset = ET.SubElement(mujoco_node, "asset")
    ET.SubElement(asset, "material", name="floor_mat", rgba="0.9 0.9 0.86 1")

    worldbody = ET.SubElement(mujoco_node, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="4 4 0.02",
        material="floor_mat",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="key",
        pos="0 -3 5",
        dir="0 1 -1",
        diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="replay",
        pos="0 -4.3 1.7",
        xyaxes="1 0 0 0 0.25 1",
        fovy="42",
    )

    for name in landmark_names:
        body = ET.SubElement(
            worldbody,
            "body",
            name=f"mocap_{safe_name(name)}",
            mocap="true",
            pos="0 0 -10",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"marker_{safe_name(name)}",
            type="sphere",
            size="0.035",
            rgba=rgba_for_landmark(name),
        )

    return ET.tostring(mujoco_node, encoding="unicode")


def select_landmarks(frames: list[dict]) -> list[str]:
    available = set()
    for frame in frames:
        available.update(frame.get("landmarks", {}).keys())
    selected = [name for name in DEFAULT_LANDMARKS if name in available]
    return selected or sorted(available)


def set_mocap_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    landmark_name: str,
    pos: tuple[float, float, float],
) -> None:
    body_id = model.body(f"mocap_{safe_name(landmark_name)}").id
    mocap_id = model.body_mocapid[body_id]
    data.mocap_pos[mocap_id] = pos


def render_motion(
    motion_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    world_scale: float,
    min_visibility: float,
) -> None:
    with motion_path.open() as handle:
        motion = json.load(handle)

    frames = motion.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {motion_path}")

    landmark_names = select_landmarks(frames)
    mjcf = build_mjcf(landmark_names)

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
        handle.write(mjcf)
        model_path = handle.name

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hidden_pos = (0.0, 0.0, -10.0)
    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            landmarks = frame.get("landmarks", {})
            for name in landmark_names:
                landmark = landmarks.get(name)
                if not landmark or float(landmark.get("visibility") or 0.0) < min_visibility:
                    set_mocap_position(model, data, name, hidden_pos)
                    continue
                set_mocap_position(model, data, name, landmark_to_world(landmark, world_scale))

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="replay")
            writer.append_data(renderer.render())

    renderer.close()
    Path(model_path).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    render_motion(
        motion_path=args.motion_json.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        width=args.width,
        height=args.height,
        fps=args.fps,
        world_scale=args.world_scale,
        min_visibility=args.min_visibility,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
