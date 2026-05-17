# Project Notes

## Goal

Build a demo that turns a short kung fu video into motion data and replays it in simulation. There is no physical robot. The current plan is:

1. Extract human pose from the video with MediaPipe.
2. Convert pose landmarks into joint angles and normalized motion data.
3. Replay the motion in Colab with MuJoCo.
4. Later, retarget the motion from mocap markers onto an articulated humanoid model.

## Local Repo

Local path:

```text
/Users/sreyagogineni/Desktop/sauce/kungfurobot
```

GitHub repo:

```text
https://github.com/gosreya/kungfurobot
git@github.com:gosreya/kungfurobot.git
```

The repo is public. SSH auth from this Mac to GitHub is working.

## Important Local Files

Raw/source video, ignored by git:

```text
Screen Recording 2026-05-17 at 6.20.28 PM.mov
```

Generated pose/motion outputs, ignored by git:

```text
pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose.json
pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose.csv
pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose_preview.mp4
pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose_angles.csv
pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose_angles.json
pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose_motion.json
pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose_motion.csv
pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose_motion_preview.mp4
```

The most important upload for Colab is:

```text
pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose_motion.json
```

## Source Files

`run_pose.py`

- Runs MediaPipe Pose Landmarker on the source video.
- Writes pose JSON/CSV.
- Can write annotated source-video preview.
- Needs the MediaPipe `.task` model at `models/pose_landmarker_lite.task`.
- On this Mac, MediaPipe needed to run outside Codex sandbox because it uses macOS graphics services.

`export_angles.py`

- Reads pose JSON.
- Computes per-frame angles:
  - elbows
  - knees
  - shoulders
  - hips
  - ankles
- Angle convention is `A-B-C`, measured at middle point `B`.

`record_motion.py`

- Reads pose JSON.
- Writes normalized "motion capture" data:
  - hip-centered landmark positions
  - per-frame joint angles
  - skeleton-only preview video

`render_mujoco_motion.py`

- Intended for Colab.
- Reads `*_pose_motion.json`.
- Renders colored MuJoCo mocap markers to an MP4.
- Current first version is marker replay, not yet a true articulated humanoid retarget.

`colab_mujoco_replay.ipynb`

- Notebook added to the repo.
- Installs Colab requirements.
- Uploads `*_pose_motion.json`.
- Runs `render_mujoco_motion.py`.
- Displays/downloads `mujoco_motion_replay.mp4`.

## Setup Commands

Local MediaPipe venv already exists at:

```text
/Users/sreyagogineni/Desktop/sauce/.venv-mediapipe
```

Pose extraction command:

```bash
cd /Users/sreyagogineni/Desktop/sauce
.venv-mediapipe/bin/python kungfurobot/run_pose.py \
  "kungfurobot/Screen Recording 2026-05-17 at 6.20.28 PM.mov" \
  --out-dir kungfurobot/pose_output \
  --preview
```

Angle export:

```bash
cd /Users/sreyagogineni/Desktop/sauce/kungfurobot
python3 export_angles.py pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose.json
```

Motion recording:

```bash
cd /Users/sreyagogineni/Desktop/sauce/kungfurobot
/Users/sreyagogineni/Desktop/sauce/.venv-mediapipe/bin/python record_motion.py \
  pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose.json
```

## Colab Workflow

In Colab:

```python
!git clone https://github.com/gosreya/kungfurobot.git
%cd kungfurobot
!pip install -r requirements-colab.txt
```

Then upload local file:

```text
/Users/sreyagogineni/Desktop/sauce/kungfurobot/pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose_motion.json
```

Render:

```python
!python render_mujoco_motion.py "Screen_Recording_2026-05-17_at_6.20.28_PM_pose_motion.json" \
  --output mujoco_motion_replay.mp4
```

Display:

```python
from IPython.display import Video
Video("mujoco_motion_replay.mp4", embed=True)
```

## Colab MCP

The user asked about using the official Google Colab MCP server.

Reference:

```text
https://developers.googleblog.com/announcing-the-colab-mcp-server-connect-any-ai-agent-to-google-colab/
```

Installed locally:

```bash
python3 -m pip install --user uv
```

Added to Codex:

```bash
codex mcp add colab-proxy-mcp -- \
  /Users/sreyagogineni/Library/Python/3.11/bin/uvx \
  git+https://github.com/googlecolab/colab-mcp
```

`codex mcp list` shows:

```text
colab-proxy-mcp enabled
```

The current Codex session did not see the new MCP tools immediately, so the user is planning to restart Codex. After restart, check MCP resources/tools again.

## Current Git State

The notebook `colab_mujoco_replay.ipynb` and README update were committed locally:

```text
Add Colab MuJoCo replay notebook
```

The push of that notebook commit was interrupted by the user while discussing MCP. After restart, check:

```bash
cd /Users/sreyagogineni/Desktop/sauce/kungfurobot
git status
git log --oneline --decorate -5
git push
```

The previous pushed commit was:

```text
Add MuJoCo motion replay
```

## Next Steps

1. Restart Codex so the Colab MCP server is loaded.
2. Verify MCP availability.
3. Push the latest local notebook commit if it is not on GitHub yet.
4. In Colab, pull latest repo or reopen the notebook from GitHub.
5. Upload `*_pose_motion.json`.
6. Run MuJoCo render.
7. If marker replay works, improve demo by adding an articulated humanoid retarget.
