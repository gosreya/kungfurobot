# Kung Fu Robot

Pose-estimation and motion-recording pipeline for a short kung fu clip.

## Current Pipeline

1. `run_pose.py` runs MediaPipe Pose Landmarker on a video.
2. `export_angles.py` computes joint angles from the pose landmarks.
3. `record_motion.py` creates a normalized motion recording with:
   - hip-centered landmark positions
   - per-frame joint angles
   - skeleton-only preview video
4. `render_mujoco_motion.py` renders the normalized motion as MuJoCo mocap markers.

## Local Setup

```bash
python3 -m venv .venv-mediapipe
.venv-mediapipe/bin/python -m pip install -r requirements.txt
mkdir -p models
curl -L -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Run Pose Estimation

```bash
.venv-mediapipe/bin/python run_pose.py "Screen Recording 2026-05-17 at 6.20.28 PM.mov" \
  --out-dir pose_output \
  --preview
```

## Export Angles

```bash
python3 export_angles.py pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose.json
```

## Record Normalized Motion

```bash
.venv-mediapipe/bin/python record_motion.py pose_output/Screen_Recording_2026-05-17_at_6.20.28_PM_pose.json
```

## GitHub + Colab Workflow

Keep source files in GitHub. Keep raw videos and generated outputs local or upload them manually to Colab when needed.

In Colab:

```python
!git clone https://github.com/YOUR_USER/kungfurobot.git
%cd kungfurobot
!pip install -r requirements-colab.txt
```

Then upload `*_pose_motion.json` or the source video into the notebook session.

## Colab MuJoCo Replay

You can open `colab_mujoco_replay.ipynb` directly in Colab after cloning/opening the GitHub repo.

After cloning the repo in Colab:

```python
%cd kungfurobot
!pip install -r requirements-colab.txt
```

Upload the local `*_pose_motion.json` file:

```python
from google.colab import files
uploaded = files.upload()
motion_json = next(name for name in uploaded if name.endswith("_pose_motion.json"))
motion_json
```

Render the MuJoCo replay:

```python
!python render_mujoco_motion.py "$motion_json" --output mujoco_motion_replay.mp4
```

Show the rendered video:

```python
from IPython.display import Video
Video("mujoco_motion_replay.mp4", embed=True)
```
