# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyFlyt Drone — RL training and control for fixed-wing UAVs. Backbone: PyFlyt (PyBullet) + Stable-Baselines3 PPO. Visual sim2real tasks use Ultralytics YOLO/YOLOE + supervision for object detection and segmentation (see `envs/detector/yolo.py`). Companion agent-rules live in `AGENTS.md` (build, style, commits, security) — this file focuses on architecture and run commands.

## Setup

No `pyproject.toml` / lock file.

### macOS / Apple Silicon (recommended path)

`pybullet` ships **no** Apple Silicon wheel and its source build fails against modern Xcode SDKs (gzlib K&R syntax + `NULL` macro conflict with vendored zlib). Use miniforge + conda-forge for the C/C++ stack, then pip-install the pure-Python packages:

```bash
brew install --cask miniforge
mamba create -n pyflyt python=3.11 -y
mamba activate pyflyt
mamba install -y -c conda-forge pybullet numpy pyyaml matplotlib pyopengl gymnasium
pip install PyFlyt stable-baselines3 ultralytics supervision
# torch installed via pip for MPS (Apple GPU) support; mamba's torch lacks MPS
pip install torch torchvision
```

Verify the simulator chain locally:

```bash
python -c "import pybullet as p; p.connect(p.GUI); print('PyBullet GUI OK')"
python -c "import torch; print('torch', torch.__version__, 'mps=', torch.backends.mps.is_available())"
```

### Linux / WSL

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install PyFlyt stable-baselines3 gymnasium pybullet torch numpy pyyaml ultralytics supervision matplotlib
```

For WSLg users, GUI windows are forwarded automatically; pure WSL (no WSLg) needs an X server. If `matplotlib` plots fail to show, set `MPLBACKEND=Agg`.

### Run-from-SSH gotcha

If you SSH into the Mac from another machine, PyBullet's GUI window won't appear on your local display — it opens on the Mac's screen (or fails if the SSH session has no GUI access). For remote work, either:
- run training/eval headless (`render_mode=None` or `'rgb_array'`) and save frames, or
- VNC / screen-share into the Mac, or
- use a Mac as the training host and pull checkpoints over.

The repo was originally driven from a Windows box via SSH — that's why no Mac-specific path setup existed. Now it's portable: `train/train_objlock.py` and `train/train_vtail_objlock.py` resolve `drone_model_dir` relative to the script, so no hardcoded host paths.

## Smoke Check

```bash
python -m compileall envs train eval scripts utils   # syntax + import-path check; not a test suite
```

There is no automated test suite. When adding tests, put them under `tests/` with `test_*.py` names; use deterministic seeds and short episodes. Behavior changes should also be smoke-tested via a fast training/eval run with reduced timesteps and `--no-render`.

## Environment Hierarchy

```
FixedwingBaseEnv          envs/fixedwing_envs/base/fixedwing_base_env.py
└── FixedwingVtailBaseEnv envs/fixedwing_envs/base/fixedwing_vtail_base_env.py
    ├── FixedwingObjLockEnv        envs/fixedwing_envs/fixedwing_objlock_env.py        (V-tail, visual duck strike)
    │   └── FlattenObjLockEnv      envs/fixedwing_envs/flatten_objlock_env.py         (Dict obs → Box)
    ├── FixedwingWaypointObjLockEnv envs/fixedwing_envs/fixedwing_waypoint_objlock_env.py
    │   └── FlattenWaypointEnv     envs/fixedwing_envs/flatten_waypoint_env.py
    ├── FixedwingLowLevelEnv       envs/fixedwing_envs/fixedwing_lowlevel_env.py
    └── ObjLockYoloEnv             envs/fixedwing_envs/objlock_yolo_env.py            (real YOLO in the loop)

AbFixedwingEnv             envs/fixedwing_envs/ab_fixedwing_env.py
└── PyFlyt/Fixedwing-Waypoints-v3 via gym.make, FlattenWaypointEnv (PyFlyt)
```

Two airframes: standard (`my_models/fixedwing/`) and V-tail (`my_models/fixedwing_vtail/`). The V-tail is what `FixedwingObjLockEnv` actually inherits from.

## Environment Variants

| Task | Train | Eval |
| --- | --- | --- |
| A→B waypoints (base) | `train/train_ppo_ab.py` | `eval/eval_waypoints.py` |
| ObjLock — visual duck strike (no waypoints) | `train/train_objlock.py` | `eval/eval_objlock.py` |
| ObjLock with V-tail airframe | `train/train_vtail_objlock.py` | `eval/eval_vtail_objlock.py` |
| ObjLock + real YOLO detector in the loop | (built on `ObjLockYoloEnv`) | `eval/eval_objlock_yolo.py` |
| Waypoints + ObjLock combined | `train/train_Fixedwing_Waypoints_ObjLock.py` | `eval/eval_waypoint_objlock.py` |
| Official PyFlyt Waypoints v3 | `train/train_Fixedwing_Waypoints_v3.py` | — |
| Low-level cmd (roll/pitch/yaw/thrust) | `train/train_lowlevel_cmd.py` | `eval/eval_lowlevel.py` |
| High-level cmd (heading/speed/alt) | `train/train_highlevel_cmd.py` | — |

`eval/eval_objlock_yolo.py` is the sim2real entry point: it pulls the detector out of `envs/detector/yolo.py` (`YoloeDetector` for visual-prompted seg, `Yolo11Detector` for text/class seg) and feeds detections back into the obs.

## Common Commands

### Training (defaults; most scripts honor CLI overrides like `--total_timesteps`)

```bash
python train/train_ppo_ab.py
python train/train_objlock.py
python train/train_vtail_objlock.py
python train/train_Fixedwing_Waypoints_ObjLock.py
python train/train_Fixedwing_Waypoints_v3.py
python train/train_lowlevel_cmd.py
python train/train_highlevel_cmd.py
```

Each train script writes to `models/<run>/` (`best_model.zip`, `vecnorm.pkl`) and `logs/<run>/`. Both directories are git-ignored. Keep `best_model.zip` and `vecnorm.pkl` together — `VecNormalize` stats are required to load the policy.

### Evaluation

```bash
python eval/eval_waypoints.py --model models/<run>/best_model.zip --episodes 10
python eval/eval_objlock.py   --model models/<run>/best_model.zip --episodes 10 --no-render
python eval/eval_vtail_objlock.py   --model models/<run>/best_model.zip --episodes 5
python eval/eval_waypoint_objlock.py --model models/<run>/best_model.zip --episodes 5
python eval/eval_objlock_yolo.py    --model models/<run>/best_model.zip --episodes 3
python eval/eval_lowlevel.py        --model models/<run>/best_model.zip --episodes 5
```

### Configuration

- `configs/env.yaml` — env params: `render_mode`, `flight_dome_size`, `max_duration_seconds`, `agent_hz`, `angle_representation` (`quaternion`|`euler`), `wind`.
- `configs/ppo.yaml` — PPO hyperparams: `num_envs`, `n_steps`, `batch_size`, `learning_rate`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `seed`, `normalize_obs`, `normalize_reward`.

## Key Implementation Details

### Action & obs spaces
4D continuous action `[roll, pitch, yaw, thrust]` ∈ `[-1, 1]`. Obs is a `gymnasium.spaces.Dict`; each task ships a `Flatten*Env` wrapper that converts to a single `Box` for the SB3 MLP policy.

### VecNormalize
All training uses `VecNormalize` (obs + reward). Eval must load the matching `vecnorm.pkl` (most eval scripts default to `models/vecnorm.pkl` next to the model). Use `training=False` and `norm_reward=False` for eval.

### Wind field
`envs/utils.py::WindOnResetWrapper` registers a wind callback on the underlying PyFlyt Aviary on every reset. Modes: `constant` or `gust_sine`. Set `randomize_on_reset=true` and supply `*_range` lists to randomize per-axis. Wind is in ENU (m/s). Add the wrapper around any env that exposes a PyFlyt aviary via `env.unwrapped.env`.

### Camera profiles
ObjLock envs accept a `camera_profile` arg: `"default"`, `"cockpit_fpv"` (forward-looking, what the agent learns to fly toward the duck with), `"chase"` (third-person follow). FPV is the canonical sim2real setup.

### YOLO / sim2real
`envs/detector/yolo.py` holds two detector classes:
- `YoloeDetector` — visual-prompted segmentation (reference image + bbox prompts → `YOLOEVPSegPredictor`).
- `Yolo11Detector` — plain YOLO11 inference / train.

`scripts/yoloe_offline_seg.py` runs offline segmentation on a folder of images for dataset curation; `scripts/modelscope_download.py` fetches model weights. `ObjLockYoloEnv` puts a real detector inside the env loop so the agent trains on the same noisy detections it will see at inference.

### Debug overlay
`envs/utils.py::PyBulletDebugOverlay` prints alt / airspeed / thrust / target distance to the terminal and draws a HUD in the PyBullet viewer. Construct one in your eval driver if you want telemetry.

### Callbacks
`utils/callbacks.py` has `InfoLoggerCallback` (stub for custom SB3 logging — extend `_on_step` to pull env stats). Eval/checkpoint callbacks are wired up directly in each train script.

### Style / conventions (per `AGENTS.md`)
Python 3.10+, 4-space indent, PEP 8. Imports grouped stdlib → third-party → local. Classes `PascalCase` (`FixedwingObjLockEnv`), functions/files `snake_case`, config dicts `UPPER_CASE` (`TRAIN_CONFIG`). Keep repo-relative paths in configs — never commit absolute paths or local env state.

### Commits
Short imperative subject, scope prefix preferred (`feat:`, `fix:`, `refactor:`). Many existing commits use Chinese descriptions — that's fine. Do not commit `models/`, `logs/`, downloaded weights, or rendered PNG/MP4 outputs (already in `.gitignore`, extend if new artifact types appear).