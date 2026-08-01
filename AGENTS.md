# Repository Guidelines

Project context & docs live in ./ctx — managed by the ctx skills; apply their consistency rules when editing ctx docs.

## Project Structure & Module Organization

This repository contains Python code for PyFlyt fixed-wing drone simulation and reinforcement-learning training. Core environments live in `envs/`, with reusable base classes under `envs/fixedwing_envs/` and detector integration under `envs/detector/`. Training entry points are in `train/`; evaluation scripts are in `eval/`. Runtime configuration is kept in `configs/env.yaml` and `configs/ppo.yaml`. Utility code is in `utils/` and `scripts/`. Simulation assets, including URDF and YAML aircraft definitions, are in `my_models/`. Small runnable examples are in `examples/`.

## Build, Test, and Development Commands

There is no package build step or dependency lock file. Create a virtual environment and install the project dependencies used by the scripts, including `PyFlyt`, `stable-baselines3`, `gymnasium`, `pybullet`, `torch`, `numpy`, and `pyyaml`.

```bash
python -m compileall envs train eval scripts utils
python train/train_ppo_ab.py
python train/train_objlock.py --total_timesteps 10000
python eval/eval_objlock.py --model models/<run>/best_model.zip --episodes 10 --no-render
```

Use `compileall` as a quick syntax/import-path smoke check. Training scripts write generated artifacts to `models/` and `logs/`, which are intentionally ignored by Git.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax, 4-space indentation, and PEP 8-compatible formatting. Keep imports grouped as standard library, third-party packages, then local modules. Environment classes should use `PascalCase` names such as `FixedwingObjLockEnv`; functions, methods, and files should use `snake_case`. Keep task constants and default configs clearly named, usually in `UPPER_CASE` dictionaries such as `TRAIN_CONFIG`.

## Testing Guidelines

No automated test suite is currently present. When adding tests, place them under `tests/` using `test_*.py` names and prefer deterministic seeds plus small episode counts. For behavior changes, run a fast smoke check with `compileall`, then run the relevant training or evaluation script with reduced timesteps or `--no-render`. Models that use `VecNormalize` must keep `best_model.zip` and `vecnorm.pkl` together for evaluation.

## Commit & Pull Request Guidelines

Recent history uses short, descriptive commits, often with `feat:` for feature work and Chinese descriptions. Keep messages imperative and scoped, for example `feat: add wind randomization for objlock`. Pull requests should describe the affected environment or script, list commands run, summarize training/evaluation results when relevant, and link issues. Do not commit generated model checkpoints, logs, downloaded assets, or large render outputs.

## Security & Configuration Tips

Prefer repository-relative paths for assets and configs. Avoid committing local absolute paths, credentials, private datasets, or machine-specific settings. Update `.gitignore` if new generated artifacts are introduced.
