# ReKep on SO-101

Implementation of [ReKep (arXiv 2409.01652)](https://arxiv.org/abs/2409.01652) — Relational Keypoint Constraints for robot manipulation — on a real **SO-101** 5-DOF robot arm with an Intel RealSense D435 RGB-D camera, running entirely on an **M1 MacBook (8GB RAM)**.

## Overview

ReKep represents manipulation tasks as Python constraint functions over semantic keypoints in the scene. A Vision-Language Model (VLM) generates these constraints from a single RGB image and a task description. The robot then executes a sequence of stages — approach, grasp, transit, place — by solving for end-effector positions that satisfy the constraints at each stage.

```
RGB-D capture → DINOv2 + MobileSAM keypoints → VLM constraint generation → subgoal solver → IK → robot
```

## Hardware

| Component | Spec |
|-----------|------|
| Robot arm | SO-101 (5-DOF, Feetech STS3215 motors) |
| Camera | Intel RealSense D435 (1280×720, RGB+depth) |
| Compute | Apple M1 MacBook (8GB RAM) |
| OS | macOS (conda env: `rekep_mac`) |

## Repository Structure

```
lerobot/
├── run_record_skills/      ← main implementation (start here)
│   ├── core/               ← pipeline modules
│   ├── tools/              ← diagnostic and utility scripts
│   ├── skills/             ← VLM skills library (failure patterns, guides)
│   └── outputs/            ← generated images and videos
├── calibration/            ← camera intrinsics and extrinsics
└── so101.urdf              ← robot kinematic model
```

> **All main work is in `run_record_skills/`.** See its [README](run_record_skills/run_record_skills.md) for details.

## Quick Start

```bash
conda activate rekep_mac
export OPENROUTER_API_KEY="sk-or-..."
cd run_record_skills
sudo -E python run_trajectory.py --task "pick up the tape and place it on the book"
```

## Key Contributions vs Original ReKep

| Aspect | Original ReKep | This Implementation |
|--------|---------------|---------------------|
| Arm | 6-DOF (UR5) | 5-DOF (SO-101) |
| IK solver | curobo (GPU) | custom scipy L-BFGS-B (CPU/MPS) |
| Subgoal solver | SE(3) pose optimization | 3D position + elevation/roll |
| Compute | GPU server | M1 MacBook |
| Depth | simulated | RealSense D435 |
| Orientation | full SE(3) | elevation angle + wrist roll |

## Calibration

Camera-robot extrinsic calibration via Umeyama 3D-3D registration. Mean residual: ~1.7cm.

```bash
# intrinsic calibration
python calibration/intrinsic.py

# extrinsic calibration (requires robot touches on stickers)
python calibration/extrinsics.py
```

## Dependencies

```bash
conda create -n rekep_mac python=3.11
pip install torch torchvision  # MPS backend for M1
pip install pyrealsense2 ikpy scipy opencv-python
pip install lerobot             # v0.3.2
pip install openai              # for OpenRouter VLM access
```

Model weights (download separately — not included due to size):
- `mobile_sam.pt` — [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- DINOv2 — downloaded automatically via HuggingFace (`facebook/dinov2-with-registers-small`)
