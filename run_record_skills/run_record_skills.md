# run_record_skills

Main implementation directory. All pipeline code lives here.

## Structure

```
run_record_skills/
├── run_trajectory.py       ← MAIN ENTRY POINT — full pipeline
├── core/                   ← reusable pipeline modules
│   ├── ik_solver.py        ← IK solver (custom L-BFGS-B + ikpy chain)
│   ├── custom_ik.py        ← scipy-based IK with soft joint limit penalty
│   ├── subgoal_solver.py   ← constraint optimization (L-BFGS-B)
│   ├── keypoint_proposal.py← DINOv2 + MobileSAM keypoint pipeline
│   ├── constraint_generation.py ← VLM constraint generation
│   ├── helper.py           ← shared utilities (reachability, axes, projection)
│   └── vlm_openrouter.py   ← VLM client (OpenRouter API)
├── tools/                  ← diagnostic scripts
│   ├── click_and_go.py     ← click pixel → robot moves there
│   ├── click_orient_go.py  ← click + W/S/A/D to tune elevation/roll interactively
│   ├── ik_test.py          ← input 3D coords → check IK reachability, optionally move
│   ├── reachability.py     ← build reachability boundary overlay image
│   └── orientation_check.py← verify which FK axis is the gripper approach axis
├── skills/                 ← VLM prompt engineering
│   ├── loader.py           ← loads failure patterns into VLM prompt
│   ├── failures/           ← JSON records of known failure patterns
│   └── guides/             ← JSON reference guides (elevation/roll per object type)
└── outputs/                ← generated files (images, videos, constraints)
```

## Running the Pipeline

```bash
sudo -E python run_trajectory.py --task "pick up the tape and place it on the book"
sudo -E python run_trajectory.py --task "..." --dry-run   # VLM only, no robot
```

## Diagnostic Tools

```bash
# click a point, robot moves there (tests calibration + IK)
sudo -E python tools/click_and_go.py

# click + keyboard to tune approach angle interactively
# W/S = elevation +/-5°   A/D = roll +/-10°   SPACE = move
sudo -E python tools/click_orient_go.py

# check if a 3D coordinate is reachable at given angles
python tools/ik_test.py --x 25 --y -5 --z 8 --elevation 45 --roll 90
python tools/ik_test.py   # interactive mode

# build reachability boundary overlay (run once, slow)
python tools/reachability.py
```

## Pipeline Stages

Each run of `run_trajectory.py` executes:

1. **Capture** — RGB + aligned depth from RealSense D435
2. **Backproject** — depth → 3D point cloud in robot base frame
3. **Keypoint proposal** — DINOv2 features + MobileSAM masks → numbered 3D keypoints
4. **Constraint generation** — VLM sees overlay image + reachability report → Python constraint functions
5. **Stage execution loop** — for each stage:
   - Update held keypoints (rigidity assumption)
   - Solve subgoal (L-BFGS-B minimizing constraint costs)
   - IK → move robot → set gripper

## Orientation Control

The SO-101 is 5-DOF. Orientation is parameterized as:
- `approach_elevation_deg` — 0=horizontal, 90=top-down
- `gripper_roll_deg` — spin of fingers around approach axis

The VLM reasons about object geometry to choose these values. The reachability report (injected into the VLM prompt) shows IK error at each elevation for each keypoint.

## Skills Library

`skills/failures/` contains JSON records of known failure patterns injected into every VLM prompt as negative examples. Add new failures as you encounter them:

```json
{
  "id": "no_approach_stage",
  "category": "stage_structure",
  "description": "Skipped approach stage — gripper never opened before grasping",
  "correct_pattern": {
    "rule": "STAGE_GRIPPER_ACTION[0] must always be open"
  }
}
```

## Configuration

Key constants in `run_trajectory.py`:

```python
PORT            = "/dev/tty.usbmodem5B140297181"   # robot USB port
WS_MIN          = np.array([ 0.05, -0.25, -0.05])  # workspace bounds (metres)
WS_MAX          = np.array([ 0.45,  0.20,  0.30])
IK_TOLERANCE_CM = 1.5                               # max IK error to execute move
ROLL_CORRECTION = 0.582                             # wrist roll calibration factor
```