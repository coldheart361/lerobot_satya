"""
constraint_generation.py
========================
Generate Python constraint functions + grasp orientation from VLM.

ORIENTATION EXTENSION:
  VLM now also outputs:
    approach_elevation_deg : 0=horizontal approach, 90=top-down (default)
    gripper_roll_deg       : spin of gripper around the approach axis

  How the VLM reasons about these:
    - Cylinders (tape roll, bottle): side approach ~30-45deg, roll to align
      fingers along the long axis
    - Flat objects (book, paper): top-down 90deg
    - Boxes: 90deg top-down is fine
    - Elongated objects lying flat: 90deg top-down but roll 90deg so fingers
      straddle the long axis
"""

import numpy as np
from .vlm_openrouter import VLM
from .helper import check_keypoint_reachability, check_reachability, reachability_report_str
from skills.loader import load_failure_skills, format_failures_for_prompt

CONSTRAINT_PROMPT = """\
You are controlling a robot arm to perform manipulation tasks by writing
Python constraint functions.

The image shows the scene with numbered red keypoints. The task is given as text.

## Your job

Before writing constraints, analyze the target object's geometry:

1. What is the object being grasped? (from the task description)
2. What geometric primitive best describes it?
   - thin disk / roll (tape, spool) → approach from the side, 
     fingers wrap around the rim
   - elongated cylinder (pen, bottle) → approach perpendicular 
     to the long axis
   - flat rectangular (book, phone) → top-down
   - box / cube → top-down
3. Based on that shape, choose approach_elevation_deg and 
   gripper_roll_deg

Example reasoning:
  "The object is a tape roll — a thin disk. 
   Approaching from the top (elevation=90) would press on the 
   flat face which is slippery. 
   Approaching from the side (elevation=30-45) lets the fingers 
   wrap around the rim — much more stable.
   gripper_roll=90 aligns the fingers parallel to the rim edge."

For each task:

1. Decide num_stages. Grasping MUST be its own stage. Example for pick+place:
   stage 1: approach object (EE above it)
   stage 2: grasp object (EE at it, gripper closes)
   stage 3: transit to target
   stage 4: place (release)

2. For each stage write constraint functions:
   fn(end_effector: np.ndarray[3], keypoints: np.ndarray[K,3]) -> float
   cost <= 0 means satisfied. Lower is better.

   Note that You have access to check_reachability(point_m, elevation_deg, roll_deg) which returns {"reachable": bool, "ik_error_cm": float}. You must use it to verify approach points before committing to them!

   Report if you cannot use it.

    Example:
    # verify approach point is reachable before writing constraint
    approach_pt = keypoints[0] + np.array([0, 0, 0.12])
    reach = check_reachability(approach_pt, elevation_deg=90)
    # if not reachable, reduce hover height or change elevation
    hover_z = 0.12 if reach["reachable"] else 0.06
    
    def stage1_subgoal_constraint1(end_effector, keypoints):
        target = keypoints[0] + np.array([0.0, 0.0, hover_z])
        return np.linalg.norm(end_effector - target)

3. STAGE_MOVABLE_MASK[stage]: list of bool length K.
   True = that keypoint moves with the gripper (object being held).

4. STAGE_GRIPPER_ACTION[i]: "open" or "closed" at END of stage i.

5. approach_elevation_deg: how steeply the gripper descends to grasp.
   - 90 = straight down from above (top-down) — best for flat objects, boxes
   - 45 = diagonal approach — good for most objects
   - 0  = horizontal approach — rarely useful, hard for arm to achieve
   Reason about the object shape visible in the image.

6. gripper_roll_deg: rotation of the gripper fingers around the approach axis.
   - 0   = default finger orientation
   - 90  = fingers rotated 90deg — good for cylindrical objects (tape roll,
           bottle) where you want fingers along the long axis
   Reason about which way the object's longest axis is oriented.

## Rules
- Each constraint takes end_effector [3] and keypoints [K,3], returns float.
- No if-statements. Use np.linalg.norm, abs, etc.
- All positions in METRES, robot base frame.
- For grasp stage: ONE subgoal constraint aligning EE with the grasp keypoint.
- For transit/place: reference keypoint positions, not end_effector directly.
- The image shows colored arrows indicating the robot's coordinate axes:
  RED arrow   = +X direction (forward, away from robot base)
  GREEN arrow = +Y direction (left when facing the robot)
  BLUE arrow  = +Z direction (up)
Origin is at the robot base center.
Use these when writing XYZ offsets relative to keypoints.
- MANDATORY: At the TOP of your code (before function definitions), call 
check_reachability and print the result:

# then use hover_z in your constraint:
def stage1_subgoal_constraint1(end_effector, keypoints):
    target = keypoints[GRASP_KP] + np.array([0.0, 0.0, hover_z])
    return float(np.linalg.norm(end_effector - target))

## Output ONLY a Python code block:

```python
num_stages = ...

def stage1_subgoal_constraint1(end_effector, keypoints):
    ...

STAGE_CONSTRAINTS      = [[...], ...]
STAGE_PATH_CONSTRAINTS = [[], ...]
STAGE_NAMES            = [...]
STAGE_MOVABLE_MASK     = [[False]*K, ...]
STAGE_GRIPPER_ACTION   = [...]

grasp_keypoints   = [keypoint_index_to_grasp]
release_keypoints = [stage_index_to_release]  # which stage opens gripper

approach_elevation_deg = 90   # 0=horizontal, 90=top-down
gripper_roll_deg       = 0    # spin of fingers around approach axis
```
"""


def _safe_wrap(fn):
    """Broken constraint → huge penalty, not 0 (0 would mean 'satisfied')."""
    def safe(ee, kps):
        try:
            return float(fn(ee, kps))
        except Exception:
            return 1e6
    return safe

def generate_constraints(overlay_rgb, task, num_keypoints, keypoints_3d, ik_solver, vlm=None):
    vlm = vlm or VLM()
    failures = load_failure_skills()
    failure_str = format_failures_for_prompt(failures)
    
    kp_report = check_keypoint_reachability(keypoints_3d, ik_solver)

    def _check_reachability(point_m, elevation_deg=90, roll_deg=0):
        return check_reachability(point_m, ik_solver, elevation_deg, roll_deg)

    report_str = reachability_report_str(kp_report)

    prompt = (f"{CONSTRAINT_PROMPT}\n\n{failure_str}\n\n"
              f"Task: \"{task}\"\n"
              f"Number of keypoints in image: {num_keypoints}"
              f"""
                REACHABILITY REPORT (IK error in cm, <5cm = reachable):
                {report_str}

                Choose approach_elevation_deg based on the grasp keypoint's reachability above.
                """)

    raw = vlm.query(prompt, image=overlay_rgb, max_tokens=3000)
    print(f"[ConstraintGen] Raw VLM output:\n{raw}\n")

    # extract code block
    code = raw
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0]

    ns = {"np": np,
          "numpy": np,
          "check_reachability": _check_reachability, 
          "keypoints_3d": keypoints_3d,
          "keypoints": keypoints_3d, }
    try:
        exec(compile(code, "<vlm_constraints>", "exec"), ns)
    except Exception as e:
        print(f"[ConstraintGen] compile failed: {e}\nCode:\n{code}")
        raise

    num_stages = ns.get("num_stages")
    if num_stages is None:
        raise ValueError("VLM output missing num_stages")

    stage_constraints = [
        [_safe_wrap(fn) for fn in stage]
        for stage in ns.get("STAGE_CONSTRAINTS", [])
    ]
    stage_path_constraints = [
        [_safe_wrap(fn) for fn in stage]
        for stage in ns.get("STAGE_PATH_CONSTRAINTS", [])
    ]

    return {
        "num_stages":             num_stages,
        "stage_constraints":      stage_constraints,
        "stage_path_constraints": stage_path_constraints,
        "stage_names":            ns.get("STAGE_NAMES", []),
        "stage_movable_mask":     ns.get("STAGE_MOVABLE_MASK", []),
        "stage_gripper_action":   ns.get("STAGE_GRIPPER_ACTION", []),
        "grasp_keypoints":        ns.get("grasp_keypoints", []),
        "release_keypoints":      ns.get("release_keypoints", []),
        "approach_elevation_deg": float(ns.get("approach_elevation_deg", 90)),
        "gripper_roll_deg":       float(ns.get("gripper_roll_deg", 0)),
        "code_str":               code,
    }