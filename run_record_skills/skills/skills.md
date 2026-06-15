# skills/

VLM prompt engineering library. Injects known failure patterns and reference guides into every constraint generation call so the VLM avoids repeating past mistakes.

## Structure

```
skills/
├── loader.py       ← loads and formats all skills into prompt text
├── failures/       ← things that went wrong and how to fix them
│   ├── no_approach_stage.json
│   ├── upright_object_wrong_roll.json
│   ├── shallow_elevation_near_base.json
│   └── ...
└── guides/
    └── gripper_orientation_guide.json   ← elevation/roll reference per object type
```

## How It Works

`loader.py` reads all JSON files and formats them as prompt text:

```
## GRIPPER ORIENTATION REFERENCE:
  object_standing_upright: elevation=20°  roll=90°  (e.g. standing box, bottle, pen)
  cylinder_lying_on_side:  elevation=30°  roll=90°  (e.g. tape roll, bottle on side)
  ...

## KNOWN FAILURE PATTERNS (avoid these):
  ✗ no_approach_stage: Skipped approach stage, gripper never opened before grasping
    → STAGE_GRIPPER_ACTION[0] must always be open
  ✗ upright_object_wrong_roll: Used roll=0 for standing object, fingers opened vertically
    → For standing objects use roll=90 so fingers open horizontally
```

## Adding a New Failure

Create a JSON file in `failures/`:

```json
{
  "id": "descriptive_id",
  "category": "stage_structure | grasp_orientation | subgoal_solver | ik_solver | keypoints",
  "description": "What went wrong in one sentence",
  "bad_example": {
    "what_happened": "Robot did X when it should have done Y"
  },
  "correct_pattern": {
    "rule": "The rule to follow instead"
  }
}
```