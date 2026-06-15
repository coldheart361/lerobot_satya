"""
test_pipeline.py
================
Step 1: Run DINOv2 + SAM keypoint proposal on an image
Step 2: Send the numbered overlay to GPT-4o via OpenRouter
Step 3: Print the generated constraint code

Usage:
    python test_pipeline.py --image images/random.jpg

Set your OpenRouter key first:
    $env:OPENROUTER_API_KEY = "sk-or-..."
"""

import os
import sys
import base64
import argparse
import numpy as np
from PIL import Image
from openai import OpenAI

# ── import the proposer we wrote ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from run.keypoint_proposal import KeypointProposer

TASK = "(single-arm) pick up the double-coated tape and lift it"

# ── prompt from ReKep paper appendix A.6 ─────────────────────────────────────
SYSTEM_PROMPT = """\
You are controlling a robot arm to perform manipulation tasks by writing
constraint functions in Python.

The task and a photo of the scene are given. Keypoints are overlaid as
numbered red dots on the image.

For each stage write:
- sub-goal constraints: must be satisfied AT THE END of the stage
- path constraints: must be satisfied THROUGHOUT the stage

Rules:
- Each constraint fn(end_effector, keypoints) -> float. Cost <= 0 = satisfied.
- end_effector: np.array shape (3,) — current EE position in metres
- keypoints: np.array shape (K, 3) — keypoint world positions in metres
- Grasping must be its own stage. For grasp stage: one sub-goal constraint
  aligning EE with a keypoint. No path constraints.
- No if-statements. No collision avoidance (handled separately).
- Use only numpy functions inside constraint functions.

Output ONLY a Python code block, nothing else:

```python
num_stages = ?

### stage 1 sub-goal constraints
def stage1_subgoal_constraint1(end_effector, keypoints):
    \"\"\"explanation\"\"\"
    ...
    return cost

### stage 1 path constraints
# (none)

# repeat for more stages
```
"""


def image_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_gpt4o(overlay_path, task, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    b64 = image_to_b64(overlay_path)
    print("[VLM] Querying GPT-4o via OpenRouter ...")
    resp = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": f"{SYSTEM_PROMPT}\n\nTask: \"{task}\"\n\n"
                            f"Look at the numbered keypoints in the image and "
                            f"write the constraint functions for this task.",
                },
            ],
        }],
        max_tokens=1500,
    )
    return resp.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",    required=True,  help="path to scene image")
    ap.add_argument("--overlay",  default="overlay.jpg",
                    help="where to save the numbered keypoint overlay")
    ap.add_argument("--task",     default=TASK,   help="task description")
    ap.add_argument("--k",        type=int, default=3,
                    help="k-means clusters per mask (default 3)")
    ap.add_argument("--min-mask", type=int, default=1000,
                    help="minimum mask size in pixels (default 1000)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY environment variable first")
        print('  PowerShell: $env:OPENROUTER_API_KEY = "sk-or-..."')
        sys.exit(1)

    # ── Step 1: keypoint proposal ─────────────────────────────────────────────
    print(f"\n=== Step 1: Keypoint Proposal ===")
    print(f"Image : {args.image}")
    print(f"Task  : {args.task}")

    rgb = np.array(Image.open(args.image).convert("RGB"))

    kp = KeypointProposer(
        k_per_mask=args.k,
        min_mask_pixels=args.min_mask,
        load_sam=True,
    )
    result = kp.get_keypoints(rgb, points=None, visualize=True)

    n = len(result["pixels"])
    print(f"\nFound {n} keypoints")

    if n == 0:
        print("No keypoints found. Try lowering --min-mask.")
        sys.exit(1)

    if n > 50:
        print(f"WARNING: {n} keypoints is a lot. GPT-4o works best with <40.")
        print("Consider increasing --min-mask or decreasing --k.")

    Image.fromarray(result["overlay"]).save(args.overlay)
    print(f"Overlay saved → {args.overlay}")
    print("Open it and check: do the numbered dots land on the tape?")
    input("\nPress ENTER to continue to GPT-4o query (or Ctrl+C to stop)...")

    # ── Step 2: VLM query ─────────────────────────────────────────────────────
    print(f"\n=== Step 2: GPT-4o Constraint Generation ===")
    raw = query_gpt4o(args.overlay, args.task, api_key)

    print("\n--- Raw GPT-4o output ---")
    print(raw)

    # ── Step 3: try to exec the code ─────────────────────────────────────────
    print("\n=== Step 3: Checking if code runs ===")
    code = raw
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    try:
        ns = {}
        exec(compile(code, "<gpt4o>", "exec"), ns)
        num_stages = ns.get("NUM_STAGES", ns.get("num_stages", "?"))
        print(f"✓ Code executes cleanly — {num_stages} stages")

        # quick sanity check: call stage 1 constraint with dummy values
        fns = [v for k, v in ns.items()
               if k.startswith("stage") and callable(v)]
        if fns:
            ee  = np.array([0.3, 0.1, 0.15])
            kps = np.random.rand(n, 3) * 0.3 + 0.1
            cost = fns[0](ee, kps)
            print(f"  Stage 1 constraint called → cost = {cost:.4f}  (should be a scalar)")
    except Exception as e:
        print(f"✗ Code failed to execute: {e}")
        print("  This means the prompt needs refinement for this model.")


if __name__ == "__main__":
    main()