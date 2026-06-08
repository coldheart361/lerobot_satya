import os
import re
import numpy as np
from vlm_openrouter import VLM

# Embed the comprehensive prompt template from the paper
CONSTRAINT_PROMPT_TEMPLATE = """\
## Instructions
Suppose you are controlling a robot to perform manipulation tasks by writing constraint functions in Python. The manipulation task is given as an image of the environment, overlayed with keypoints marked with their indices, along with a text instruction. For each given task, please perform the following steps:
- Determine how many stages are involved in the task. Grasping must be an independent stage. Some examples:
  - "pouring tea from teapot":
    - 3 stages: "grasp teapot", "align teapot with cup opening", and "pour liquid"
  - "put red block on top of blue block":
    - 3 stages: "grasp red block", "drop the red block on top of blue block"
  - "reorient bouquet and drop it upright into vase":
    - 3 stages: "grasp bouquet", "reorient bouquet", and "keep upright and drop into vase"
- For each stage, write two kinds of constraints, "sub-goal constraints" and "path constraints". The "sub-goal constraints" are constraints that must be satisfied **at the end of the stage**, while the "path constraints" are constraints that must be satisfied **within the stage**. Some examples:
  - "pouring liquid from teapot":
    - "grasp teapot" stage:
      - 1 sub-goal constraints: "align the end-effector with the teapot handle"
      - 0 path constraints
    - "align teapot with cup opening" stage:
      - 1 sub-goal constraints: "the teapot spout needs to be 10cm above the cup opening"
      - 2 path constraints: "the robot must still be grasping the teapot handle", "the teapot must stay upright to avoid spilling"
    - "pour liquid" stage:
      - 2 sub-goal constraints: "the teapot spout needs to be 5cm above the cup opening", "the teapot spout must be tilted to pour liquid"
      - 2 path constraints: "the robot must still be grasping the teapot handle", "the teapot spout is directly above the cup opening"
  - "put red block on top of blue block":
    - "grasp red block" stage:
      - 1 sub-goal constraints: "align the end-effector with the red block"
      - 0 path constraints
    - "drop the red block on top of blue block" stage:
      - 1 sub-goal constraints: "the red block is 10cm on top of the blue block"
      - 1 path constraints: "the robot must still be grasping the red block"
  - "reorient bouquet and drop it upright into vase":
    - "grasp bouquet" stage:
      - 1 sub-goal constraints: "align the end-effector with the bouquet stem"
      - 0 path constraints
    - "reorient bouquet" stage:
      - 1 sub-goal constraints: "the bouquet is upright (parallel to the z-axis)"
      - 1 path constraints: "the robot must still be grasping the bouquet stem"
    - "keep upright and drop into vase" stage:
      - 2 sub-goal constraints: "the bouquet must still stay upright (parallel to the z-axis)", "the bouquet is 20cm above the vase opening"
      - 1 path constraints: "the robot must still be grasping the bouquet stem"
- Summarize keypoints to be grasped in all grasping stages by defining the `grasp_keypoints` variable.
- Summarize at the end of which stage the robot should release the keypoints by defining the `release_keypoints` variable.

**Note:**
- Each constraint takes a dummy end-effector point and a set of keypoints as input and returns a numerical cost, where the constraint is satisfied if the cost is smaller than or equal to zero.
- For each stage, you may write 0 or more sub-goal constraints and 0 or more path constraints.
- Avoid using "if" statements in your constraints.
- Avoid using path constraints when manipulating deformable objects (e.g., clothing, towels).
- You do not need to consider collision avoidance. Focus on what is necessary to complete the task.
- Inputs to the constraints are as follows:
  - `end_effector`: np.array of shape `(3,)` representing the end-effector position.
  - `keypoints`: np.array of shape `(K, 3)` representing the keypoint positions.
- For any path constraint that requires the robot to be still grasping a keypoint `i`, you may use the provided function `get_grasping_cost_by_keypoint_idx` by calling `return get_grasping_cost_by_keypoint_idx(i)` where `i` is the index of the keypoint. 
- Inside of each function, you may use native Python functions, any NumPy functions, and the provided `get_grasping_cost_by_keypoint_idx` function.
- For grasping stage, you should only write one sub-goal constraint that associates the end-effector with a keypoint. No path constraints are needed.
- In order to move a keypoint, its associated object must be grasped in one of the previous stages.
- The robot can only grasp one object at a time.
- Grasping must be an independent stage from other stages.
- You may use two keypoints to form a vector, which can be used to specify a rotation (by specifying the angle between the vector and a fixed axis).
- You may use multiple keypoints to specify a surface or volume.
- The keypoints marked on the image start with index 0, same as the given argument `keypoints` array.
- For a point `i` to be relative to another point `j`, the function should define an `offsetted_point` variable that has the delta added to keypoint `j` and then calculate the norm of the xyz coordinates of the keypoint `i` and the `offsetted_point`.
- If you would like to specify a location not marked by a keypoint, try using multiple keypoints to specify the location (e.g., you may take the mean of multiple keypoints if the desired location is in the center of those keypoints).

**Structure your output in a single python code block as follows:**
```python

# Your explanation of how many stages are involved in the task and what each stage is about.
# ...

num_stages = [INSERT_NUM_STAGES]

### stage 1 sub-goal constraints (if any)
def stage1_subgoal_constraint1(end_effector, keypoints):
    \"\"\"Put your explanation here.\"\"\"
    ...
    return cost

### stage 1 path constraints (if any)
def stage1_path_constraint1(end_effector, keypoints):
    \"\"\"Put your explanation here.\"\"\"
    ...
    return cost

# repeat for more stages
...

grasp_keypoints = [..., ..., ...]
release_keypoints = [..., ..., ...]
Query
Query Task: "{instruction}"
Query Image:
"""

def _extract_and_write_separate_constraints(full_code_str, target_dir, num_stages):
    """Splits the raw VLM script into separate txt files expected by the framework."""
    lines = full_code_str.split('\n')

    for stage in range(1, num_stages + 1):
        for c_type in ['subgoal', 'path']:
            filename = f"stage{stage}_{c_type}_constraints.txt"
            filepath = os.path.join(target_dir, filename)
            
            # Find lines matching function definitions for this explicit category
            stage_fns = []
            capture = False
            current_fn_lines = []
            
            for line in lines:
                # Check for starting signatures
                if line.startswith(f"def stage{stage}_{c_type}_"):
                    capture = True
                    if current_fn_lines:
                        stage_fns.append("\n".join(current_fn_lines))
                        current_fn_lines = []
                elif line.startswith("def stage") or line.startswith("grasp_keypoints") or line.startswith("release_keypoints"):
                    if capture and current_fn_lines:
                        stage_fns.append("\n".join(current_fn_lines))
                        current_fn_lines = []
                    capture = False
                
                if capture:
                    current_fn_lines.append(line)
            
            if capture and current_fn_lines:
                stage_fns.append("\n".join(current_fn_lines))
                
            # Write out to disk directory
            with open(filepath, 'w') as f:
                if stage_fns:
                    f.write("\n\n".join(stage_fns) + "\n")
                else:
                    f.write("") # Keep file empty if no path constraints were defined
def generate_constraints(overlay_rgb: np.ndarray, task: str, num_keypoints: int, vlm: VLM) -> dict:
    """
    Queries the high-end VLM to map semantics to constraints, executes
    the resulting source in memory, and logs backups to vlm_query/constraints/
    """
    # 1. Prepare target output workspace
    output_dir = os.path.join(os.path.dirname(os.path.abspath(file)), 'vlm_query', 'constraints')
    os.makedirs(output_dir, exist_ok=True)

    # 2. Format query structures
    prompt = CONSTRAINT_PROMPT_TEMPLATE.replace("[INSERT_NUM_STAGES]", "YOUR_DECISION")
    prompt = f"{prompt}\nTask: \"{task}\"\nNumber of keypoints marked: {num_keypoints}"

    print(f"[ConstraintGen] Sending frame analysis token stream via {vlm.model}...")
    raw_response = vlm.query(prompt, image=overlay_rgb, max_tokens=3000)

    # 3. Clean markdown code fences
    code_block = raw_response
    if "```python" in code_block:
        code_block = code_block.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in code_block:
        code_block = code_block.split("```", 1)[1].split("```", 1)[0]
        
    # Write full monolithic backup raw file
    with open(os.path.join(output_dir, 'full_generated_code.py'), 'w') as f:
        f.write(code_block.strip() + "\n")

    # 4. Create local runtime environment namespace with injected hooks
    # ReKep placeholder mapping for grasping indicators inside simulation scopes
    def mock_grasping_cost(keypoint_idx):
        return 0.0 # Standard local pass override hook
        
    ns = {
        "np": np, 
        "numpy": np,
        "get_grasping_cost_by_keypoint_idx": mock_grasping_cost
    }

    try:
        exec(compile(code_block, "<vlm_constraints_evaluation>", "exec"), ns)
    except Exception as e:
        print(f"[ConstraintGen] DYNAMIC COMPILE FAILED: {e}")
        print("Review code artifact in vlm_query/constraints/full_generated_code.py")
        raise

    num_stages = ns.get("num_stages")
    if num_stages is None:
        raise ValueError("VLM code execution completed but variable 'num_stages' was undefined.")

    # 5. Extract structured text files for the path loading modules
    _extract_and_write_separate_constraints(code_block, output_dir, num_stages)
    print(f"[ConstraintGen] Complete. Segmented assets written to: {output_dir}")

    # 6. Gather tracking items for real-time memory usage
    # This automatically matches your main pipeline parsing lookups
    stage_constraints = []
    stage_path_constraints = []

    for s in range(1, num_stages + 1):
        sub_list = []
        path_list = []
        for name, obj in ns.items():
            if callable(obj):
                if name.startswith(f"stage{s}_subgoal_"):
                    sub_list.append(obj)
                elif name.startswith(f"stage{s}_path_"):
                    path_list.append(obj)
        stage_constraints.append(sub_list)
        stage_path_constraints.append(path_list)

    return {
        "num_stages": num_stages,
        "stage_constraints": stage_constraints,
        "stage_path_constraints": stage_path_constraints,
        "grasp_keypoints": ns.get("grasp_keypoints", []),
        "release_keypoints": ns.get("release_keypoints", []),
        "code_str": code_block
    }