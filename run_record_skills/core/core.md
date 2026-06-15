# core/

Reusable pipeline modules. Import these from `run_trajectory.py` and the diagnostic tools.

## Modules

### `ik_solver.py` — `IKSolver`
Wraps the ikpy kinematic chain with orientation-aware IK and multiple restarts.

```python
ik = IKSolver(urdf_path, tcp_offset_m)
joints, err = ik.solve(target_m, approach_dir=[0,0,-1], roll_override=roll_rad)
action = ik.joints_to_action(joints)   # → lerobot degree dict
```

- `approach_dir` — unit vector for gripper approach direction (SO-101 uses Z axis)
- `roll_override` — directly sets wrist_roll after IK (bypasses orientation constraint)
- Multiple random restarts via `custom_ik.py` to escape local minima

### `custom_ik.py` — `solve_lm()`
Custom L-BFGS-B IK objective with soft joint limit penalty. Avoids slamming into joint limits (the main cause of poor IK accuracy at elevation=90°).

Objective:
```
minimize: position_error_cm² + 30 × orientation_error + 10 × joint_limit_penalty
```

### `subgoal_solver.py` — `solve_subgoal()`
Finds the EE position that minimizes VLM constraint costs. Uses L-BFGS-B with multiple random restarts and held keypoint propagation.

Key: for stages where the robot holds an object (transit, place), held keypoints are projected forward to the candidate EE position so the constraint gradient is correct.

### `keypoint_proposal.py` — `KeypointProposer`
DINOv2 features + MobileSAM masks → semantically meaningful 3D keypoints.

Pipeline: DINOv2 → SAM masks → PCA + k-means per mask → project to 3D → MeanShift merge → workspace filter

### `constraint_generation.py` — `generate_constraints()`
Sends the keypoint overlay image to the VLM and gets back executable Python constraint functions.

VLM outputs:
- Stage constraint functions `f(end_effector, keypoints) → float`
- `STAGE_MOVABLE_MASK` — which keypoints move with the gripper
- `STAGE_GRIPPER_ACTION` — open/closed per stage
- `approach_elevation_deg`, `gripper_roll_deg` — grasp orientation

The VLM also has access to `check_reachability(point_m, elevation_deg)` which it can call at code-generation time to verify approach heights before committing to them.

### `helper.py` — utility functions
- `approach_from_elevation(target_m, elevation_deg)` — compute approach direction vector
- `project_point_to_pixel(point_base_cm, T_bc, K)` — 3D robot frame → image pixel
- `draw_robot_axes(image, T_bc, K)` — draw X/Y/Z arrows from robot base onto image
- `check_keypoint_reachability(keypoints_3d, ik_solver)` — test IK at each keypoint
- `reachability_report_str(report)` — format as text for VLM prompt
- `build_boundary_overlay(image, ik_solver, T_bc, K)` — reachability annulus visualization

### `vlm_openrouter.py` — `VLM`
OpenRouter API client. Supports image input via base64.

```python
vlm = VLM(model="anthropic/claude-sonnet-4.6")
response = vlm.query(prompt, image=overlay_rgb, max_tokens=3000)
```