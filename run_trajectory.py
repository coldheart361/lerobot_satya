import os
import sys
import time
import json
import argparse
import subprocess
import signal
import numpy as np
import cv2
import pyrealsense2 as rs

from ikpy.chain import Chain
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keypoint_proposal      import KeypointProposer
from vlm_openrouter         import VLM
from constraint_generation  import generate_constraints
from subgoal_solver         import solve_subgoal
from path_solver            import PathSolver
import transform_utils as T

# ── CONFIG ────────────────────────────────────────────────────────────────────
PORT = "/dev/tty.usbmodem5B140297181"
URDF_PATH = "../so101.urdf"

# Workspace safety bounds in robot base frame (METRES)
WS_MIN = np.array([ 0.05, -0.25, -0.05])
WS_MAX = np.array([ 0.45,  0.20,  0.30])

IK_TOLERANCE_CM = 5.0
TCP_OFFSET_M = np.array([0.004016, -0.004152, 0.015589, 1.0])

GRIPPER_OPEN_VAL   = 50.0
GRIPPER_CLOSED_VAL = 0.0

# PathSolver step sizes and boundaries configuration
SOLVER_CONFIG = {
    'bounds_min': WS_MIN,
    'bounds_max': WS_MAX,
    'opt_pos_step_size': 0.15,          # Size of coarse optimization steps
    'opt_rot_step_size': 0.5,
    'opt_interpolate_pos_step_size': 0.03, # Fine dense waypoint interpolation size
    'opt_interpolate_rot_step_size': 0.1,
    'sampling_maxfun': 100,             # Fast global search budgets
    'minimizer_options': {'maxiter': 15, 'ftol': 1e-3}, # Snap fast local passes
    'constraint_tolerance': 1e-3
}

# ── LOAD CALIBRATION ──────────────────────────────────────────────────────────
print("Loading calibration...")
K = np.load("../calibration/intrinsics/camera_matrix.npy")
depth_scale = np.load("../calibration/intrinsics/depth_scale.npy")[0]
T_bc = np.load("../calibration/T_base_camera.npy")

# ── CAMERA HELPERS ────────────────────────────────────────────────────────────
def reset_camera():
    subprocess.run(["killall", "VDCAssistant"], capture_output=True)
    subprocess.run(["killall", "AppleCameraAssistant"], capture_output=True)
    try:
        ctx = rs.context()
        if len(ctx.devices) > 0:
            ctx.devices[0].hardware_reset()
            time.sleep(3)
    except Exception:
        pass

def try_capture():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgra8, 6)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,   6)
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"  start failed: {e}")
        return None, None
    align = rs.align(rs.stream.color)
    try:
        for i in range(50):
            frames = pipeline.wait_for_frames(5000)
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            depth = np.asarray(depth_frame.get_data())
            valid = np.sum((depth > 0) & (depth < 65535))
            if valid > 800000:
                color = np.asarray(frames.get_color_frame().get_data())
                rgb = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
                pipeline.stop()
                return rgb, depth
        pipeline.stop()
        return None, None
    except Exception as e:
        print(f"  capture failed: {e}")
        try: pipeline.stop()
        except Exception: pass
        return None, None

def capture_with_retry(max_attempts=10):
    for attempt in range(1, max_attempts + 1):
        print(f"\nCapture attempt {attempt}...")
        reset_camera()
        time.sleep(1)
        rgb, depth = try_capture()
        if rgb is not None:
            return rgb, depth
        print("  failed, retrying...")
        time.sleep(1)
    raise RuntimeError("Could not capture")

def build_points_array(depth_raw):
    H, W = depth_raw.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z_m = depth_raw.astype(np.float32) * depth_scale
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z_cm = z_m * 100.0
    X_cm = (us - cx) * z_cm / fx
    Y_cm = (vs - cy) * z_cm / fy
    Z_cm = z_cm
    invalid = (depth_raw == 0) | (depth_raw >= 65535)
    X_cm[invalid] = np.nan; Y_cm[invalid] = np.nan; Z_cm[invalid] = np.nan
    ones = np.ones_like(X_cm)
    cam_homo  = np.stack([X_cm, Y_cm, Z_cm, ones], axis=-1)
    base_homo = cam_homo @ T_bc.T
    return base_homo[..., :3] / 100.0

# ── ROBOT KINEMATICS SETUP ────────────────────────────────────────────────────
print("Loading IK chain...")
chain = Chain.from_urdf_file(
    URDF_PATH,
    base_elements=["base_link", "shoulder_pan", "shoulder_link",
                   "shoulder_lift", "upper_arm_link",
                   "elbow_flex", "lower_arm_link",
                   "wrist_flex", "wrist_link",
                   "wrist_roll", "gripper_link",
                   "gripper_frame_joint", "gripper_frame_link"],
    base_element_type="link",
    active_links_mask=[False, True, True, True, True, True, False],
)

def current_joints(robot):
    obs = robot.get_observation()
    q = np.zeros(7)
    q[1:6] = np.deg2rad([
        obs["shoulder_pan.pos"], obs["shoulder_lift.pos"],
        obs["elbow_flex.pos"],   obs["wrist_flex.pos"],
        obs["wrist_roll.pos"],
    ])
    for i, link in enumerate(chain.links):
        if hasattr(link, "bounds") and link.bounds is not None:
            q[i] = np.clip(q[i], link.bounds[0], link.bounds[1])
    return q

def current_ee_pose_7d(robot):
    """Returns 7D pose array: [x, y, z, qx, qy, qz, qw] via Forward Kinematics."""
    q = current_joints(robot)
    fk = chain.forward_kinematics(q)
    ee_mat = fk @ TCP_OFFSET_M[:, None] # Apply tool center point offset matrix
    pos = ee_mat[:3, 0]
    
    # Extract orientation from upper-left 3x3 rotation block
    quat = T.convert_pose_mat2quat(fk)
    return np.concatenate([pos, quat])

def execute_trajectory(robot, waypoints, gripper_val, label=""):
    """Drives the arm sequentially down an optimized list of 7D target waypoints."""
    for idx, pose in enumerate(waypoints):
        target_m = pose[:3]
        if np.any(target_m < WS_MIN) or np.any(target_m > WS_MAX):
            continue
            
        guess = current_joints(robot)
        joints = chain.inverse_kinematics(target_m, initial_position=guess)
        tip_m = (chain.forward_kinematics(joints) @ TCP_OFFSET_M)[:3]
        err_cm = np.linalg.norm(tip_m - target_m) * 100
        
        if err_cm > IK_TOLERANCE_CM:
            print(f"  [{label}] Step {idx} IK error too high ({err_cm:.2f} cm) — skipping step")
            continue
            
        angles = np.rad2deg(joints[1:6])
        robot.send_action({
            "shoulder_pan.pos":  float(angles[0]),
            "shoulder_lift.pos": float(angles[1]),
            "elbow_flex.pos":    float(angles[2]),
            "wrist_flex.pos":    float(angles[3]),
            "wrist_roll.pos":    float(angles[4]),
            "gripper.pos":       float(gripper_val),
        })
        time.sleep(0.3) # Fast execution interpolation delay

# ── KEYPOINT TRACKING CORE ────────────────────────────────────────────────────
def update_keypoints(initial_kps, movable_mask, ee_at_grasp, ee_now):
    kps = initial_kps.copy()
    if ee_at_grasp is None or ee_now is None:
        return kps
    displacement = ee_now - ee_at_grasp
    for i, is_held in enumerate(movable_mask):
        if is_held:
            kps[i] = initial_kps[i] + displacement
    return kps

# ── MAIN PIPELINE COORDINATOR ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 1. Perception
    print("\n=== Step 1: Capture scene ===")
    rgb, depth_raw = capture_with_retry()
    os.makedirs("image", exist_ok=True)
    cv2.imwrite("image/traj_scene.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print("\n=== Step 2: Backproject ===")
    points_base_m = build_points_array(depth_raw)

    print("\n=== Step 3: Keypoint proposal ===")
    kp = KeypointProposer(k_per_mask=3, min_mask_pixels=1000, workspace_bounds=(WS_MIN, WS_MAX))
    result = kp.get_keypoints(rgb, points=points_base_m, visualize=True)
    overlay = result["overlay"]
    initial_kps = result["keypoints_3d"]   # [K, 3] matrix
    K_kps = len(initial_kps)
    cv2.imwrite("image/traj_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Registered {K_kps} base spatial keypoints.")

    if K_kps == 0:
        print("No reachable keypoints discovered. Aborting execution loop.")
        return

    # 2. Constraint Generation
    print(f"\n=== Step 4: Querying VLM for Task Logic ===")
    vlm = VLM(model="anthropic/claude-opus-4.8")
    cg = generate_constraints(overlay, args.task, num_keypoints=K_kps, vlm=vlm)
    print(f"Task compiled into {cg['num_stages']} operational stage definitions.")

    if args.dry_run:
        print("\n[Dry Run Flag Set] Terminating pipeline before hardware assertion.")
        return

    # 3. Connect to Physical Arm
    print("\n=== Step 5: Connecting to SO101 Hardware ===")
    rconfig = SO101FollowerConfig(port=PORT, id="my_arm")
    robot = SO101Follower(rconfig)
    robot.connect(calibrate=False)

    def safe_exit(sig=None, frame=None):
        print("\nExecution interrupted. Safe-stopping and dropping motor torques...")
        try: robot.disconnect()
        except Exception: pass
        sys.exit(0)
    signal.signal(signal.SIGINT, safe_exit)

    # Instantiate our optimized path generator
    reset_joints = current_joints(robot)
    path_solver = PathSolver(config=SOLVER_CONFIG, chain=chain, tcp_offset=TCP_OFFSET_M)

    input("\nClear the scene workspace boundaries and press [ENTER] to execute...")

    # ── CLOSED-LOOP RE-PLANNING EXECUTION LOOP ────────────────────────────────
    ee_at_grasp = None
    from_scratch = True # Force global optimization ONLY on stage initiation

    try:
        for s in range(cg["num_stages"]):
            print(f"\n============================================================")
            print(f"EXECUTING STAGE {s+1}/{cg['num_stages']}: {cg['stage_names'][s]}")
            print(f"============================================================")

            mask = cg["stage_movable_mask"][s] if s < len(cg["stage_movable_mask"]) else [False]*K_kps
            subgoal_constraints = cg["stage_constraints"][s] if s < len(cg["stage_constraints"]) else []
            # Extract distinct path constraints for orientation tracking
            path_constraints = cg["stage_path_constraints"][s] if "stage_path_constraints" in cg and s < len(cg["stage_path_constraints"]) else []
            gripper_state = cg["stage_gripper_action"][s] if s < len(cg["stage_gripper_action"]) else "open"

            # 1. Update internal keypoint tracking states
            curr_ee_pose = current_ee_pose_7d(robot)
            kps_now = update_keypoints(initial_kps, mask, ee_at_grasp, curr_ee_pose[:3])

            # 2. Step A: Compute the Subgoal target state boundary
            print("  Solving subgoal constraints target...")
            target_pos_3d, cost = solve_subgoal(
                subgoal_constraints, kps_now,
                workspace_min=WS_MIN, workspace_max=WS_MAX,
                movable_mask=mask, initial_ee=curr_ee_pose[:3],
                from_scratch=from_scratch
            )
            
            # Formulate full target pose (Match subgoal position, inherit current EE orientation)
            target_pose_7d = np.concatenate([target_pos_3d, curr_ee_pose[3:]])

            # 3. Step B: Generate smooth trajectory waypoints satisfying tracking parameters
            print("  Optimizing path constraints trajectory...")
            try:
                waypoints = path_solver.solve(
                    start_pose=curr_ee_pose,
                    end_pose=target_pose_7d,
                    keypoints=kps_now,
                    keypoint_movable_mask=mask,
                    path_constraints=path_constraints,
                    initial_joint_pos=current_joints(robot),
                    from_scratch=from_scratch
                )
                from_scratch = False # Switch to hyper-fast local minimize cycles for the remainder of this stage
            except Exception as e:
                print(f"  PathSolver error: {e}. Falling back to clean linear interpolation.")
                waypoints = [target_pose_7d]

            # 4. Step C: Drive hardware down the optimized waypoint path
            gripper_val = GRIPPER_CLOSED_VAL if gripper_state == "closed" else GRIPPER_OPEN_VAL
            execute_trajectory(robot, waypoints, gripper_val, label=cg["stage_names"][s])

            # 5. Handle grasp moment rigidity lock transitions
            next_mask = cg["stage_movable_mask"][s+1] if s+1 < len(cg["stage_movable_mask"]) else mask
            if any(next_mask) and not any(mask) and ee_at_grasp is None:
                ee_at_grasp = current_ee_pose_7d(robot)[:3]
                print(f"  [Grasp Locked] Rigidity anchor point captured at: {ee_at_grasp}")

            # Reset the warm-start seed so the next stage initiates a fresh global optimization sweep
            from_scratch = True

        print("\n=== Pipeline Executed Successfully! ===")

    finally:
        print("\nClosing connection and cleaning up resources...")
        robot.disconnect()

if __name__ == "__main__":
    main()