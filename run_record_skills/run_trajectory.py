"""
run_trajectory.py — full ReKep pipeline
========================================
Pipeline:
  1. Capture RGB + aligned depth from D435
  2. Backproject depth -> points[H,W,3] in robot base frame
  3. DINOv2 + MobileSAM + clustering -> numbered keypoint overlay + 3D positions
  4. VLM -> Python constraint code for each stage
  5. STAGE EXECUTION LOOP:
       for each stage:
         a. update keypoint positions (rigidity for held objects)
         b. solve subgoal -> EE target via scipy (orientation-aware for grasps)
         c. IK + move robot to target (top-down approach for grasp stages)
         d. set gripper state for this stage
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import threading

import numpy as np
import cv2
import pyrealsense2 as rs

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.keypoint_proposal     import KeypointProposer
from core.vlm_openrouter        import VLM
from core.constraint_generation import generate_constraints
from core.subgoal_solver        import solve_subgoal
from core.ik_solver             import IKSolver          
from core.helper                import (approach_from_elevation, draw_robot_axes, build_points_array)


# ── CONFIG ────────────────────────────────────────────────────────────────────
PORT      = "/dev/tty.usbmodem5B140297181"
URDF_PATH = "../so101.urdf"

WS_MIN = np.array([ 0.05, -0.25, -0.05])
WS_MAX = np.array([ 0.45,  0.20,  0.30])

IK_TOLERANCE_CM    = 5.0
TCP_OFFSET_M       = np.array([0.004016, -0.004152, 0.015589, 1.0])
GRIPPER_OPEN_VAL   = 50.0
GRIPPER_CLOSED_VAL = 0.0


# ── LOAD CALIBRATION ──────────────────────────────────────────────────────────
print("Loading calibration...")
K           = np.load("../calibration/intrinsics/camera_matrix.npy")
depth_scale = np.load("../calibration/intrinsics/depth_scale.npy")[0]
T_bc        = np.load("../calibration/T_base_camera.npy")


# ── IK SOLVER (instantiated once at startup) ──────────────────────────────────
# The IKSolver wraps ikpy and adds orientation-aware solve():
#   - solve(xyz)                       : position only (transit/place stages)
#   - solve(xyz, approach_dir=[0,0,-1]): top-down approach (grasp stages)
# Having it as a module means subgoal_solver can also call it for reachability.
print("Loading IK solver...")
ik_solver = IKSolver(urdf_path=URDF_PATH, tcp_offset_m=TCP_OFFSET_M)


# ── VIDEO RECORDER ────────────────────────────────────────────────────────────
_record_flag = threading.Event()

def _record_thread(filename):
    subprocess.run(["killall", "VDCAssistant"], capture_output=True)
    subprocess.run(["killall", "AppleCameraAssistant"], capture_output=True)
    try:
        ctx = rs.context()
        if len(ctx.devices) > 0:
            ctx.devices[0].hardware_reset()
            time.sleep(3)
    except Exception:
        pass
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        pipeline.start(cfg)
    except Exception as e:
        print(f"[Recorder] failed: {e}"); return
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(filename, fourcc, 30, (640, 480))
    print(f"[Recorder] recording -> {filename}")
    while not _record_flag.is_set():
        try:
            frames = pipeline.wait_for_frames(5000)
            img = np.asarray(frames.get_color_frame().get_data())
            writer.write(img)
        except Exception:
            continue
    writer.release()
    pipeline.stop()
    print(f"[Recorder] saved {filename}")

def start_recording(filename="image/trajectory_execution.avi"):
    _record_flag.clear()
    t = threading.Thread(target=_record_thread, args=(filename,), daemon=True)
    t.start()
    time.sleep(4)
    return t

def stop_recording(thread):
    _record_flag.set()
    if thread:
        thread.join(timeout=5)


# ── CAMERA HELPERS ────────────────────────────────────────────────────────────
def reset_camera():
    subprocess.run(["killall", "VDCAssistant"],        capture_output=True)
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
        print(f"  start failed: {e}"); return None, None
    align = rs.align(rs.stream.color)
    try:
        for i in range(50):
            frames = pipeline.wait_for_frames(5000)
            frames = align.process(frames)
            depth  = np.asarray(frames.get_depth_frame().get_data())
            valid  = np.sum((depth > 0) & (depth < 65535))
            print(f"  frame {i}: {valid} valid pixels")
            if valid > 800000:
                rgb = cv2.cvtColor(
                    np.asarray(frames.get_color_frame().get_data()),
                    cv2.COLOR_BGRA2RGB)
                pipeline.stop()
                return rgb, depth
        pipeline.stop(); return None, None
    except Exception as e:
        print(f"  capture failed: {e}")
        try: pipeline.stop()
        except Exception: pass
        return None, None

def capture_with_retry(max_attempts=10):
    for attempt in range(1, max_attempts + 1):
        print(f"\nCapture attempt {attempt}...")
        reset_camera(); time.sleep(1)
        rgb, depth = try_capture()
        if rgb is not None:
            return rgb, depth
        print("  failed, retrying..."); time.sleep(1)
    raise RuntimeError("Could not capture")

# ── ROBOT HELPERS ─────────────────────────────────────────────────────────────
def current_joints(robot):
    obs = robot.get_observation()
    q = np.zeros(7)
    q[1:6] = np.deg2rad([
        obs["shoulder_pan.pos"], obs["shoulder_lift.pos"],
        obs["elbow_flex.pos"],   obs["wrist_flex.pos"],
        obs["wrist_roll.pos"],
    ])
    return ik_solver.clamp_joints(q)    # clamp to URDF bounds

def current_ee_m(robot):
    return ik_solver.fk(current_joints(robot))


def move_to_m(robot, target_m, gripper_val, label="",
              approach_dir=None, roll_override=None):
    """
    Move EE to target_m (metres).

    approach_dir  : [3] unit vector for gripper Z-axis direction. None = position only.
    roll_override : float (radians) to set wrist_roll after IK. None = don't override.
    """
    target_m = np.asarray(target_m, float)
    if np.any(target_m < WS_MIN) or np.any(target_m > WS_MAX):
        print(f"  [{label}] OUTSIDE bounds — skipped"); return False

    guess = current_joints(robot)
    joints, err_m = ik_solver.solve(target_m, initial_guess=guess,
                                    approach_dir=approach_dir,
                                    roll_override=roll_override)
    err_cm = err_m * 100
    tip_m  = ik_solver.fk(joints)

    orient_str = ""
    if approach_dir is not None:
        orient_str = f"  approach={np.round(approach_dir,2)}"
    if roll_override is not None:
        orient_str += f"  roll={np.rad2deg(roll_override):.0f}°"

    print(f"  [{label}] target {target_m*100} cm -> tip {tip_m*100} cm  "
          f"err {err_cm:.2f} cm{orient_str}")

    if err_cm > IK_TOLERANCE_CM:
        print(f"  [{label}] IK error > {IK_TOLERANCE_CM} cm — skipped"); return False

    action = ik_solver.joints_to_action(joints)
    action["gripper.pos"] = float(gripper_val)
    robot.send_action(action)
    time.sleep(1.5)
    return True


# ── KEYPOINT TRACKING ─────────────────────────────────────────────────────────
def update_keypoints(initial_kps, movable_mask, ee_at_grasp, ee_now):
    kps = initial_kps.copy()
    if ee_at_grasp is None or ee_now is None:
        return kps
    disp = ee_now - ee_at_grasp
    for i, held in enumerate(movable_mask):
        if held:
            kps[i] = initial_kps[i] + disp
    return kps


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs("image", exist_ok=True)

    # ── perception ────────────────────────────────────────────────────────────
    print("\n=== Step 1: Capture scene ===")
    rgb, depth_raw = capture_with_retry()
    cv2.imwrite("image/traj_scene.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save("image/traj_depth.npy", depth_raw)

    print("\n=== Step 2: Backproject ===")
    points_base_m = build_points_array(depth_raw)

    print("\n=== Step 3: Keypoint proposal ===")
    kp = KeypointProposer(k_per_mask=3, min_mask_pixels=500, workspace_bounds=(WS_MIN, WS_MAX))
    result      = kp.get_keypoints(rgb, points=points_base_m, visualize=True)
    overlay     = result["overlay"]
    overlay_with_axes = draw_robot_axes(overlay, T_bc, K, axis_length_cm=15)
    initial_kps = result["keypoints_3d"]
    K_kps       = len(initial_kps)
    cv2.imwrite("image/traj_overlay.png", 
            cv2.cvtColor(overlay_with_axes, cv2.COLOR_RGB2BGR))
    print(f"Got {K_kps} keypoints")
    if K_kps == 0:
        print("No keypoints — aborting"); return

    # ── constraint generation ─────────────────────────────────────────────────
    print(f"\n=== Step 4: Generate constraints ===")
    vlm = VLM(model="anthropic/claude-opus-4.8")
    cg  = generate_constraints(overlay_with_axes, args.task, num_keypoints=K_kps, keypoints_3d=initial_kps, ik_solver=ik_solver, vlm=vlm)
    print(f"Got {cg['num_stages']} stages: {cg['stage_names']}")
    print(f"Gripper: {cg['stage_gripper_action']}")
    with open("image/traj_constraints.py", "w") as f:
        f.write(cg["code_str"])

    if args.dry_run:
        print("\n--dry-run set, not moving robot"); return

    # ── robot connect ─────────────────────────────────────────────────────────
    print("\n=== Step 5: Robot connect ===")
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_arm"))
    robot.connect(calibrate=False)
    rec_thread = None

    def safe_exit(sig=None, frame=None):
        print("\nInterrupted...")
        stop_recording(rec_thread)
        try: robot.disconnect()
        except Exception: pass
        sys.exit(0)
    signal.signal(signal.SIGINT, safe_exit)

    input("\nPress ENTER to start (clear the area)...")
    rec_thread = start_recording()

    # ── stage execution loop ──────────────────────────────────────────────────
    ee_at_grasp = None

    # orientation from VLM — applies to approach + grasp stages
    elevation_deg = cg.get("approach_elevation_deg", 90)
    ROLL_CORRECTION = 0.582   
    roll_rad = np.deg2rad(cg.get("gripper_roll_deg", 0) * ROLL_CORRECTION)
    print(f"\nGrasp orientation: elevation={elevation_deg}°  roll={np.rad2deg(roll_rad):.0f}°")

    try:
        for s in range(cg["num_stages"]):
            print(f"\n{'='*60}")
            print(f"STAGE {s+1}/{cg['num_stages']}: {cg['stage_names'][s]}")
            print(f"{'='*60}")

            mask         = cg["stage_movable_mask"][s] if s < len(cg["stage_movable_mask"]) else [False]*K_kps
            constraints  = cg["stage_constraints"][s]  if s < len(cg["stage_constraints"])  else []
            gripper_state= cg["stage_gripper_action"][s] if s < len(cg["stage_gripper_action"]) else "open"
            next_mask    = cg["stage_movable_mask"][s+1] if s+1 < len(cg["stage_movable_mask"]) else mask

            is_grasp = any(next_mask) and not any(mask) and ee_at_grasp is None

            # approach = stage right before grasp (same orientation for smooth transition)
            next_next_mask = cg["stage_movable_mask"][s+2] if s+2 < len(cg["stage_movable_mask"]) else next_mask
            is_approach = (not any(mask) and not any(next_mask)
                           and any(next_next_mask) and ee_at_grasp is None)

            use_orientation = is_grasp or is_approach

            ee_now  = current_ee_m(robot)
            kps_now = update_keypoints(initial_kps, mask, ee_at_grasp, ee_now)

            stage_type = "GRASP" if is_grasp else "APPROACH" if is_approach else "transit/place"
            print(f"  solving subgoal ({len(constraints)} constraints, {stage_type})...")

            target_m, cost = solve_subgoal(
                constraints, kps_now,
                workspace_min=WS_MIN, workspace_max=WS_MAX,
                initial_ee=ee_now,
                ik_solver=ik_solver,
                is_grasp_stage=is_grasp,
                movable_mask=mask,
            )
            print(f"  subgoal: {target_m*100} cm  cost {cost:.4f}")

            # compute approach direction from elevation + target XY
            approach = approach_from_elevation(target_m, elevation_deg) if use_orientation else None
            roll     = roll_rad if use_orientation else None

            gripper_val = GRIPPER_CLOSED_VAL if gripper_state == "closed" else GRIPPER_OPEN_VAL
            move_to_m(robot, target_m, gripper_val,
                      label=cg["stage_names"][s],
                      approach_dir=approach,
                      roll_override=roll)

            if is_grasp:
                ee_at_grasp = current_ee_m(robot)
                print(f"  locked grasp EE: {ee_at_grasp*100} cm")

        print("\n=== Done! ===")
    finally:
        stop_recording(rec_thread)
        try:
            robot.disconnect()
        except Exception as e:
            print(f"Disconnect warning: {e}")


if __name__ == "__main__":
    main()