import os
import sys
import time
import subprocess

import numpy as np
import cv2
import pyrealsense2 as rs

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ik_solver import IKSolver
from core.keypoint_proposal     import KeypointProposer
from core.ik_solver             import IKSolver          
from core.helper import (draw_robot_axes, 
                         check_keypoint_reachability, 
                         build_points_array, 
                         reachability_report_str,
                         build_boundary_overlay)


# ── CONFIG ───────────────────────────────────────────────────────────────────
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
K           = np.load("../../calibration/intrinsics/camera_matrix.npy")
depth_scale = np.load("../../calibration/intrinsics/depth_scale.npy")[0]
T_bc        = np.load("../../calibration/T_base_camera.npy")


# ── IK SOLVER (instantiated once at startup) ──────────────────────────────────
# The IKSolver wraps ikpy and adds orientation-aware solve():
#   - solve(xyz)                       : position only (transit/place stages)
#   - solve(xyz, approach_dir=[0,0,-1]): top-down approach (grasp stages)
# Having it as a module means subgoal_solver can also call it for reachability.
print("Loading IK solver...")
ik_solver = IKSolver(urdf_path=URDF_PATH, tcp_offset_m=TCP_OFFSET_M)


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

def main():
    print("\n=== Step 1: Capture scene ===")
    rgb, depth_raw = capture_with_retry()
    cv2.imwrite("../reachability/traj_scene.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print("\n=== Step 2: Build boundary overlay ===")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # draw robot axes
    bgr_with_axes = draw_robot_axes(bgr, T_bc, K, axis_length_cm=15)
    
    # draw reachability boundary
    boundary = build_boundary_overlay(
        bgr_with_axes, ik_solver, T_bc, K,
        z_levels=(-0.03, 0.00, 0.03, 0.06, 0.09, 0.12),   # ← add more z levels for a richer visualization
        elevations=(90, 0,),                        # ← show both top-down and forward-facing reachability
        n_angles=18,
    )

    cv2.imwrite("../reachability/traj_boundary.png", boundary)
    print("Saved → ../reachability/traj_boundary.png")


if __name__ == "__main__":
    os.makedirs("../reachability", exist_ok=True)
    main()