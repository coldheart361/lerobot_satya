"""
click_orient_go.py — interactive orientation tuning
=====================================================
Click a point, then use keyboard to tune elevation and roll live.

Controls:
  Click image   → set target point
  W / S         → elevation +5° / -5°  (W = more top-down, S = more horizontal)
  A / D         → roll -10° / +10°     (spin gripper)
  SPACE         → move robot to current target + orientation
  R             → re-capture image
  Q             → quit
"""

import os, sys, time, argparse, subprocess, signal
import numpy as np
import cv2
import pyrealsense2 as rs
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ik_solver import IKSolver

# ── CONFIG ────────────────────────────────────────────────────────────────────
PORT         = "/dev/tty.usbmodem5B140297181"
URDF_PATH    = "../so101.urdf"
TCP_OFFSET_M = np.array([0.004016, -0.004152, 0.015589, 1.0])
WS_MIN       = np.array([ 0.05, -0.25, -0.05])
WS_MAX       = np.array([ 0.45,  0.20,  0.30])
IK_TOL_CM    = 6.0

K           = np.load("../../calibration/intrinsics/camera_matrix.npy")
depth_scale = np.load("../../calibration/intrinsics/depth_scale.npy")[0]
T_bc        = np.load("../../calibration/T_base_camera.npy")

print("Loading IK solver...")
ik = IKSolver(urdf_path=URDF_PATH, tcp_offset_m=TCP_OFFSET_M)

ROLL_MIN = -157   # np.rad2deg(-2.74)
ROLL_MAX =  163   # np.rad2deg(2.84)

# ── CAMERA ────────────────────────────────────────────────────────────────────
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

def capture():
    for attempt in range(10):
        print(f"Capture attempt {attempt+1}...")
        reset_camera(); time.sleep(1)
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgra8, 6)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,   6)
        try:
            pipeline.start(cfg)
        except Exception as e:
            print(f"  failed: {e}"); continue
        align = rs.align(rs.stream.color)
        try:
            for _ in range(50):
                frames = pipeline.wait_for_frames(5000)
                frames = align.process(frames)
                depth  = np.asarray(frames.get_depth_frame().get_data())
                if np.sum((depth > 0) & (depth < 65535)) > 800000:
                    bgr = cv2.cvtColor(
                        np.asarray(frames.get_color_frame().get_data()),
                        cv2.COLOR_BGRA2BGR)
                    pipeline.stop()
                    print(f"  captured OK")
                    return bgr, depth
            pipeline.stop()
        except Exception as e:
            print(f"  failed: {e}")
            try: pipeline.stop()
            except Exception: pass
    raise RuntimeError("Could not capture")


# ── PIXEL → WORLD ─────────────────────────────────────────────────────────────
def depth_at(depth_raw, u, v, patch=7):
    r = patch // 2
    H, W = depth_raw.shape
    region = depth_raw[max(0,v-r):v+r+1, max(0,u-r):u+r+1]
    valid  = region[(region > 0) & (region < 65535)]
    return float(np.median(valid)) * depth_scale if len(valid) >= 4 else None

def pixel_to_world(u, v, depth_raw):
    z_m = depth_at(depth_raw, u, v)
    if z_m is None: return None
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    z_cm = z_m * 100
    p_cam = np.array([(u-cx)*z_cm/fx, (v-cy)*z_cm/fy, z_cm, 1.0])
    return (T_bc @ p_cam)[:3] / 100.0


# ── ORIENTATION ───────────────────────────────────────────────────────────────
def approach_from_elevation(target_m, elevation_deg):
    el = np.deg2rad(np.clip(elevation_deg, 0, 90))
    h  = np.array([target_m[0], target_m[1], 0.0])
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-6:
        return np.array([0.0, 0.0, -1.0])
    h_unit = h / h_norm
    d = np.array([np.cos(el)*h_unit[0], np.cos(el)*h_unit[1], -np.sin(el)])
    return d / np.linalg.norm(d)


# ── ROBOT ─────────────────────────────────────────────────────────────────────
def current_joints(robot):
    obs = robot.get_observation()
    q = np.zeros(7)
    q[1:6] = np.deg2rad([obs["shoulder_pan.pos"], obs["shoulder_lift.pos"],
                         obs["elbow_flex.pos"], obs["wrist_flex.pos"],
                         obs["wrist_roll.pos"]])
    return ik.clamp_joints(q)

def move_to(robot, target_m, elevation_deg, roll_deg, hover_m):
    t = target_m.copy()
    t[2] += hover_m

    if np.any(t < WS_MIN) or np.any(t > WS_MAX):
        print(f"  OUTSIDE bounds"); return False

    approach = approach_from_elevation(t, elevation_deg)
    roll_rad = np.deg2rad(roll_deg)
    guess    = current_joints(robot)

    joints, err_m = ik.solve(t, initial_guess=guess,
                              approach_dir=approach, roll_override=roll_rad)
    if err_m * 100 > IK_TOL_CM:
        print(f"  orientation IK failed ({err_m*100:.1f}cm) — falling back")
        joints, err_m = ik.solve(t, initial_guess=guess, roll_override=roll_rad)

    err_cm = err_m * 100
    tip    = ik.fk(joints) * 100
    deg    = np.rad2deg(joints[1:6])
    print(f"  target {t*100}  tip {np.round(tip,1)}  err {err_cm:.2f}cm")
    print(f"  joints: pan={deg[0]:.1f} lift={deg[1]:.1f} "
          f"elbow={deg[2]:.1f} wflex={deg[3]:.1f} wroll={deg[4]:.1f}")

    if err_cm > IK_TOL_CM:
        print(f"  IK still > {IK_TOL_CM}cm — skipped"); return False

    action = ik.joints_to_action(joints)
    action["gripper.pos"] = 50.0
    robot.send_action(action)
    time.sleep(1.5)
    return True


# ── OVERLAY HUD ───────────────────────────────────────────────────────────────
def draw_hud(base_img, target_px, elevation, roll, world_cm, status):
    img = base_img.copy()
    if target_px is not None:
        cv2.circle(img, target_px, 8, (0, 0, 255), -1)
        cv2.circle(img, target_px, 8, (255, 255, 255), 2)

    # HUD box
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h-120), (520, h), (0,0,0), -1)

    lines = [
        f"W/S: elevation {elevation:.0f}deg  |  A/D: roll {roll:.0f}deg  |  SPACE: move  |  Q: quit",
        f"approach_elevation={elevation:.0f}  gripper_roll={roll:.0f}",
        f"target: {world_cm}",
        f"status: {status}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, h-100+i*24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
    return img


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hover", type=float, default=0.03)
    args = ap.parse_args()

    bgr, depth_raw = capture()
    base_img = bgr.copy()

    robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_arm"))
    robot.connect(calibrate=False)

    def safe_exit(sig=None, frame=None):
        try: robot.disconnect()
        except Exception: pass
        cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, safe_exit)

    # state
    target_world = [None]   # [3] metres
    target_px    = [None]   # (u, v) pixels
    elevation    = [90.0]
    roll         = [0.0]
    status       = ["click a point to set target"]
    world_str    = ["—"]

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        p = pixel_to_world(x, y, depth_raw)
        if p is None:
            status[0] = f"NO DEPTH at ({x},{y}) — try another spot"
            return
        d_cm = depth_at(depth_raw, x, y) * 100
        target_world[0] = p
        target_px[0]    = (x, y)
        world_str[0]    = f"({p[0]*100:.1f}, {p[1]*100:.1f}, {p[2]*100:.1f}) cm  depth={d_cm:.1f}cm"
        status[0]       = "target set — SPACE to move, W/S/A/D to adjust"
        print(f"\nTarget: {world_str[0]}")

    cv2.namedWindow("click orient go")
    cv2.setMouseCallback("click orient go", on_click)

    print("\nControls:")
    print("  Click  → set target")
    print("  W/S    → elevation +5/-5 deg")
    print("  A/D    → roll -10/+10 deg")
    print("  SPACE  → move robot")
    print("  R      → re-capture")
    print("  Q      → quit\n")

    while True:
        frame = draw_hud(base_img, target_px[0],
                         elevation[0], roll[0], world_str[0], status[0])
        cv2.imshow("click orient go", frame)
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('w'):
            elevation[0] = min(90, elevation[0] + 5)
            status[0] = f"elevation → {elevation[0]:.0f}°"

        elif key == ord('s'):
            elevation[0] = max(0, elevation[0] - 5)
            status[0] = f"elevation → {elevation[0]:.0f}°"

        elif key == ord('d'):
            roll[0] = min(ROLL_MAX, roll[0] + 20)
            status[0] = f"roll → {roll[0]:.0f}°"

        elif key == ord('a'):
            roll[0] = max(ROLL_MIN, roll[0] - 20)
            status[0] = f"roll → {roll[0]:.0f}°"

        elif key == ord('r'):
            print("\nRe-capturing...")
            bgr, depth_raw = capture()
            base_img = bgr.copy()
            target_world[0] = None
            target_px[0]    = None
            world_str[0]    = "—"
            status[0]       = "re-captured — click a point"

        elif key == ord(' '):
            if target_world[0] is None:
                status[0] = "click a point first"
                continue
            print(f"\nMoving: elevation={elevation[0]:.0f}°  roll={roll[0]:.0f}°")
            ok = move_to(robot, target_world[0].copy(),
                         elevation[0], roll[0], args.hover)
            status[0] = "moved OK" if ok else "move failed — see terminal"

    cv2.destroyAllWindows()
    robot.disconnect()
    print("Done")

if __name__ == "__main__":
    main()