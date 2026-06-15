"""
click.py
===============
Click a point in the camera image, robot moves to that 3D world position.

Pipeline per click:
  pixel (u,v) -> depth at that pixel -> p_camera (backproject with K)
              -> p_base = T_base_camera @ p_camera
              -> add 1cm hover -> bounds check -> IK -> send to robot

Requires:
  camera_matrix.npy
  depth_scale.npy
  T_base_camera.npy
  scene.png        (the captured RGB frame)
  depth.npy        (the aligned depth frame from same capture)
  ../so101.urdf
"""

import signal
import sys
import time
import numpy as np
import cv2
from ikpy.chain import Chain
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ── CONFIG ────────────────────────────────────────────────────────────────────
PORT      = "/dev/tty.usbmodem5B140297181"
URDF_PATH = "../so101.urdf"

HOVER_CM  = 5.0     # how many cm above the clicked point to actually go

# workspace bounds in robot base frame (cm)
BOUNDS_MIN = np.array([  5.0, -35.0,  -5.0])
BOUNDS_MAX = np.array([ 50.0,  35.0,  40.0])

TCP_OFFSET_M = np.array([0.004016, -0.004152, 0.015589, 1.0])


# ── LOAD CALIBRATION ──────────────────────────────────────────────────────────
K           = np.load("intrinsics/camera_matrix.npy")
depth_scale = np.load("intrinsics/depth_scale.npy")[0]
T_bc        = np.load("T_base_camera.npy")
depth_raw   = np.load("image/depth.npy")
img         = cv2.imread("image/scene.png")

print(f"Loaded calibration:")
print(f"  K shape: {K.shape}")
print(f"  depth scale: {depth_scale}")
print(f"  T_base_camera shape: {T_bc.shape}")
print(f"  depth image: {depth_raw.shape}")
print(f"  color image: {img.shape}")


# ── LOAD ROBOT ────────────────────────────────────────────────────────────────
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

config = SO101FollowerConfig(port=PORT, id="my_arm")
robot  = SO101Follower(config)
robot.connect(calibrate=False)
print(f"\nConnected on {PORT}")


# ── SAFE EXIT ─────────────────────────────────────────────────────────────────
def safe_exit(sig=None, frame=None):
    print("\nInterrupted — disconnecting safely...")
    try:
        robot.disconnect()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, safe_exit)


# ── DEPTH READING ─────────────────────────────────────────────────────────────
def depth_cm_at(u, v, patch=7):
    """Median depth (cm) in patch around (u,v), ignoring invalid pixels."""
    H, W = depth_raw.shape
    r = patch // 2
    region = depth_raw[max(0, v-r):min(H, v+r+1),
                       max(0, u-r):min(W, u+r+1)]
    valid = region[(region > 0) & (region < 65535)]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) * depth_scale * 100  # cm


# ── PIXEL -> ROBOT BASE FRAME ────────────────────────────────────────────────
def pixel_to_base(u, v):
    """Returns (p_base_cm, depth_cm) or (None, None) if no depth."""
    d_cm = depth_cm_at(u, v)
    if d_cm is None:
        return None, None
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    z = d_cm
    p_cam = np.array([(u - cx) * z / fx,
                      (v - cy) * z / fy,
                      z, 1.0])
    p_base = (T_bc @ p_cam)[:3]
    return p_base, d_cm


# ── ROBOT MOTION ──────────────────────────────────────────────────────────────
def move_to(target_cm):
    """Solve IK and send the robot to target_cm position."""
    target_m = np.array(target_cm) / 100.0

    # IK gives joints. Pass target as np array
    joints = chain.inverse_kinematics(target_m)

    # Compute where the chain says the tip will be
    fk_pose = chain.forward_kinematics(joints)
    tip_m   = fk_pose @ TCP_OFFSET_M
    tip_cm  = tip_m[:3] * 100
    err     = np.linalg.norm(tip_cm - np.array(target_cm))
    print(f"  IK gives tip at ({tip_cm[0]:.1f}, {tip_cm[1]:.1f}, {tip_cm[2]:.1f}) cm, err {err:.2f} cm")

    if err > 5.0:
        print(f"  IK error too high — refusing to move")
        return False

    # Send joint angles (degrees) to robot
    angles_deg = np.rad2deg(joints[1:6])
    action = {
        "shoulder_pan.pos":  float(angles_deg[0]),
        "shoulder_lift.pos": float(angles_deg[1]),
        "elbow_flex.pos":    float(angles_deg[2]),
        "wrist_flex.pos":    float(angles_deg[3]),
        "wrist_roll.pos":    float(angles_deg[4]),
        "gripper.pos":       0.0,   # open gripper
    }
    robot.send_action(action)
    return True


# ── CLICK CALLBACK ────────────────────────────────────────────────────────────
def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    print(f"\nClicked pixel ({x}, {y})")

    p_base, d_cm = pixel_to_base(x, y)
    if p_base is None:
        print("  NO DEPTH at this pixel — ignored")
        return

    print(f"  depth at pixel: {d_cm:.1f} cm")
    print(f"  p_base (clicked point): ({p_base[0]:.1f}, {p_base[1]:.1f}, {p_base[2]:.1f}) cm")

    target = p_base.copy()
    target[2] += HOVER_CM
    print(f"  target with {HOVER_CM}cm hover: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) cm")

    # bounds check
    if np.any(target < BOUNDS_MIN) or np.any(target > BOUNDS_MAX):
        print(f"  OUTSIDE WORKSPACE BOUNDS — refusing to move")
        print(f"    min: {BOUNDS_MIN}")
        print(f"    max: {BOUNDS_MAX}")
        return

    # draw marker on image
    cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
    cv2.putText(img, f"{d_cm:.0f}cm", (x+8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("click to move robot", img)

    print("  moving...")
    move_to(target)
    print("  done")


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  CLICK TO MOVE — hover {HOVER_CM}cm above clicked point")
print(f"  Workspace: X[{BOUNDS_MIN[0]}, {BOUNDS_MAX[0]}] "
      f"Y[{BOUNDS_MIN[1]}, {BOUNDS_MAX[1]}] "
      f"Z[{BOUNDS_MIN[2]}, {BOUNDS_MAX[2]}] cm")
print("=" * 55)
print("Click in the image to send the robot. Press Q to quit.")

cv2.imshow("click to move robot", img)
cv2.setMouseCallback("click to move robot", on_click)

while True:
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
robot.disconnect()
print("Disconnected")