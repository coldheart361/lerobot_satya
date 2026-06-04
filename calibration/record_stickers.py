"""
record_stickers.py
===================
Record stickers for extrinsic calibration.

FLOW PER WAYPOINT:
  1. Script releases arm torque -> arm becomes free to move by hand
  2. You physically move the arm to the desired position
  3. Press ENTER
  4. Torque re-enables, arm locks in place
  5. Script reads and records the position
  6. Repeat for next waypoint
"""

import json
from os import name
import signal
import sys
import numpy as np
from ikpy.chain import Chain
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ── CONFIG ────────────────────────────────────────────────────────────────────
PORT = "/dev/tty.usbmodem5B140297181"
URDF_PATH    = "../so101.urdf"
OUTPUT_FILE  = "stickers.json"

TCP_OFFSET_M = np.array([0.004016, -0.004152, 0.015589, 1.0])

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]

STICKERS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]


# ── LOAD CHAIN ────────────────────────────────────────────────────────────────
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


def release_arm(robot):
    try:
        robot.bus.disable_torque()
        print("  [ARM RELEASED — move it freely]")
    except Exception as e:
        print(f"  [WARNING: could not release torque: {e}]")

def hold_arm(robot):
    try:
        robot.bus.enable_torque()
        print("  [ARM LOCKED — reading position...]")
    except Exception as e:
        print(f"  [WARNING: could not re-enable torque: {e}]")

def get_tip_cm(robot):
    """Read current gripper tip position in base_link frame (cm)."""
    obs = robot.get_observation()
    q = np.zeros(7)
    q[1:6] = np.deg2rad([
        obs["shoulder_pan.pos"],
        obs["shoulder_lift.pos"],
        obs["elbow_flex.pos"],
        obs["wrist_flex.pos"],
        obs["wrist_roll.pos"],
    ])
    fk    = chain.forward_kinematics(q)
    tip_m = fk @ TCP_OFFSET_M
    return tip_m[:3] * 100.0


# ── SAFE EXIT ─────────────────────────────────────────────────────────────────
robot_global = None

def safe_exit(sig=None, frame=None):
    print("\n\nInterrupted — disconnecting robot safely...")
    if robot_global is not None:
        try:
            robot_global.disconnect()
            print("Robot disconnected. Arm should be free.")
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, safe_exit)


# ── MAIN ──────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  STICKER RECORDER — stapler → tissue box demo")
print("=" * 55)

config = SO101FollowerConfig(port=PORT, id="my_arm")
robot  = SO101Follower(config)
robot.connect(calibrate=False)
robot_global = robot
print(f"Connected on {PORT}!\n")
print("TIP: The arm will go LIMP for each waypoint so you can")
print("     move it by hand. Press ENTER to lock and record.\n")

STICKERS_POINTS = {}

for number in STICKERS:
    print(f"\n{'─'*45}")
    print(f"  → {number}")
    print(f"{'─'*45}")

    release_arm(robot)
    input("  Move arm into position, then press ENTER...")
    hold_arm(robot)

    tip_cm = get_tip_cm(robot)
    STICKERS_POINTS[number] = {
        "tip_cm":  tip_cm.tolist(),
    }
    print(f"  ✓ Recorded: ({tip_cm[0]:.1f}, {tip_cm[1]:.1f}, {tip_cm[2]:.1f}) cm")

print("\n" + "=" * 55)
print("All 10 stickers recorded!")
print("=" * 55)
print("\nSummary:")

with open(OUTPUT_FILE, "w") as f:
    json.dump(STICKERS_POINTS, f, indent=2)

print(f"\nSaved → {OUTPUT_FILE}")
print("Run:  python execute_pickup.py --dry-run   (to check)")
print("Then: python execute_pickup.py             (to run)")

robot.disconnect()
robot_global = None