"""
record_waypoints.py
===================
Record waypoints for the stapler -> tissue box demo.

FLOW PER WAYPOINT:
  1. Script releases arm torque -> arm becomes free to move by hand
  2. You physically move the arm to the desired position
  3. Press ENTER
  4. Torque re-enables, arm locks in place
  5. Script reads and records the position
  6. Repeat for next waypoint
"""

import json
import signal
import sys
import numpy as np
from ikpy.chain import Chain
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ── CONFIG ────────────────────────────────────────────────────────────────────
PORT = "/dev/tty.usbmodem5B140297181"
URDF_PATH    = "../so101.urdf"
OUTPUT_FILE  = "waypoints.json"

TCP_OFFSET_M = np.array([0.004016, -0.004152, 0.015589, 1.0])

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]

WAYPOINT_NAMES = [
    "above_stapler",
    "grasp_stapler",
    "lift",
    "above_tissue",
    "place_tissue",
]

GRIPPER_OPEN   = 70.0
GRIPPER_CLOSED = 0.0

GRIPPER_AT = {
    "above_stapler": GRIPPER_OPEN,
    "grasp_stapler": GRIPPER_OPEN,
    "lift":          GRIPPER_CLOSED,
    "above_tissue":  GRIPPER_CLOSED,
    "place_tissue":  GRIPPER_OPEN,
}

INSTRUCTIONS = {
    "above_stapler": "Hover ABOVE the stapler (gripper open, ~5cm above)",
    "grasp_stapler": "Position gripper AT the stapler grasp point",
    "lift":          "Lift the arm UP to a safe carry height",
    "above_tissue":  "Move OVER the tissue box (still holding stapler)",
    "place_tissue":  "Lower DOWN to place the stapler ON the tissue box",
}

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
print("  WAYPOINT RECORDER — stapler → tissue box demo")
print("=" * 55)

config = SO101FollowerConfig(port=PORT, id="my_arm")
robot  = SO101Follower(config)
robot.connect(calibrate=False)
robot_global = robot
print(f"Connected on {PORT}!\n")
print("TIP: The arm will go LIMP for each waypoint so you can")
print("     move it by hand. Press ENTER to lock and record.\n")

waypoints = {}

for name in WAYPOINT_NAMES:
    print(f"\n{'─'*45}")
    print(f"  {name.upper()}")
    print(f"  → {INSTRUCTIONS[name]}")
    print(f"{'─'*45}")

    release_arm(robot)
    input("  Move arm into position, then press ENTER...")
    hold_arm(robot)

    tip_cm = get_tip_cm(robot)
    waypoints[name] = {
        "tip_cm":  tip_cm.tolist(),
        "gripper": GRIPPER_AT[name],
    }
    print(f"  ✓ Recorded: ({tip_cm[0]:.1f}, {tip_cm[1]:.1f}, {tip_cm[2]:.1f}) cm")

print("\n" + "=" * 55)
print("All 5 waypoints recorded!")
print("=" * 55)
print("\nSummary:")
for n, wp in waypoints.items():
    t = wp["tip_cm"]
    g = "open" if wp["gripper"] == GRIPPER_OPEN else "closed"
    print(f"  {n:<20} ({t[0]:6.1f}, {t[1]:6.1f}, {t[2]:6.1f}) cm  gripper={g}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(waypoints, f, indent=2)

print(f"\nSaved → {OUTPUT_FILE}")
print("Run:  python execute_pickup.py --dry-run   (to check)")
print("Then: python execute_pickup.py             (to run)")

robot.disconnect()
robot_global = None