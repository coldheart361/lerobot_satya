"""
execute_pickup.py
=================
Execute the stapler -> tissue box pick-and-place using recorded waypoints.

Run AFTER record_waypoints.py has saved waypoints.json.

Usage:
    python execute_pickup.py
    python execute_pickup.py --dry-run    # print plan without moving
"""

import json
import time
import argparse
import numpy as np
from ikpy.chain import Chain
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ── CONFIG ────────────────────────────────────────────────────────────────────
PORT = "/dev/tty.usbmodem5B140297181"
URDF_PATH      = "../so101.urdf"
WAYPOINTS_FILE = "waypoints.json"

MOVE_STEPS = 30      # interpolation steps between waypoints (higher = smoother)
MOVE_DELAY = 0.06    # seconds between steps
GRIPPER_PAUSE = 1.0  # seconds to wait after opening/closing gripper

GRIPPER_OPEN   = 70.0
GRIPPER_CLOSED = 0.0

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


def move_to_xyz(robot, x_cm, y_cm, z_cm, gripper=GRIPPER_OPEN,
                steps=MOVE_STEPS, delay=MOVE_DELAY, dry_run=False):
    """Move gripper tip to (x, y, z) in base_link frame (cm)."""
    target_m = np.array([x_cm, y_cm, z_cm]) / 100.0
    ik_angles = chain.inverse_kinematics(target_m)
    fk = chain.forward_kinematics(ik_angles)
    achieved_cm = fk[:3, 3] * 100.0
    error = np.linalg.norm(np.array([x_cm, y_cm, z_cm]) - achieved_cm)

    print(f"    target:   ({x_cm:.1f}, {y_cm:.1f}, {z_cm:.1f}) cm")
    print(f"    IK gives: ({achieved_cm[0]:.1f}, {achieved_cm[1]:.1f}, {achieved_cm[2]:.1f}) cm  |  error {error:.2f} cm")

    if error > 8.0:
        print("    !! IK error too large — skipping this move!")
        return False

    if dry_run:
        print("    [dry-run: not moving]")
        return True

    ik_deg = np.degrees(ik_angles)
    obs    = robot.get_observation()
    start  = {k: obs[k] for k in obs if k in [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
    ]}

    target_action = {
        "shoulder_pan.pos":  ik_deg[1],
        "shoulder_lift.pos": ik_deg[2],
        "elbow_flex.pos":    ik_deg[3],
        "wrist_flex.pos":    ik_deg[4],
        "wrist_roll.pos":    ik_deg[5],
        "gripper.pos":       gripper,
    }

    for i in range(steps + 1):
        t      = i / steps
        action = {k: start.get(k, target_action[k]) +
                     (target_action[k] - start.get(k, target_action[k])) * t
                  for k in target_action}
        robot.send_action(action)
        time.sleep(delay)

    return True


def set_gripper(robot, value, dry_run=False):
    """Open or close the gripper and wait."""
    state = "OPENING" if value == GRIPPER_OPEN else "CLOSING"
    print(f"    {state} gripper...")
    if not dry_run:
        obs = robot.get_observation()
        action = {k: obs[k] for k in obs if k in [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos",
        ]}
        action["gripper.pos"] = value
        for _ in range(20):
            robot.send_action(action)
            time.sleep(0.05)
        time.sleep(GRIPPER_PAUSE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan without moving the robot")
    args = ap.parse_args()

    with open(WAYPOINTS_FILE) as f:
        wps = json.load(f)

    print("=" * 55)
    print("  EXECUTE: stapler → tissue box demo")
    print("=" * 55)
    print("\nWaypoints loaded:")
    for name, wp in wps.items():
        t = wp["tip_cm"]
        print(f"  {name:<20} ({t[0]:6.1f}, {t[1]:6.1f}, {t[2]:6.1f}) cm")

    if args.dry_run:
        print("\n[DRY RUN — no robot movement]\n")
        robot = None
    else:
        print(f"\nConnecting to robot on {PORT}...")
        config = SO101FollowerConfig(port=PORT, id="my_arm")
        robot  = SO101Follower(config)
        robot.connect(calibrate=False)
        print("Connected!\n")
        input("Press ENTER to start the sequence (make sure area is clear)...")
        print()

    # ── SEQUENCE ─────────────────────────────────────────────────────────────

    steps = [
        # (step description,  waypoint_name,  gripper_after)
        ("Move above stapler",      "above_stapler",  None),
        ("Descend to stapler",      "grasp_stapler",  None),
        ("Close gripper (GRAB)",    None,             GRIPPER_CLOSED),
        ("Lift stapler",            "lift",           None),
        ("Move above tissue box",   "above_tissue",   None),
        ("Place on tissue box",     "place_tissue",   None),
        ("Open gripper (RELEASE)",  None,             GRIPPER_OPEN),
        ("Retreat to lift height",  "lift",           None),
    ]

    for i, (description, wp_name, gripper_cmd) in enumerate(steps, 1):
        print(f"\nStep {i}: {description}")

        if gripper_cmd is not None:
            set_gripper(robot, gripper_cmd, dry_run=args.dry_run)
        elif wp_name in wps:
            tip = wps[wp_name]["tip_cm"]
            gripper = wps[wp_name]["gripper"]
            move_to_xyz(robot, tip[0], tip[1], tip[2],
                        gripper=gripper, dry_run=args.dry_run)
        else:
            print(f"    !! Waypoint '{wp_name}' not found in file — skipping")

    print("\n" + "=" * 55)
    print("  Done! Stapler placed on tissue box.")
    print("=" * 55)

    if robot and not args.dry_run:
        robot.disconnect()


if __name__ == "__main__":
    main()