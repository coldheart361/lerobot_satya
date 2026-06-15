"""
ik_test.py — IK reachability checker
=====================================
Input a 3D coordinate + elevation + roll, see if the arm can reach it.

Usage:
  python ik_test.py
  python ik_test.py --x 25 --y -5 --z 8 --elevation 30 --roll 90
"""

import argparse
import numpy as np
import sys, os

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ik_solver import IKSolver

URDF_PATH    = "../so101.urdf"
TCP_OFFSET_M = np.array([0.004016, -0.004152, 0.015589, 1.0])
WS_MIN       = np.array([ 0.05, -0.25, -0.05])
WS_MAX       = np.array([ 0.45,  0.20,  0.30])

ik = IKSolver(urdf_path=URDF_PATH, tcp_offset_m=TCP_OFFSET_M)


def approach_from_elevation(target_m, elevation_deg):
    el = np.deg2rad(np.clip(elevation_deg, 0, 90))
    h  = np.array([target_m[0], target_m[1], 0.0])
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-6:
        return np.array([0.0, 0.0, -1.0])
    h_unit = h / h_norm
    d = np.array([np.cos(el)*h_unit[0], np.cos(el)*h_unit[1], -np.sin(el)])
    return d / np.linalg.norm(d)


def test_ik(x_cm, y_cm, z_cm, elevation_deg, roll_deg):
    target_m  = np.array([x_cm, y_cm, z_cm]) / 100.0
    approach  = approach_from_elevation(target_m, elevation_deg)
    roll_rad  = np.deg2rad(roll_deg)

    in_ws = np.all(target_m >= WS_MIN) and np.all(target_m <= WS_MAX)

    print(f"\n{'='*55}")
    print(f"  Target     : ({x_cm}, {y_cm}, {z_cm}) cm")
    print(f"  Elevation  : {elevation_deg}°")
    print(f"  Roll       : {roll_deg}°")
    print(f"  approach_dir: {np.round(approach, 3)}")
    print(f"  In workspace: {'YES' if in_ws else 'NO  ← outside bounds!'}")
    print(f"{'='*55}")

    if not in_ws:
        print(f"  WS_MIN: {WS_MIN*100} cm")
        print(f"  WS_MAX: {WS_MAX*100} cm")

    # test 1: orientation-constrained IK
    joints_ori, err_ori = ik.solve(target_m, approach_dir=approach, roll_override=roll_rad)
    tip_ori = ik.fk(joints_ori) * 100
    deg_ori = np.rad2deg(joints_ori[1:6])

    print(f"\n  [With orientation (elevation={elevation_deg}°)]")
    print(f"  IK tip   : {np.round(tip_ori, 2)} cm")
    print(f"  IK error : {err_ori*100:.2f} cm  {'✓ reachable' if err_ori*100 <= 5 else '✗ UNREACHABLE'}")
    print(f"  Joints   : pan={deg_ori[0]:.1f}  lift={deg_ori[1]:.1f}  "
          f"elbow={deg_ori[2]:.1f}  wflex={deg_ori[3]:.1f}  wroll={deg_ori[4]:.1f}")

    # test 2: position-only fallback
    joints_pos, err_pos = ik.solve(target_m, roll_override=roll_rad)
    tip_pos = ik.fk(joints_pos) * 100
    deg_pos = np.rad2deg(joints_pos[1:6])

    print(f"\n  [Position-only fallback]")
    print(f"  IK tip   : {np.round(tip_pos, 2)} cm")
    print(f"  IK error : {err_pos*100:.2f} cm  {'✓ reachable' if err_pos*100 <= 5 else '✗ UNREACHABLE'}")
    print(f"  Joints   : pan={deg_pos[0]:.1f}  lift={deg_pos[1]:.1f}  "
          f"elbow={deg_pos[2]:.1f}  wflex={deg_pos[3]:.1f}  wroll={deg_pos[4]:.1f}")

    # verdict
    print(f"\n  Verdict:")
    if err_ori*100 <= 5:
        print(f"  ✓ Orientation IK works — elevation={elevation_deg}° is achievable here")
    elif err_pos*100 <= 5:
        print(f"  ⚠ Orientation IK fails ({err_ori*100:.1f}cm) but position-only works")
        print(f"    → arm will reach the position but ignore the {elevation_deg}° approach angle")
        print(f"    → try a different elevation (closer to 90° is usually more reliable)")
    else:
        print(f"  ✗ Position unreachable entirely ({err_pos*100:.1f}cm)")
        print(f"    → target is outside the arm's physical reach")


def move_robot(x_cm, y_cm, z_cm, elevation_deg, roll_deg, robot):
    """Solve IK and send to robot. Falls back to position-only if orientation fails."""
    target_m = np.array([x_cm, y_cm, z_cm]) / 100.0
    approach = approach_from_elevation(target_m, elevation_deg)
    roll_rad = np.deg2rad(roll_deg)

    obs = robot.get_observation()
    q = np.zeros(7)
    q[1:6] = np.deg2rad([obs["shoulder_pan.pos"], obs["shoulder_lift.pos"],
                         obs["elbow_flex.pos"], obs["wrist_flex.pos"],
                         obs["wrist_roll.pos"]])
    guess = ik.clamp_joints(q)

    joints, err_m = ik.solve(target_m, initial_guess=guess,
                              approach_dir=approach, roll_override=roll_rad)
    if err_m * 100 > 5.0:
        print(f"  orientation IK failed ({err_m*100:.1f}cm) — falling back to position-only")
        joints, err_m = ik.solve(target_m, initial_guess=guess, roll_override=roll_rad)

    if err_m * 100 > 5.0:
        print(f"  position IK also failed ({err_m*100:.1f}cm) — not moving")
        return

    import time
    action = ik.joints_to_action(joints)
    action["gripper.pos"] = 50.0
    print(f"  sending to robot...")
    robot.send_action(action)
    time.sleep(2.0)

    obs2 = robot.get_observation()
    q2 = np.zeros(7)
    q2[1:6] = np.deg2rad([obs2["shoulder_pan.pos"], obs2["shoulder_lift.pos"],
                           obs2["elbow_flex.pos"], obs2["wrist_flex.pos"],
                           obs2["wrist_roll.pos"]])
    actual = ik.fk(ik.clamp_joints(q2)) * 100
    print(f"  actual tip: {np.round(actual, 1)} cm")
    print(f"  Δ from target: {np.linalg.norm(actual - np.array([x_cm,y_cm,z_cm])):.2f} cm")


def interactive(robot=None):
    print("IK Reachability Tester")
    if robot:
        print("Robot connected — will move after confirmation.")
    print("Enter coordinates in cm, angles in degrees. Ctrl+C to quit.\n")
    while True:
        try:
            x   = float(input("x (cm): "))
            y   = float(input("y (cm): "))
            z   = float(input("z (cm): "))
            el  = float(input("elevation_deg [0-90, default 90]: ") or "90")
            rol = float(input("gripper_roll_deg [default 0]: ") or "0")
            test_ik(x, y, z, el, rol)
            if robot:
                go = input("\n  Move robot there? [y/N]: ").strip().lower()
                if go == 'y':
                    move_robot(x, y, z, el, rol, robot)
            print()
        except (ValueError, EOFError):
            print("Invalid input, try again")
        except KeyboardInterrupt:
            print("\nDone")
            break


def main():
    import signal
    ap = argparse.ArgumentParser()
    ap.add_argument("--x",         type=float)
    ap.add_argument("--y",         type=float)
    ap.add_argument("--z",         type=float)
    ap.add_argument("--elevation", type=float, default=90)
    ap.add_argument("--roll",      type=float, default=0)
    ap.add_argument("--move",      action="store_true",
                    help="connect robot and move (requires sudo -E)")
    args = ap.parse_args()

    robot = None
    if args.move:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        PORT = "/dev/tty.usbmodem5B140297181"
        robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_arm"))
        robot.connect(calibrate=False)
        def safe_exit(sig=None, frame=None):
            try: robot.disconnect()
            except Exception: pass
            sys.exit(0)
        signal.signal(signal.SIGINT, safe_exit)

    try:
        if args.x is not None:
            test_ik(args.x, args.y, args.z, args.elevation, args.roll)
            if robot:
                go = input("\n  Move robot there? [y/N]: ").strip().lower()
                if go == 'y':
                    move_robot(args.x, args.y, args.z, args.elevation, args.roll, robot)
        else:
            interactive(robot)
    finally:
        if robot:
            robot.disconnect()


if __name__ == "__main__":
    main()