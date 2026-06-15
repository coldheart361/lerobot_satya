"""
ik_solver.py
============
Wraps ikpy for the SO-101. Position IK with OPTIONAL orientation control.

5-DOF orientation model:
  - approach_dir : direction the gripper Z-axis should point.
                   3 pos + 2 direction = 5 constraints = exactly 5 DOF.
  - roll_override: after IK, directly set wrist_roll to spin the gripper.
                   Doesn't move the tip since it sits near the roll axis.

Calling conventions:
  solve(xyz)                                   position only
  solve(xyz, approach_dir=[0,0,-1])            top-down
  solve(xyz, approach_dir=..., roll_override=) approach + finger rotation
"""

import numpy as np
from ikpy.chain import Chain
from .custom_ik import solve_lm


class IKSolver:
    def __init__(self, urdf_path, tcp_offset_m, base_elements=None, active_mask=None):
        if base_elements is None:
            base_elements = ["base_link", "shoulder_pan", "shoulder_link",
                             "shoulder_lift", "upper_arm_link",
                             "elbow_flex", "lower_arm_link",
                             "wrist_flex", "wrist_link",
                             "wrist_roll", "gripper_link",
                             "gripper_frame_joint", "gripper_frame_link"]
        if active_mask is None:
            active_mask = [False, True, True, True, True, True, False]
        self.chain = Chain.from_urdf_file(
            urdf_path, base_elements=base_elements,
            base_element_type="link", active_links_mask=active_mask,
        )
        self.tcp_offset    = np.asarray(tcp_offset_m, float)
        self.wrist_roll_idx = 5   # index 5 = wrist_roll (last True in active_mask)

    def fk(self, joints):
        return (self.chain.forward_kinematics(joints) @ self.tcp_offset)[:3]

    def clamp_joints(self, q):
        q = np.array(q, float)
        for i, link in enumerate(self.chain.links):
            b = getattr(link, "bounds", None)
            if b is not None and b[0] is not None and b[1] is not None:
                q[i] = np.clip(q[i], b[0], b[1])
        return q

    def solve(self, target_m, initial_guess=None, approach_dir=None, roll_override=None):
        return solve_lm(self.chain, np.asarray(target_m), self.tcp_offset,
                        approach_dir=approach_dir, roll_override=roll_override,
                        wrist_roll_idx=self.wrist_roll_idx)

    def reachability_cost(self, target_m, initial_guess=None, approach_dir=None):
        _, err = self.solve(target_m, initial_guess, approach_dir)
        return err

    def joints_to_action(self, joints):
        deg = np.rad2deg(joints[1:6])
        return {
            "shoulder_pan.pos":  float(deg[0]),
            "shoulder_lift.pos": float(deg[1]),
            "elbow_flex.pos":    float(deg[2]),
            "wrist_flex.pos":    float(deg[3]),
            "wrist_roll.pos":    float(deg[4]),
        }