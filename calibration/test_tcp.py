# test_tcp.py — run with robot in a known pose
import numpy as np
from ikpy.chain import Chain
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem5B140297181"
TCP_OFFSET_M = np.array([0.004016, -0.004152, 0.015589, 1.0])

chain = Chain.from_urdf_file("../so101.urdf",
    base_elements=["base_link","shoulder_pan","shoulder_link",
                   "shoulder_lift","upper_arm_link","elbow_flex",
                   "lower_arm_link","wrist_flex","wrist_link",
                   "wrist_roll","gripper_link","gripper_frame_joint",
                   "gripper_frame_link"],
    base_element_type="link",
    active_links_mask=[False,True,True,True,True,True,False])

robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_arm"))
robot.connect(calibrate=False)

obs = robot.get_observation()
q = np.zeros(7)
q[1:6] = np.deg2rad([obs["shoulder_pan.pos"], obs["shoulder_lift.pos"],
                     obs["elbow_flex.pos"], obs["wrist_flex.pos"],
                     obs["wrist_roll.pos"]])

fk  = chain.forward_kinematics(q)
tip = (fk @ TCP_OFFSET_M)[:3] * 100
print(f"FK says tip is at: ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}) cm")

robot.disconnect()