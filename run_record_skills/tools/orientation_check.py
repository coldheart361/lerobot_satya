import numpy as np
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from ik_solver import IKSolver

PORT = "/dev/tty.usbmodem5B140297181"
ik = IKSolver("../so101.urdf", np.array([0.004016, -0.004152, 0.015589, 1.0]))

robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_arm"))
robot.connect(calibrate=False)

obs = robot.get_observation()
print("Current joint readings:", {k: round(v,1) for k,v in obs.items() if k.endswith(".pos")})

q = np.zeros(7)
q[1:6] = np.deg2rad([
    obs["shoulder_pan.pos"],
    obs["shoulder_lift.pos"],
    obs["elbow_flex.pos"],
    obs["wrist_flex.pos"],
    obs["wrist_roll.pos"],
])
q = ik.clamp_joints(q)

fk = ik.chain.forward_kinematics(q)
print("\nFK rotation matrix:")
print(fk[:3, :3].round(3))
print("\nX axis:", fk[:3, 0].round(3))
print("Y axis:", fk[:3, 1].round(3))
print("Z axis:", fk[:3, 2].round(3))

robot.disconnect()