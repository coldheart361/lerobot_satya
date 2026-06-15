import time
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem5B140297181"
robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_arm"))
robot.connect(calibrate=False)

robot.disconnect()