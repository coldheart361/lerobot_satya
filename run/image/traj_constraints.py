
# Keypoint 0: on the masking/big tape roll (front edge)
# Keypoint 1: on the book cover ("SPIRIT" book)
# Keypoint 2: on a pink sticky note / table surface (front)
# Keypoint 3: on the robot arm (upper part)
# Keypoint 4: on the robot arm base/foot
# Keypoint 5: on the table surface (left area)

num_stages = 4

def stage1_subgoal_constraint1(end_effector, keypoints):
    """EE 10cm above the tape roll keypoint."""
    target = keypoints[0] + np.array([0, 0, 0.10])
    return np.linalg.norm(end_effector - target)

def stage2_subgoal_constraint1(end_effector, keypoints):
    """EE at the tape roll keypoint (grasp moment)."""
    return np.linalg.norm(end_effector - keypoints[0])

def stage3_subgoal_constraint1(end_effector, keypoints):
    """Held tape (keypoint 0) positioned above the book (keypoint 1)."""
    return np.linalg.norm(keypoints[0][:2] - keypoints[1][:2]) + abs(keypoints[0][2] - keypoints[1][2] - 0.15)

def stage4_subgoal_constraint1(end_effector, keypoints):
    """Held tape placed just above the book surface."""
    return np.linalg.norm(keypoints[0][:2] - keypoints[1][:2]) + abs(keypoints[0][2] - keypoints[1][2] - 0.04)

STAGE_CONSTRAINTS       = [[stage1_subgoal_constraint1], [stage2_subgoal_constraint1],
                           [stage3_subgoal_constraint1], [stage4_subgoal_constraint1]]
STAGE_PATH_CONSTRAINTS  = [[], [], [], []]
STAGE_NAMES             = ["approach", "grasp", "transit", "place"]
STAGE_MOVABLE_MASK      = [
    [False] * 6,                                 # stage 1: nothing grasped
    [False] * 6,                                 # stage 2: grasp moment
    [i == 0 for i in range(6)],                  # stage 3: tape held
    [i == 0 for i in range(6)],                  # stage 4: still held
]
STAGE_GRIPPER_ACTION    = ["open", "closed", "closed", "open"]
