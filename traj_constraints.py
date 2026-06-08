
# Keypoint 0: appears to be on the double-coated tape roll (masking tape)
# Keypoint 1: appears to be on the robot arm (top/head)
# Keypoint 2: appears to be on the table surface (pink marker)
# Keypoint 3: appears to be on the robot arm base/gripper area
# The little box is the HD-10 box on the table (no keypoint, but near keypoint 0).
# We'll grasp the box using its approximate location; since the only graspable
# object keypoint scheme uses keypoint 0 for tape and box has no keypoint,
# we treat the box position relative to keypoint 0. We'll use keypoint 0 area.
# For this task we assume the box is the grasp target — we approximate its
# keypoint as the box. Since no dedicated keypoint exists, we use keypoint 2
# region is table. We will use the box near keypoint 0 region offset.
# To keep it simple and valid, grasp keypoint chosen = keypoint 0 vicinity is tape;
# the box sits to its lower-right. We pick the box by an offset from keypoint 0.

num_stages = 4

def stage1_subgoal_constraint1(end_effector, keypoints):
    """EE 8cm above the little box (approx offset from tape keypoint 0)."""
    box = keypoints[0] + np.array([0.10, -0.10, 0.0])
    target = box + np.array([0, 0, 0.08])
    return np.linalg.norm(end_effector - target)

def stage2_subgoal_constraint1(end_effector, keypoints):
    """EE at the box (grasp moment)."""
    box = keypoints[0] + np.array([0.10, -0.10, 0.0])
    return np.linalg.norm(end_effector - box)

def stage3_subgoal_constraint1(end_effector, keypoints):
    """Held box above the tape (keypoint 0)."""
    box = keypoints[0] + np.array([0.10, -0.10, 0.0])
    return np.linalg.norm(box[:2] - keypoints[0][:2]) + abs(box[2] - keypoints[0][2] - 0.12)

def stage4_subgoal_constraint1(end_effector, keypoints):
    """Place box just on top of the tape."""
    box = keypoints[0] + np.array([0.10, -0.10, 0.0])
    return np.linalg.norm(box[:2] - keypoints[0][:2]) + abs(box[2] - keypoints[0][2] - 0.03)

STAGE_CONSTRAINTS       = [[stage1_subgoal_constraint1], [stage2_subgoal_constraint1],
                           [stage3_subgoal_constraint1], [stage4_subgoal_constraint1]]
STAGE_PATH_CONSTRAINTS  = [[], [], [], []]
STAGE_NAMES             = ["approach", "grasp", "transit", "place"]
STAGE_MOVABLE_MASK      = [
    [False] * 4,
    [False] * 4,
    [i == 0 for i in range(4)],
    [i == 0 for i in range(4)],
]
STAGE_GRIPPER_ACTION    = ["open", "closed", "closed", "open"]
