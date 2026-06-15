
# Keypoint 0: appears to be on the tape roll (front/side face, center)
# Keypoint 1: appears to be on the robot arm (far right)
# Keypoint 2: appears to be on the robot arm (upper area)
# Keypoint 3: appears to be on the book cover (top left area)
# Keypoint 4: appears to be on the tape roll (right side)
# Keypoint 5: appears to be on the table surface (empty area, lower center)
# Keypoint 6: appears to be on the robot arm (middle area)
# Keypoint 7: appears to be on the book cover (lower area)

# For grasping the tape roll from above (side grasp from above):
# We'll use keypoint 0 (center of tape roll front face) as the grasp point
# For placing on top of the book, we'll use keypoint 3 (book cover top area)
# The book top surface center can be estimated between keypoints 3 and 7

num_stages = 4

def stage1_subgoal_constraint1(end_effector, keypoints):
    """EE positioned 10cm above the tape roll center (keypoint 0) for approach from above."""
    target = keypoints[0] + np.array([0, 0, 0.10])
    return np.linalg.norm(end_effector - target)

def stage2_subgoal_constraint1(end_effector, keypoints):
    """EE at the tape roll side center (keypoint 0) - grasp from above."""
    return np.linalg.norm(end_effector - keypoints[0])

def stage3_subgoal_constraint1(end_effector, keypoints):
    """Held tape roll (keypoint 0 moves with EE) positioned above the book surface.
    Book center estimated between keypoints 3 and 7. Tape should be ~12cm above book."""
    book_center_xy = (keypoints[3][:2] + keypoints[7][:2]) / 2.0
    book_top_z = max(keypoints[3][2], keypoints[7][2])
    tape_xy = keypoints[0][:2]
    tape_z = keypoints[0][2]
    xy_cost = np.linalg.norm(tape_xy - book_center_xy)
    z_cost = abs(tape_z - (book_top_z + 0.12))
    return xy_cost + z_cost

def stage4_subgoal_constraint1(end_effector, keypoints):
    """Tape roll (keypoint 0) placed on top of the book - 
    aligned with book center and resting on book surface."""
    book_center_xy = (keypoints[3][:2] + keypoints[7][:2]) / 2.0
    book_top_z = max(keypoints[3][2], keypoints[7][2])
    tape_xy = keypoints[0][:2]
    tape_z = keypoints[0][2]
    xy_cost = np.linalg.norm(tape_xy - book_center_xy)
    # Tape roll height ~4cm, so center should be ~4cm above book top
    z_cost = abs(tape_z - (book_top_z + 0.04))
    return xy_cost + z_cost

# Path constraint: keep tape roll elevated during transit to avoid collisions
def stage3_path_constraint1(end_effector, keypoints):
    """Keep EE (and tape) above table level during transit - at least 10cm above table."""
    table_z = keypoints[5][2]
    return table_z + 0.10 - end_effector[2]

STAGE_CONSTRAINTS = [
    [stage1_subgoal_constraint1],
    [stage2_subgoal_constraint1],
    [stage3_subgoal_constraint1],
    [stage4_subgoal_constraint1],
]
STAGE_PATH_CONSTRAINTS = [[], [], [stage3_path_constraint1], []]
STAGE_NAMES = ["approach tape", "grasp tape", "transit to book", "place on book"]
STAGE_MOVABLE_MASK = [
    [False] * 8,                              # stage 1: nothing grasped
    [False] * 8,                              # stage 2: grasp moment
    [i == 0 for i in range(8)],               # stage 3: keypoint 0 (tape) held
    [i == 0 for i in range(8)],               # stage 4: keypoint 0 (tape) still held
]
STAGE_GRIPPER_ACTION = ["open", "closed", "closed", "open"]
