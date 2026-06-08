import numpy as np
from scipy.optimize import dual_annealing, minimize
import copy
import time

# Assuming your transform_utils file handles basic Euler/Quaternion/Matrix conversions
import transform_utils as T
from utils import (
    get_linear_interpolation_steps,
    linear_interpolate_poses,
    normalize_vars,
    unnormalize_vars,
    get_samples_jitted,
    path_length,
    transform_keypoints,
)

def objective(opt_vars,
              og_bounds,
              start_pose,
              end_pose,
              keypoints_centered,
              keypoint_movable_mask,
              path_constraints,
              opt_interpolate_pos_step_size,
              opt_interpolate_rot_step_size,
              chain, # Passing your ikpy chain directly
              initial_joint_pos,
              tcp_offset):

    # Unnormalize decision variables from [-1, 1] back to physical world coordinates
    unnormalized_opt_vars = unnormalize_vars(opt_vars, og_bounds)
    control_points_euler = np.concatenate([start_pose[None], unnormalized_opt_vars.reshape(-1, 6), end_pose[None]], axis=0)
    control_points_homo = T.convert_pose_euler2mat(control_points_euler)
    control_points_quat = T.convert_pose_mat2quat(control_points_homo)
    
    # Interpolate dense fine-grained waypoints between our control points
    poses_quat, num_poses = get_samples_jitted(control_points_homo, control_points_quat, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size)
    poses_homo = T.convert_pose_quat2mat(poses_quat)
    
    cost = 0.0

    # 1. Penalize path length (Encourages smooth, straight lines; penalizes erratic paths)
    pos_length, rot_length = path_length(poses_homo)
    cost += 4.0 * (pos_length + rot_length * 1.0)

    # 2. Kinematic Reachability Cost (Evaluated via fast ikpy profiles)
    # Penalizes trajectories that guide the arm into unnatural singularities or blind spots
    guess = initial_joint_pos.copy()
    for cp_homo in control_points_homo:
        target_m = (cp_homo @ tcp_offset)[:3]
        # Quick analytical/numerical evaluation pass
        joints = chain.inverse_kinematics(target_m, initial_position=guess)
        tip_m = (chain.forward_kinematics(joints) @ tcp_offset)[:3]
        err_cm = np.linalg.norm(tip_m - target_m) * 100.0
        
        # Penalize solutions that deviate from achievable physical coordinates
        if err_cm > 5.0: 
            cost += 50.0 * err_cm
        guess = joints # Warm-start the next waypoint guess using this solution

    # 3. Path Constraint Violations (e.g., maintaining object orientation during transport)
    if path_constraints is not None and len(path_constraints) > 0:
        path_constraint_cost = 0.0
        start_idx, end_idx = 1, num_poses - 1
        
        for pose in poses_homo[start_idx:end_idx]:
            transformed_kps = transform_keypoints(pose, keypoints_centered, keypoint_movable_mask)
            for constraint in path_constraints:
                violation = constraint(transformed_kps[0], transformed_kps[1:])
                path_constraint_cost += np.clip(violation, 0, np.inf)
                
        cost += 200.0 * path_constraint_cost

    return cost


class PathSolver:
    """
    Optimizes intermediate end-effector waypoint trajectories using sequential 
    warm-started local optimizations to protect task constraints.
    """
    def __init__(self, config, chain, tcp_offset):
        self.config = config
        self.chain = chain
        self.tcp_offset = tcp_offset
        self.last_opt_result = None

    def _center_keypoints(self, ee_pose, keypoints, keypoint_movable_mask):
        ee_pose_homo = T.pose2mat([ee_pose[:3], T.euler2quat(ee_pose[3:])])
        centering_transform = np.linalg.inv(ee_pose_homo)
        keypoints_centered = transform_keypoints(centering_transform, keypoints, keypoint_movable_mask)
        return keypoints_centered

    def solve(self,
              start_pose,
              end_pose,
              keypoints,
              keypoint_movable_mask,
              path_constraints,
              initial_joint_pos,
              from_scratch=False):
        
        # Dynamically determine control nodes based on trajectory length
        num_control_points = get_linear_interpolation_steps(start_pose, end_pose, self.config['opt_pos_step_size'], self.config['opt_rot_step_size'])
        num_control_points = np.clip(num_control_points, 3, 6)
        
        # Convert inputs to uniform internal orientation formats (Euler)
        start_pose_euler = np.concatenate([start_pose[:3], T.quat2euler(start_pose[3:])])
        end_pose_euler = np.concatenate([end_pose[:3], T.quat2euler(end_pose[3:])])

        # Define optimization search space bounds
        og_bounds = [(b_min, b_max) for b_min, b_max in zip(self.config['bounds_min'], self.config['bounds_max'])] + [(-np.pi, np.pi) for _ in range(3)]
        og_bounds *= (num_control_points - 2)
        og_bounds = np.array(og_bounds, dtype=np.float64)
        bounds = [(-1, 1)] * len(og_bounds)

        # Handle Initial Guesses and Seed Optimization Variables
        if not from_scratch and self.last_opt_result is not None:
            init_sol = self.last_opt_result.x
            if len(init_sol) < len(bounds):
                new_x0 = np.empty(len(bounds))
                new_x0[:len(init_sol)] = init_sol
                for i in range(len(init_sol), len(bounds), 6):
                    new_x0[i:i+6] = init_sol[-6:] + np.random.randn(6) * 0.01
                init_sol = new_x0
            else:
                init_sol = init_sol[-len(bounds):]
        else:
            interp_poses = linear_interpolate_poses(start_pose_euler, end_pose_euler, num_control_points)
            init_sol = normalize_vars(interp_poses[1:-1].flatten(), og_bounds)

        init_sol = np.clip(init_sol, -1, 1)
        keypoints_centered = self._center_keypoints(start_pose_euler, keypoints, keypoint_movable_mask)

        aux_args = (og_bounds,
                    start_pose_euler,
                    end_pose_euler,
                    keypoints_centered,
                    keypoint_movable_mask,
                    path_constraints,
                    self.config['opt_interpolate_pos_step_size'],
                    self.config['opt_interpolate_rot_step_size'],
                    self.chain,
                    initial_joint_pos,
                    self.tcp_offset)

        # Execute Optimization
        if from_scratch:
            # Global Stochastic Optimization Pass (Only run on state transitions)
            opt_result = dual_annealing(
                func=objective,
                bounds=bounds,
                args=aux_args,
                maxfun=self.config['sampling_maxfun'],
                x0=init_sol,
                no_local_search=True,
                minimizer_kwargs={'method': 'SLSQP', 'options': self.config['minimizer_options']}
            )
        else:
            # Fast Local Gradient-Based Optimization Pass (Runs in milliseconds during execution)
            opt_result = minimize(
                fun=objective,
                x0=init_sol,
                args=aux_args,
                bounds=bounds,
                method='SLSQP',
                options=self.config['minimizer_options']
            )

        # Format optimized parameter sets back into standard 7D pose states
        sol = unnormalize_vars(opt_result.x, og_bounds)
        poses_euler = np.concatenate([sol.reshape(-1, 6), end_pose_euler[None]], axis=0)
        poses_quat = T.convert_pose_euler2quat(poses_euler)

        if opt_result.success or "iteration" in opt_result.message.lower():
            self.last_opt_result = copy.deepcopy(opt_result)

        return poses_quat