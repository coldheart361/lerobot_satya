"""
subgoal_solver.py
=================
Find the EE position that minimizes VLM constraint costs.

DESIGN PHILOSOPHY:
  The VLM constraints already encode where the arm should go.
  Extra costs (reachability, regularization) fight against them
  and cause the solver to land in wrong positions.

  For typical VLM constraints like:
    return np.linalg.norm(end_effector - target)       # = 0 at target
    return np.linalg.norm(keypoints[0] - some_point)   # = 0 when kp0 is there

  The constraint alone has a well-defined minimum. We just find it.
  IK reachability is handled separately — if the target is truly
  unreachable, move_to_m reports the IK error and skips.

HELD KEYPOINT PROPAGATION:
  For transit/place stages with held objects, the VLM writes:
    np.linalg.norm(keypoints[0][:2] - keypoints[3][:2])
  where keypoints[0] is the held tape. Without propagation,
  keypoints[0] is constant regardless of candidate xyz — zero gradient,
  solver stays put. With propagation, the solver can see that moving
  the EE moves the held tape, and finds the right target.
"""

import numpy as np
from scipy.optimize import minimize


def _project_keypoints(keypoints, movable_mask, initial_ee, candidate_xyz):
    """
    Return what keypoints would be if EE moved from initial_ee to candidate_xyz.
    Held keypoints translate with the EE. Static ones stay put.
    """
    if movable_mask is None or not any(movable_mask):
        return keypoints
    kps  = keypoints.copy()
    disp = candidate_xyz - initial_ee
    for i, held in enumerate(movable_mask):
        if held:
            kps[i] = keypoints[i] + disp
    return kps


def solve_subgoal(
    constraints,
    keypoints,
    workspace_min,
    workspace_max,
    initial_ee,
    ik_solver=None,         # kept for API compatibility, not used
    is_grasp_stage=False,   # kept for API compatibility, not used
    movable_mask=None,
    n_restarts=8,
):
    """
    Find EE position minimizing VLM constraint costs only.

    Args:
      constraints   : list of fn(ee[3], keypoints[K,3]) -> cost  (<=0 satisfied)
      keypoints     : [K,3] current keypoint positions (metres)
      workspace_min : [3] lower bounds (metres)
      workspace_max : [3] upper bounds (metres)
      initial_ee    : [3] current EE — used as one of the starting points
      movable_mask  : list[bool] — which keypoints move with the EE
      n_restarts    : number of random starts to find global minimum
    """
    workspace_min = np.asarray(workspace_min, float)
    workspace_max = np.asarray(workspace_max, float)
    initial_ee    = np.asarray(initial_ee,    float)
    keypoints     = np.asarray(keypoints,     float)

    def objective(xyz):
        kps  = _project_keypoints(keypoints, movable_mask, initial_ee, xyz)
        cost = 0.0
        for fn in constraints:
            try:
                c = float(fn(xyz, kps))
                cost += max(c, 0.0)   # only penalize violations (c > 0)
            except Exception:
                cost += 1e6           # broken constraint → huge penalty, NOT 0
        return cost

    bounds = list(zip(workspace_min, workspace_max))

    # Many restarts: first is current EE, rest are random across workspace.
    # More restarts = better chance of finding the true constraint minimum.
    rng = np.random.default_rng(42)
    starts = [initial_ee.copy()] + [
        rng.uniform(workspace_min, workspace_max)
        for _ in range(n_restarts - 1)
    ]

    best_x, best_cost = None, np.inf
    for x0 in starts:
        x0 = np.clip(x0, workspace_min, workspace_max)
        res = minimize(
            objective, x0=x0,
            method="L-BFGS-B",       # better than SLSQP for box-constrained
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6},
        )
        if res.fun < best_cost:
            best_cost = float(res.fun)
            best_x    = res.x.copy()

    return best_x, best_cost