# custom_ik.py — drop-in replacement for ikpy
import numpy as np
from scipy.optimize import minimize

def solve_lm(chain, target_m, tcp_offset, approach_dir=None, 
             roll_override=None, wrist_roll_idx=5, n_restarts=8):
    """
    Levenberg-Marquardt style IK via scipy minimize.
    Better than ikpy for near-limit configurations.
    """
    tcp_xyz = tcp_offset[:3]

    def fk_tip(q7):
        fk = chain.forward_kinematics(q7)
        return (fk @ tcp_offset)[:3], fk[:3, :3]

    def objective(q5):
        q7 = np.zeros(7)
        q7[1:6] = q5
        tip, R = fk_tip(q7)

        # position error (primary)
        pos_err = np.sum((tip - target_m) ** 2)

        # orientation error (if approach_dir given)
        orient_err = 0.0
        if approach_dir is not None:
            gripper_z = R[:, 2]   # SO-101 uses Z axis
            orient_err = 1.0 - np.dot(gripper_z, approach_dir)

        # soft joint limit penalty — penalize being NEAR limits
        limit_err = 0.0
        margin = 0.05  # 3° margin
        for i, link in enumerate(chain.links):
            b = getattr(link, 'bounds', None)
            if b is None: continue
            j = i  # joint index in q7
            if j < 1 or j > 5: continue
            qi = q5[j-1]
            if b[1] is not None and qi > b[1] - margin:
                limit_err += (qi - (b[1] - margin)) ** 2
            if b[0] is not None and qi < b[0] + margin:
                limit_err += ((b[0] + margin) - qi) ** 2

        return 100.0 * pos_err + 3.0 * orient_err + 10.0 * limit_err

    # joint bounds
    bounds = []
    for i, link in enumerate(chain.links):
        if 1 <= i <= 5:
            b = getattr(link, 'bounds', None)
            if b is not None:
                bounds.append((b[0], b[1]))
            else:
                bounds.append((-np.pi, np.pi))

    rng = np.random.default_rng(42)
    best_q7, best_err = None, np.inf

    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.zeros(5)
        else:
            x0 = rng.uniform([b[0] for b in bounds],
                             [b[1] for b in bounds])

        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7})

        q7 = np.zeros(7)
        q7[1:6] = res.x
        if roll_override is not None:
            q7[wrist_roll_idx] = roll_override

        tip, _ = fk_tip(q7)
        err = float(np.linalg.norm(tip - target_m))
        if err < best_err:
            best_err = err
            best_q7  = q7.copy()

    return best_q7, best_err