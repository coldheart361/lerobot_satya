# compare_ik.py
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from ik_solver import IKSolver

TCP_OFFSET_M = np.array([0.004016, -0.004152, 0.015589, 1.0])
URDF_PATH    = "../so101.urdf"

# load with roboticstoolbox
robot_rtb = rtb.ERobot.URDF(URDF_PATH)
print(robot_rtb)   # verify it loaded correctly, check joint names

# load with ikpy (current)
ik = IKSolver(URDF_PATH, TCP_OFFSET_M)

targets = [
    np.array([0.280,  0.094, 0.087]),
    np.array([0.297,  0.000, 0.088]),
    np.array([0.338,  0.034, 0.050]),
    np.array([0.300,  0.000, 0.050]),
]

approach = np.array([0., 0., -1.])

print(f"\n{'target':30s}  {'ikpy':>10s}  {'rtb_LM':>10s}  {'rtb_QP':>10s}")
for t in targets:
    # ikpy with 20 restarts
    best_ikpy = float('inf')
    for _ in range(20):
        g = np.zeros(7)
        g[1:6] = np.random.uniform(-1.5, 1.5, 5)
        _, err = ik.solve(t, initial_guess=g, approach_dir=approach)
        best_ikpy = min(best_ikpy, err * 100)

    # roboticstoolbox LM
    T_target = SE3(t)
    sol_lm = robot_rtb.ik_LM(T_target, joint_limits=True)
    err_lm = 999.0
    if sol_lm.success:
        # compute actual tip error
        fk = robot_rtb.fkine(sol_lm.q)
        err_lm = np.linalg.norm(fk.t - t) * 100

    # roboticstoolbox QP
    sol_qp = robot_rtb.ik_QP(T_target, joint_limits=True)
    err_qp = 999.0
    if sol_qp.success:
        fk = robot_rtb.fkine(sol_qp.q)
        err_qp = np.linalg.norm(fk.t - t) * 100

    print(f"{str(t*100):30s}  {best_ikpy:>9.2f}cm  {err_lm:>9.2f}cm  {err_qp:>9.2f}cm")