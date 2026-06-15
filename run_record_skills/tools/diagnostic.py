import numpy as np
from ik_solver import IKSolver

ik = IKSolver("../so101.urdf", np.array([0.004016, -0.004152, 0.015589, 1.0]))

# test the positions that gave bad results earlier
targets = [
    np.array([0.280,  0.094, 0.087]),   # gave 5.28cm in click_orient_go
    np.array([0.297,  0.000, 0.088]),   # gave 5.30cm
    np.array([0.338,  0.034, 0.050]),   # gave 5.90cm
    np.array([0.300,  0.000, 0.050]),   # your diagnostic (works fine)
]

for target in targets:
    best_err = float('inf')
    for _ in range(20):
        guess = np.zeros(7)
        guess[1:6] = np.random.uniform(-1.5, 1.5, 5)
        joints, err = ik.solve(target, initial_guess=guess, approach_dir=np.array([0., 0., -1.]))
        best_err = min(best_err, err * 100)
    print(f"target {target*100}cm → best_err={best_err:.2f}cm")
approach = np.array([0., 0., -1.])      # top-down