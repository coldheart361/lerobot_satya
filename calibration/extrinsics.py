"""
register_3d_3d.py
=================
3D-3D camera -> robot calibration (the "mentor's approach").

Recovers T_base_camera from pairs of the SAME points expressed two ways:
  - p_camera : 3D position in the camera frame  (from pixel + metric depth)
  - p_robot  : 3D position in the robot base frame (from touching with the gripper)

It solves for the rigid transform T such that:
      p_robot  ≈  R @ p_camera + t          (Umeyama / Kabsch algorithm)

Use this ALONGSIDE solvePnP as a cross-check (see procedure). If depth is good,
the two transforms should agree.
"""

import numpy as np


# ── pixel + metric depth -> 3D point in the camera frame (cm) ─────────────────
def backproject(u, v, depth_cm, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = float(depth_cm)
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], float)


# ── Umeyama: best rigid (or similarity) transform aligning src -> dst ─────────
def umeyama(src, dst, with_scale=False):
    """
    Find R, t, s so that  dst ≈ s * R @ src + t  (least-squares optimal).

    src, dst : (N,3) arrays of corresponding points (src=camera, dst=robot).
    with_scale=False -> rigid (use when depth is already METRIC).
    with_scale=True  -> similarity (recovers a global scale; use if your
                        camera points are only known UP TO SCALE, e.g. from
                        un-anchored relative depth). Note: a similarity fixes
                        scale but NOT an inverse-depth shift, so metric depth
                        is still preferable.
    Returns (R 3x3, t 3, s float).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    N = src.shape[0]

    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d

    Sigma = (Dc.T @ Sc) / N                 # 3x3 cross-covariance
    U, Dvals, Vt = np.linalg.svd(Sigma)

    # reflection fix so R is a proper rotation (det = +1)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    S = np.diag([1.0, 1.0, d])
    R = U @ S @ Vt

    if with_scale:
        var_s = (Sc ** 2).sum() / N
        s = float((Dvals * np.diag(S)).sum() / var_s)
    else:
        s = 1.0

    t = mu_d - s * R @ mu_s
    return R, t, s


def build_transform(R, t, s=1.0):
    """Pack into a 4x4 homogeneous matrix (scale folded into the rotation block)."""
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def residuals(src, dst, R, t, s=1.0):
    """Per-point alignment error in cm (how well the transform fits)."""
    pred = (s * (R @ src.T)).T + t
    return np.linalg.norm(pred - dst, axis=1)


# ── example wiring ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    K = np.load("calibration/camera_matrix.npy")

    # One row per calibration point. Use >= 4 (8+ better), NON-coplanar.
    #   (u, v)       : pixel in the image (click the point)
    #   depth_cm     : METRIC depth at that pixel (from metricized DA-V2)
    #   robot_xyz_cm : same point in robot base frame (touch with gripper, read TCP)
    CORR = [
        ( u,   v,   depth_cm, (Xr,    Yr,    Zr) ),
        (137, 432,  61.2,    (34.25,  3.51,  8.67)),
        (812, 455,  61.2,    (34.25,  3.51,  8.67)),
        (812, 455,  61.2,    (34.25,  3.51,  8.67)),
        (812, 455,  61.2,    (34.25,  3.51,  8.67)),
        (812, 455,  61.2,    (34.25,  3.51,  8.67)),
        (812, 455,  61.2,    (34.25,  3.51,  8.67)),
        (812, 455,  61.2,    (34.25,  3.51,  8.67)),
        (812, 455,  61.2,    (34.25,  3.51,  8.67)),
        (812, 455,  61.2,    (34.25,  3.51,  8.67)),
        (812, 455,  61.2,    (34.25,  3.51,  8.67)),
        (640, 380,  58.0,    (37.07,  9.47,  9.91)),
    ]
    if not CORR:
        raise SystemExit("Fill in CORR with your data first.")

    p_cam = np.array([backproject(u, v, d, K) for (u, v, d, _) in CORR])
    p_rob = np.array([xyz for (_, _, _, xyz) in CORR], float)

    # depth already metric -> rigid. If camera points are only up-to-scale,
    # set with_scale=True.
    R, t, s = umeyama(p_cam, p_rob, with_scale=True)
    T = build_transform(R, t, s)
    err = residuals(p_cam, p_rob, R, t, s)

    print("T_base_camera =\n", np.round(T, 4))
    print("scale s =", round(s, 5))
    print("per-point residual (cm):", np.round(err, 2))
    print(f"mean residual: {err.mean():.2f} cm   max: {err.max():.2f} cm")

    np.save("calibration/T_base_camera.npy", T)
    print("saved -> calibration/T_base_camera.npy")