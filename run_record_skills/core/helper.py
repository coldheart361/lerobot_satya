import numpy as np
import cv2
from joblib import Parallel, delayed

K           = np.load("../../calibration/intrinsics/camera_matrix.npy")
depth_scale = np.load("../../calibration/intrinsics/depth_scale.npy")[0]
T_bc        = np.load("../../calibration/T_base_camera.npy")
URDF_PATH   = "../../so101.urdf"
TCP_OFFSET_M = np.array([0.004016, -0.004152, 0.015589, 1.0])

def approach_from_elevation(target_m, elevation_deg):
    """
    Compute the gripper approach direction given:
      target_m       : [3] target position in robot base frame (metres)
      elevation_deg  : 0 = horizontal approach from robot side
                       90 = straight down (top-down)

    The horizontal component always points FROM robot base TOWARD the target.
    The vertical component is straight down.
    This matches the SO-101's constraint: once shoulder_pan locks the
    horizontal direction, elevation_deg controls how steep the approach is.
    """
    el = np.deg2rad(np.clip(elevation_deg, 0, 90))
    h  = np.array([target_m[0], target_m[1], 0.0])
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-6:
        return np.array([0.0, 0.0, -1.0])
    h_unit = h / h_norm
    direction = np.array([
        np.cos(el) * h_unit[0],
        np.cos(el) * h_unit[1],
        -np.sin(el),
    ])
    return direction / np.linalg.norm(direction)

def project_point_to_pixel(point_base_cm, T_bc, K):
    p_base_h = np.array([*point_base_cm, 1.0])
    T_cb = np.linalg.inv(T_bc)
    p_cam = T_cb @ p_base_h
    x, y, z = p_cam[:3]
    if z <= 0:
        return None
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u = int(fx * x / z + cx)
    v = int(fy * y / z + cy)
    return (u, v)


def draw_robot_axes(image, T_bc, K, axis_length_cm=0, arrow_thickness=2):
    origin = project_point_to_pixel([0, 0, 0], T_bc, K)
    x_tip  = project_point_to_pixel([axis_length_cm, 0, 0], T_bc, K)
    y_tip  = project_point_to_pixel([0, axis_length_cm, 0], T_bc, K)
    z_tip  = project_point_to_pixel([0, 0, axis_length_cm], T_bc, K)

    if None in [origin, x_tip, y_tip, z_tip]:
        print("[axes] some axis points outside frame — skipping")
        return image

    img = image.copy()
    cv2.arrowedLine(img, origin, x_tip, (0, 0, 255), arrow_thickness, tipLength=0.2)
    cv2.arrowedLine(img, origin, y_tip, (0, 255, 0), arrow_thickness, tipLength=0.2)
    cv2.arrowedLine(img, origin, z_tip, (255, 0, 0), arrow_thickness, tipLength=0.2)
    cv2.putText(img, "+X", (x_tip[0]+5, x_tip[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), arrow_thickness)
    cv2.putText(img, "+Y", (y_tip[0]+5, y_tip[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), arrow_thickness)
    cv2.putText(img, "+Z", (z_tip[0]+5, z_tip[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), arrow_thickness)
    return img


def check_keypoint_reachability(keypoints_3d, ik_solver, elevations=(90, 60, 45, 30, 0)):
    report = []
    for i, kp in enumerate(keypoints_3d):
        errors = {}
        best_err = float('inf')
        for el in elevations:
            approach = approach_from_elevation(kp, el)
            _, err = ik_solver.solve(kp, approach_dir=approach)
            errors[el] = round(err * 100, 2)
            best_err = min(best_err, errors[el])

        report.append({
            "keypoint":             i,
            "position_cm":          kp * 100,
            "reachable":            best_err < 3.0,
            "elevation_errors_cm":  errors,
        })
    return report

# injected into the exec namespace alongside np
def check_reachability(point_m, ik_solver, elevation_deg=90, roll_deg=0): 
    approach = approach_from_elevation(point_m, elevation_deg)
    _, err = ik_solver.solve(point_m, approach_dir=approach)
    return {"reachable": err * 100 < 3.0, "ik_error_cm": round(err * 100, 2)}

def reachability_report_str(report):
    """Format report as string for VLM prompt."""
    lines = []
    for r in report:
        errs = "  ".join(f"{el}°→{r['elevation_errors_cm'][el]:.1f}cm"
                         for el in sorted(r['elevation_errors_cm'], reverse=True))
        lines.append(
            f"  Keypoint {r['keypoint']}: [{errs}]"
            + ("  ✓" if r['reachable'] else "  ✗ UNREACHABLE")
        )
    return "\n".join(lines)

def build_points_array(depth_raw):
    H, W = depth_raw.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    z_cm = depth_raw.astype(np.float32) * depth_scale * 100.0
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    X_cm = (us - cx) * z_cm / fx
    Y_cm = (vs - cy) * z_cm / fy
    invalid = (depth_raw == 0) | (depth_raw >= 65535)
    X_cm[invalid] = np.nan; Y_cm[invalid] = np.nan; z_cm[invalid] = np.nan
    cam_homo  = np.stack([X_cm, Y_cm, z_cm, np.ones_like(X_cm)], axis=-1)
    base_homo = cam_homo @ T_bc.T
    return base_homo[..., :3] / 100.0

def find_boundary_radius(z_m, elevation_deg, theta, ik_solver,
                          r_min=0.05, r_max=0.44, n_steps=8):
    # pre-check: is anything reachable along this ray?
    reachable = False
    for r_test in np.linspace(0.10, 0.40, 4):
        pt = np.array([r_test * np.cos(theta), r_test * np.sin(theta), z_m])
        _, err = ik_solver.solve(pt,
                    approach_dir=approach_from_elevation(pt, elevation_deg))
        if err * 100 < 3.0:
            reachable = True
            break

    if not reachable:
        return None   # ← nothing reachable, skip this direction

    # binary search
    lo, hi = r_min, r_max
    for _ in range(n_steps):
        r  = (lo + hi) / 2
        pt = np.array([r * np.cos(theta), r * np.sin(theta), z_m])
        _, err = ik_solver.solve(pt,
                    approach_dir=approach_from_elevation(pt, elevation_deg))
        if err * 100 < 1.5:
            lo = r
        else:
            hi = r
        print(f"    z={z_m:.2f}m el={elevation_deg}° θ={np.rad2deg(theta):.1f}° → r={r:.3f}m (err={err*100:.2f}cm)")
    return lo


def build_boundary_overlay(image, ik_solver, T_bc, K,
                            z_levels=(0.0,),
                            elevations=(90,),        # ← tuple!
                            n_angles=36):
    COLORS = {
        90: (0,   255, 0  ),
        60: (0,   255, 255),
        45: (0,   165, 255),
        30: (0,   0,   255),
        0:  (255, 0,   255),   # ← add this

    }
    img    = image.copy()
    thetas = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    for z in z_levels:
        for el in elevations:
            boundary_3d = []
            for theta in thetas:
                r  = find_boundary_radius(z, el, theta, ik_solver)
                if r is None:
                    continue   # ← skip unreachable directions
                pt = np.array([r * np.cos(theta), r * np.sin(theta), z])
                boundary_3d.append(pt)

            if len(boundary_3d) < 3:
                print(f"  z={z} el={el}: not enough reachable points, skipping")
                continue

            pixels = []
            for pt in boundary_3d:
                px = project_point_to_pixel(pt * 100, T_bc, K)
                if px is not None:
                    pixels.append(px)

            if len(pixels) < 3:
                continue

            pts_arr = np.array(pixels, dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts_arr], COLORS[el])
            img = cv2.addWeighted(img, 0.85, overlay, 0.15, 0)
            cv2.polylines(img, [pts_arr], isClosed=True,
                          color=COLORS[el], thickness=2)

    return img

def find_reachable_range_nonmonotonic(z_m, elevation_deg, theta, ik_solver,
                                       r_min=0.05, r_max=0.44,
                                       threshold_cm=1.5):
    def err_at(r):
        pt = np.array([r * np.cos(theta), r * np.sin(theta), z_m])
        _, err = ik_solver.solve(pt,
                    approach_dir=approach_from_elevation(pt, elevation_deg),
                    n_restarts=2)
        return err * 100

    # ── Step 1: ternary search for minimum error ──────────────────────────
    lo, hi = r_min, r_max
    for _ in range(12):   # log3(range/precision) steps
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if err_at(m1) < err_at(m2):
            hi = m2
        else:
            lo = m1
    r_best = (lo + hi) / 2
    err_best = err_at(r_best)

    if err_best >= threshold_cm:
        return None, None   # even the sweet spot is unreachable

    # ── Step 2: binary search INWARD from sweet spot ──────────────────────
    lo_in, hi_in = r_min, r_best
    for _ in range(10):
        r = (lo_in + hi_in) / 2
        if err_at(r) < threshold_cm:
            hi_in = r   # reachable → try closer
        else:
            lo_in = r   # unreachable → move outward
    r_inner = hi_in

    # ── Step 3: binary search OUTWARD from sweet spot ─────────────────────
    lo_out, hi_out = r_best, r_max
    for _ in range(10):
        r = (lo_out + hi_out) / 2
        if err_at(r) < threshold_cm:
            lo_out = r   # reachable → try farther
        else:
            hi_out = r   # unreachable → move inward
    r_outer = lo_out

    return r_inner, r_outer

def build_boundary_overlay_v2(image, ik_solver, T_bc, K,
                            z_levels=(0.0, 0.05, 0.10),
                            elevations=(90, 60, 45, 30),
                            n_angles=18):
    COLORS = {
        90: (0,   255, 0  ),
        60: (0,   255, 255),
        45: (0,   165, 255),
        30: (0,   0,   255),
        0:  (255, 0,   255),
        None: (60, 60, 60),
    }

    img    = image.copy()
    thetas = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    for z in z_levels:
        for el in elevations:
            outer_pts, inner_pts = [], []

            for theta in thetas:
                r_inner, r_outer = find_reachable_range_nonmonotonic(
                    z, el, theta, ik_solver)

                if r_outer is not None:
                    pt = np.array([r_outer*np.cos(theta), r_outer*np.sin(theta), z])
                    px = project_point_to_pixel(pt*100, T_bc, K)
                    if px: outer_pts.append(px)

                if r_inner is not None:
                    pt = np.array([r_inner*np.cos(theta), r_inner*np.sin(theta), z])
                    px = project_point_to_pixel(pt*100, T_bc, K)
                    if px: inner_pts.append(px)

            if len(outer_pts) < 3:
                print(f"  z={z} el={el}: not enough reachable points, skipping")
                continue

            color  = COLORS.get(el, COLORS[None])
            outer  = np.array(outer_pts, dtype=np.int32)
            overlay = img.copy()

            # fill outer polygon
            cv2.fillPoly(overlay, [outer], color)

            # punch out inner polygon (black hole)
            if len(inner_pts) >= 3:
                inner = np.array(inner_pts, dtype=np.int32)
                cv2.fillPoly(overlay, [inner], (0, 0, 0))

            # blend with transparency
            img = cv2.addWeighted(img, 0.85, overlay, 0.15, 0)

            # draw boundary lines
            cv2.polylines(img, [outer], True, color, 2)
            if len(inner_pts) >= 3:
                cv2.polylines(img, [inner], True, color, 1)

    return img