"""
intrinsic.py
============
Read factory-calibrated intrinsics from Intel RealSense D435
and save as camera_matrix.npy and dist_coeffs.npy.

Usage:
    python intrinsic.py
"""

import numpy as np
import pyrealsense2 as rs

def main():
    pipeline = rs.pipeline()
    config   = rs.config()

    # Enable color stream at 1280x720
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,  30)

    print("Starting RealSense pipeline...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start pipeline: {e}")
        print("Make sure the D435 is plugged in via USB.")
        return

    # Read color intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    print(f"\nCamera intrinsics ({intr.width}x{intr.height}):")
    print(f"  fx = {intr.fx:.4f}")
    print(f"  fy = {intr.fy:.4f}")
    print(f"  cx = {intr.ppx:.4f}")
    print(f"  cy = {intr.ppy:.4f}")
    print(f"  distortion model: {intr.model}")
    print(f"  coeffs: {intr.coeffs}")

    # Build camera matrix K
    K = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1       ]
    ], dtype=np.float64)

    dist = np.array(intr.coeffs, dtype=np.float64)

    # Also read depth intrinsics (useful later)
    depth_stream = profile.get_stream(rs.stream.depth)
    depth_intr   = depth_stream.as_video_stream_profile().get_intrinsics()
    depth_scale  = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"\nDepth scale: {depth_scale} meters per unit")
    print(f"  (depth_frame[v,u] * {depth_scale:.6f} = metres)")

    pipeline.stop()

    # Save
    np.save("camera_matrix.npy", K)
    np.save("dist_coeffs.npy",   dist)
    np.save("depth_scale.npy",   np.array([depth_scale]))

    print("\nSaved:")
    print("  camera_matrix.npy")
    print("  dist_coeffs.npy")
    print("  depth_scale.npy")
    print("\nCamera matrix K:")
    print(K)

if __name__ == "__main__":
    main()