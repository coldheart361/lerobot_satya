"""
intrinsic.py
============
Read factory-calibrated intrinsics from Intel RealSense D435.
Auto-retries until the camera connects (handles M1 Mac power flakiness).
Usage:
    sudo python intrinsic.py
"""
import time
import subprocess
import numpy as np
import pyrealsense2 as rs


def grab_intrinsics():
    """Try once to read intrinsics. Returns True on success, False on failure."""
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgra8, 6)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,   6)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"  start failed: {e}")
        return False

    try:
        # confirm frames actually arrive before trusting the device
        for _ in range(3):
            pipeline.wait_for_frames(5000)

        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        print(f"\nCamera intrinsics ({intr.width}x{intr.height}):")
        print(f"  fx = {intr.fx:.4f}")
        print(f"  fy = {intr.fy:.4f}")
        print(f"  cx = {intr.ppx:.4f}")
        print(f"  cy = {intr.ppy:.4f}")
        print(f"  distortion model: {intr.model}")
        print(f"  coeffs: {intr.coeffs}")

        K = np.array([
            [intr.fx, 0,       intr.ppx],
            [0,       intr.fy, intr.ppy],
            [0,       0,       1       ]
        ], dtype=np.float64)
        dist = np.array(intr.coeffs, dtype=np.float64)

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"\nDepth scale: {depth_scale} meters per unit")

        np.save("intrinsics/camera_matrix.npy", K)
        np.save("intrinsics/dist_coeffs.npy",   dist)
        np.save("intrinsics/depth_scale.npy",   np.array([depth_scale]))

        print("\nSaved intrinsics/camera_matrix.npy, intrinsics/dist_coeffs.npy, intrinsics/depth_scale.npy")
        print("\nCamera matrix K:")
        print(K)
        return True

    except RuntimeError as e:
        print(f"  frame grab failed: {e}")
        return False
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def main():
    attempt = 0
    while True:
        attempt += 1
        print(f"\nAttempt {attempt} - resetting camera...")
        subprocess.run(["killall", "VDCAssistant"],        capture_output=True)
        subprocess.run(["killall", "AppleCameraAssistant"], capture_output=True)

        # hardware reset the camera to clear stuck state
        try:
            ctx = rs.context()
            if len(ctx.devices) > 0:
                ctx.devices[0].hardware_reset()
                print("  hardware reset sent, waiting...")
                time.sleep(3)
        except Exception as e:
            print(f"  reset failed: {e}")

        time.sleep(1)

        if grab_intrinsics():
            print(f"\nSuccess on attempt {attempt}!")
            break
        else:
            print("  retrying...")
            time.sleep(1)


if __name__ == "__main__":
    main()