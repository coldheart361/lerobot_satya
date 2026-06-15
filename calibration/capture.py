import pyrealsense2 as rs
import numpy as np
import cv2
import time
import subprocess


def try_capture():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgra8, 6)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        pipeline.start(config)

        align = rs.align(rs.stream.color)
        # no hole filling — use raw depth

        for i in range(100):
            frames = pipeline.wait_for_frames(5000)
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            depth = np.asarray(depth_frame.get_data())
            valid = np.sum((depth > 0) & (depth < 65535))
            print(f"  Frame {i}: {valid} valid pixels")
            if valid > 830000:
                color = np.asarray(frames.get_color_frame().get_data())
                pipeline.stop()
                return color, depth

        pipeline.stop()
        return None, None

    except Exception as e:
        print(f"  Error: {e}")
        try:
            pipeline.stop()
        except Exception:
            pass
        return None, None


attempt = 0
while True:
    attempt += 1
    print(f"\nAttempt {attempt} — resetting camera...")

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

    color, depth = try_capture()

    if color is not None:
        cv2.imwrite("image/scene.png", color)
        np.save("image/depth.npy", depth)
        print(f"Saved on attempt {attempt}!")
        break
    else:
        print("Failed, retrying in 1 second...")
        time.sleep(1)