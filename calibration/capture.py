import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgra8, 6)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
pipeline.start(config)

align = rs.align(rs.stream.color)
hole_filling = rs.hole_filling_filter()

# at 6fps, just wait for 2 frames
frames = pipeline.wait_for_frames(15000)
frames = pipeline.wait_for_frames(15000)

frames = align.process(frames)
color = np.asarray(frames.get_color_frame().get_data())
depth_frame = hole_filling.process(frames.get_depth_frame())
depth = np.asarray(depth_frame.get_data())

pipeline.stop()

cv2.imwrite("scene.png", color)
np.save("depth.npy", depth)
print(f"Saved — Color: {color.shape}, Depth: {depth.shape}")