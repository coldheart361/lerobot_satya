import numpy as np

depth_raw = np.load("depth.npy")       # uint16, in depth units
depth_scale = np.load("depth_scale.npy")[0]  # from intrinsic.py


import numpy as np

depth_raw = np.load("depth.npy")
depth_scale = np.load("depth_scale.npy")[0]

def depth_at(u, v, patch=5):
    """Median depth in patch, ignoring zeros (invalid pixels)."""
    H, W = depth_raw.shape
    r = patch // 2
    y0, y1 = max(0, v - r), min(H, v + r + 1)
    x0, x1 = max(0, u - r), min(W, u + r + 1)
    region = depth_raw[y0:y1, x0:x1]
    valid = region[region > 0]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) * depth_scale * 100  # cm

# for a pixel (u, v):
pixels = [(360, 646), 
          (526, 548), 
          (738, 643),
          (729, 490),
          (1006, 643),
          (560, 302),
          (563, 108),
          (629, 111),
          (683, 308),
          (757, 308)]
for u, v in pixels:
    d = depth_at(u, v, patch=7)
    print(f"({u},{v}) = {d if d else 'NO DEPTH'} cm")