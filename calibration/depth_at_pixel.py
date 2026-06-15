import numpy as np
import cv2

depth_raw = np.load("image/depth.npy")
depth_scale = np.load("intrinsics/depth_scale.npy")[0]
img = cv2.imread("image/scene.png")

def depth_at(u, v, patch=7):
    H, W = depth_raw.shape
    r = patch // 2
    region = depth_raw[max(0,v-r):v+r+1, max(0,u-r):u+r+1]
    valid = region[(region > 0) & (region < 65535)]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) * depth_scale * 100

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        d = depth_at(x, y)
        label = f"{d:.1f}cm" if d else "NO DEPTH"
        print(f"pixel ({x},{y}) = {label}")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, label, (x+8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.imshow("click stickers", img)

cv2.imshow("click stickers", img)
cv2.setMouseCallback("click stickers", on_click)
print("Click on stickers/book. Press Q to quit.")
while cv2.waitKey(1) & 0xFF != ord('q'):
    pass
cv2.destroyAllWindows()