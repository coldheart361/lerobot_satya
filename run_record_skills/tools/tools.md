# tools/

Diagnostic and utility scripts. Run these to verify individual components before running the full pipeline.

## Scripts

### `click_and_go.py`
Capture a scene, click a pixel, robot moves there. Tests calibration accuracy and IK solver.
```bash
sudo -E python tools/click_and_go.py
sudo -E python tools/click_and_go.py --hover 0.0   # go to exact clicked point
```

### `click_orient_go.py`
Interactive orientation tuning. Click a point then use keyboard to adjust approach angle and roll.
```
W / S     → elevation +5° / -5°
A / D     → roll -10° / +10°
SPACE     → move robot
R         → re-capture image
Q         → quit
```
```bash
sudo -E python tools/click_orient_go.py
```

### `ik_test.py`
Check if a 3D coordinate is reachable at a given elevation and roll. Optionally move the robot there.
```bash
# check only
python tools/ik_test.py --x 25 --y -5 --z 8 --elevation 45 --roll 90

# check + move robot
sudo -E python tools/ik_test.py --move

# interactive mode
sudo -E python tools/ik_test.py --move
```

### `reachability.py`
Build a reachability boundary overlay image showing which areas of the workspace the arm can reach at each elevation angle. Slow to compute — run once and reuse the output image.
```bash
python tools/reachability.py
# output: ../outputs/reachability/traj_boundary.png
```

The boundary is an annulus (donut shape) per elevation level — the arm can only reach positions between an inner and outer radius at each angle.

### `orientation_check.py`
Verifies which FK axis (X, Y, or Z) corresponds to the gripper's approach direction. Run this if orientation control seems wrong.
```bash
sudo -E python tools/orientation_check.py
```