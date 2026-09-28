"""
Calibrate the camera at the bird's-eye pose.

1. Put the robot at the bird's-eye pose (with the final TCP set).
2. Click >= 4 well spread, non-collinear points on the table. Each point is
   a rough click on the overview, then an exact click on the zoomed crop
   (enter/space to accept, r to redo). Overview keys: u undo, d done, q abort.
3. Touch each point with the TCP and type in X Y Z from the pendant
   (Feature: Base, mm).

Points go to testing/results/calibration_points.csv (use fit_planar.py on
it), and the 3D rigid fit is written to calibration_result.py.

usage: python testing/calibration/calibrate.py
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "testing")]

import cv2
import numpy as np

from camera import RealSenseCamera
from calibration import deproject
from vision import _median_depth
from pickers import display_scale, to_display, refine_click

RESULTS = os.path.join(ROOT, "testing", "results")
OUT_PATH = os.path.join(RESULTS, "calibration_result.py")
POINTS_CSV = os.path.join(RESULTS, "calibration_points.csv")
MIN_POINTS = 4


def collect_pixels(color_image):
    """Returns [(u, v), ...] in full-res pixel coords."""
    picks = []
    window = "click near a point  |  u=undo  d=done  q=abort"
    scale = display_scale(color_image)
    pending = {}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending["xy"] = (int(x / scale), int(y / scale))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    aborted = False
    try:
        while True:
            vis = to_display(color_image, scale)
            for i, (u, v) in enumerate(picks, start=1):
                du, dv = int(u * scale), int(v * scale)
                cv2.drawMarker(vis, (du, dv), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(vis, str(i), (du + 8, dv - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(window, vis)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("u") and picks:
                print(f"  removed point {len(picks)}")
                picks.pop()
            elif key == ord("d"):
                break
            elif key == ord("q"):
                aborted = True
                break

            if "xy" in pending:
                cx, cy = pending.pop("xy")
                refined = refine_click(color_image, cx, cy)
                if refined is not None:
                    picks.append(refined)
                    print(f"  point {len(picks)}: pixel ({refined[0]:.1f}, {refined[1]:.1f})")
    finally:
        cv2.destroyAllWindows()

    return [] if aborted else picks


def ask_base_coords(index):
    while True:
        raw = input(f"  point {index} base X,Y,Z in mm (or 'skip'): ").strip()
        if raw.lower() == "skip":
            return None
        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            print("    need three numbers, e.g. '412.5 -190.0 15.2'")
            continue
        try:
            return [float(p) for p in parts]
        except ValueError:
            print("    could not parse those as numbers")


def solve_rigid_transform(cam_points, base_points):
    """Kabsch: R, t such that base = R @ cam + t."""
    cam_centroid = cam_points.mean(axis=0)
    base_centroid = base_points.mean(axis=0)

    H = (cam_points - cam_centroid).T @ (base_points - base_centroid)
    U, _, Vt = np.linalg.svd(H)

    # avoid a reflection
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = base_centroid - R @ cam_centroid
    return R, t


def write_result(path, R, t, header):
    with open(path, "w") as f:
        for line in header:
            f.write(f"# {line}\n")
        f.write("\nimport numpy as np\n\n")
        f.write("CAM_TO_BASE_R = np.array([\n")
        for row in R:
            f.write(f"    [{row[0]: .8f}, {row[1]: .8f}, {row[2]: .8f}],\n")
        f.write("])\n\n")
        f.write(f"CAM_TO_BASE_T = np.array([{t[0]: .8f}, {t[1]: .8f}, {t[2]: .8f}])\n")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    camera = RealSenseCamera()
    try:
        color_image, depth_map, intrinsics, filename = camera.capture()
    finally:
        camera.close()
    print(f"captured {filename}\n")
    print("click each calibration point, then press 'd'")

    picks = collect_pixels(color_image)
    if not picks:
        print("aborted")
        return
    if len(picks) < MIN_POINTS:
        print(f"only {len(picks)} point(s), need at least {MIN_POINTS}")
        return

    # points without depth are still fine for the planar fit
    measured, without_depth = [], []
    for i, (u, v) in enumerate(picks, start=1):
        depth_m = _median_depth(depth_map, int(round(u)), int(round(v)))
        if depth_m > 0:
            cam = deproject(intrinsics, u, v, depth_m)
        else:
            cam = np.full(3, np.nan)
            without_depth.append(i)
        measured.append((i, u, v, depth_m, cam))

    if without_depth:
        print(f"\nno depth at point(s) {without_depth}, only used for the planar fit")

    print("\ncamera is done, you can unplug it if the cable is in the way")
    print("(don't touch the bracket)")
    print("\njog the TCP to each point and enter X/Y/Z from the pendant\n")

    cam_points, base_points = [], []
    log = open(POINTS_CSV, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["index", "u", "v", "depth_m",
                     "cam_x_m", "cam_y_m", "cam_z_m",
                     "base_x_mm", "base_y_mm", "base_z_mm"])

    try:
        for i, u, v, depth_m, cam in measured:
            base_mm = ask_base_coords(i)
            if base_mm is None:
                continue

            cam_points.append(cam)
            base_points.append(np.array(base_mm) / 1000.0)

            writer.writerow([i, f"{u:.2f}", f"{v:.2f}", f"{depth_m:.6f}",
                             *(f"{c:.6f}" for c in cam),
                             *(f"{b:.2f}" for b in base_mm)])
            log.flush()
    finally:
        log.close()
        print(f"\npoints saved to {POINTS_CSV}")

    cam_points = np.array(cam_points)
    base_points = np.array(base_points)
    usable = np.isfinite(cam_points).all(axis=1)
    cam_points, base_points = cam_points[usable], base_points[usable]

    print("\nfor the planar model run: python testing/calibration/fit_planar.py")

    if len(cam_points) < MIN_POINTS:
        print(f"only {len(cam_points)} point(s) with depth, skipping the 3D fit")
        return

    R, t = solve_rigid_transform(cam_points, base_points)

    residuals = np.linalg.norm((cam_points @ R.T + t) - base_points, axis=1) * 1000.0
    rms = np.sqrt((residuals ** 2).mean())
    print(f"\n3D fit from {len(cam_points)} points")
    print("  per-point error (mm): " + "  ".join(f"{r:.1f}" for r in residuals))
    print(f"  RMS {rms:.2f} mm   max {residuals.max():.2f} mm")
    if residuals.max() > 10.0:
        print("  one point is way off, probably a misclick or typo")

    write_result(OUT_PATH, R, t, [
        "from testing/calibration/calibrate.py",
        f"{len(cam_points)} points, RMS {rms:.2f} mm, max {residuals.max():.2f} mm",
    ])
    print(f"\nwritten to {OUT_PATH}, copy the arrays into calibration.py")


if __name__ == "__main__":
    main()
