"""Check the current calibration against the robot on new points.

Click a point, touch it with the TCP and enter the pendant X/Y/Z (Base).
The summary separates a constant offset (usually TCP) from random scatter.

usage: python testing/calibration/validate.py
"""

import csv
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "testing")]

import cv2
import numpy as np

from camera import RealSenseCamera
from calibration import (deproject, cam_to_base, pixel_to_base_planar,
                         USE_PLANAR, CALIBRATION_ID)
from vision import _median_depth
from pickers import display_scale, to_display, refine_click

RESULTS = os.path.join(ROOT, "testing", "results")
OUT_CSV = os.path.join(RESULTS, "validation_points.csv")


def ask_base_coords(index):
    while True:
        raw = input(f"  point {index} measured X,Y,Z in mm (or 'skip'): ").strip()
        if raw.lower() == "skip":
            return None
        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            print("    need three numbers, e.g. '412.5 -190.0 15.2'")
            continue
        try:
            return np.array([float(p) for p in parts])
        except ValueError:
            print("    could not parse those as numbers")


def collect(color_image, depth_map, intrinsics):
    scale = display_scale(color_image)
    window = "click a point to validate  |  d=done  q=abort"
    pending, rows = {}, []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending["xy"] = (int(x / scale), int(y / scale))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    try:
        while True:
            vis = to_display(color_image, scale)
            for row in rows:
                cv2.drawMarker(vis, (int(row["u"] * scale), int(row["v"] * scale)),
                               (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.imshow(window, vis)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("d"), ord("q")):
                break

            if "xy" in pending:
                cx, cy = pending.pop("xy")
                refined = refine_click(color_image, cx, cy)
                if refined is None:
                    continue
                u, v = refined

                depth_m = _median_depth(depth_map, int(round(u)), int(round(v)))
                if USE_PLANAR:
                    predicted = pixel_to_base_planar(u, v)
                else:
                    if depth_m <= 0:
                        print(f"  no depth at ({u:.1f},{v:.1f}), pick another")
                        continue
                    predicted = cam_to_base(deproject(intrinsics, u, v, depth_m)) * 1000.0
                print(f"\n  point {len(rows) + 1}: predicted "
                      f"({predicted[0]:.1f}, {predicted[1]:.1f}, {predicted[2]:.1f}) mm")

                measured = ask_base_coords(len(rows) + 1)
                if measured is None:
                    continue
                rows.append({"u": u, "v": v, "depth_m": depth_m,
                             "predicted": predicted, "measured": measured})
                error = measured - predicted
                print(f"    error ({error[0]:+.1f}, {error[1]:+.1f}, {error[2]:+.1f}) mm"
                      f"   |e| {np.linalg.norm(error):.1f} mm")
    finally:
        cv2.destroyAllWindows()

    return rows


def report(rows):
    errors = np.array([r["measured"] - r["predicted"] for r in rows])
    magnitudes = np.linalg.norm(errors, axis=1)
    mean = errors.mean(axis=0)
    spread = np.linalg.norm(errors - mean, axis=1)
    systematic = np.linalg.norm(mean)
    random = np.sqrt((spread ** 2).mean())

    print(f"\n{'=' * 62}")
    print(f"{len(rows)} points")
    print(f"  mean error   ({mean[0]:+.1f}, {mean[1]:+.1f}, {mean[2]:+.1f}) mm"
          f"   |mean| {systematic:.1f} mm")
    print(f"  spread       RMS {random:.1f} mm   max {spread.max():.1f} mm")
    print(f"  raw          mean {magnitudes.mean():.1f} mm   max {magnitudes.max():.1f} mm")
    print()
    if systematic > 3 * max(random, 0.5):
        print("  -> constant offset. Check the TCP is the same one used for")
        print("     calibrating and that all points were read in Base.")
    elif random > 5:
        print("  -> scattered. Bad depth, misclicks, or calibration points too")
        print("     clustered. Recalibrate with a wider spread.")
    else:
        print("  -> looks fine, just noise")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    camera = RealSenseCamera()
    try:
        color_image, depth_map, intrinsics, filename = camera.capture()
    finally:
        camera.close()
    print(f"captured {filename}")
    print(f"calibration: {CALIBRATION_ID}\n")

    rows = collect(color_image, depth_map, intrinsics)
    if not rows:
        print("no points recorded")
        return

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "u", "v", "depth_m",
                         "pred_x", "pred_y", "pred_z",
                         "meas_x", "meas_y", "meas_z",
                         "err_x", "err_y", "err_z"])
        for i, row in enumerate(rows, start=1):
            error = row["measured"] - row["predicted"]
            writer.writerow([i, f"{row['u']:.2f}", f"{row['v']:.2f}",
                             f"{row['depth_m']:.6f}",
                             *(f"{p:.1f}" for p in row["predicted"]),
                             *(f"{m:.1f}" for m in row["measured"]),
                             *(f"{e:.1f}" for e in error)])

    report(rows)
    print(f"\nlogged to {OUT_CSV}")


if __name__ == "__main__":
    main()
