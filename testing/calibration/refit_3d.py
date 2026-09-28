"""Redo the 3D fit from calibration_points.csv, optionally dropping points
(e.g. a misclick) without having to measure everything again.

usage: python testing/calibration/refit_3d.py [-x 4,11]
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "testing")]

import numpy as np

from calibrate import solve_rigid_transform, write_result, POINTS_CSV, OUT_PATH, MIN_POINTS


def load_points(exclude):
    indices, cam, base = [], [], []
    with open(POINTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            index = int(row["index"])
            if index in exclude:
                continue
            xyz = [float(row["cam_x_m"]), float(row["cam_y_m"]), float(row["cam_z_m"])]
            if not all(np.isfinite(xyz)):
                continue
            indices.append(index)
            cam.append(xyz)
            base.append([float(row["base_x_mm"]) / 1000.0,
                         float(row["base_y_mm"]) / 1000.0,
                         float(row["base_z_mm"]) / 1000.0])
    return indices, np.array(cam), np.array(base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--exclude", default="",
                        help="comma separated point indices to drop")
    args = parser.parse_args()

    if not os.path.exists(POINTS_CSV):
        print(f"no {POINTS_CSV}, run calibrate.py first")
        return

    exclude = {int(p) for p in args.exclude.replace(",", " ").split()}
    indices, cam_points, base_points = load_points(exclude)
    if exclude:
        print(f"excluding point(s): {sorted(exclude)}")
    if len(indices) < MIN_POINTS:
        print(f"only {len(indices)} point(s) left, need at least {MIN_POINTS}")
        return

    R, t = solve_rigid_transform(cam_points, base_points)
    residuals = np.linalg.norm((cam_points @ R.T + t) - base_points, axis=1) * 1000.0
    rms = np.sqrt((residuals ** 2).mean())

    print(f"\n{len(indices)} points")
    worst = residuals.argmax()
    for k, (index, residual) in enumerate(zip(indices, residuals)):
        flag = "   <-- worst" if k == worst and len(indices) > 1 else ""
        print(f"  point {index:>3}: {residual:6.2f} mm{flag}")
    print(f"  RMS {rms:.2f} mm   max {residuals.max():.2f} mm")

    header = ["from testing/calibration/refit_3d.py",
              f"{len(indices)} points, RMS {rms:.2f} mm, max {residuals.max():.2f} mm"]
    if exclude:
        header.append(f"excluded: {sorted(exclude)}")
    write_result(OUT_PATH, R, t, header)
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
