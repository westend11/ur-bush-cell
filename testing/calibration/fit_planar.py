"""Fit the planar model (homography for XY, plane for Z) from
calibration_points.csv. Look at the leave-one-out number, not the in-sample one.

usage: python testing/calibration/fit_planar.py [--ransac 2.0]
"""

import argparse
import csv
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)

import numpy as np

from planar import (estimate_homography, apply_homography, ransac_homography,
                    estimate_z_plane, apply_z_plane, loocv_error)

RESULTS = os.path.join(ROOT, "testing", "results")
POINTS_CSV = os.path.join(RESULTS, "calibration_points.csv")
OUT_PATH = os.path.join(RESULTS, "planar_result.py")


def load_points():
    with open(POINTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    pixels = np.array([[float(r["u"]), float(r["v"])] for r in rows])
    base = np.array([[float(r["base_x_mm"]), float(r["base_y_mm"]),
                      float(r["base_z_mm"])] for r in rows])
    indices = [int(r["index"]) for r in rows]
    return indices, pixels, base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ransac", type=float, metavar="MM",
                        help="reject points further than MM from the fit")
    args = parser.parse_args()

    if not os.path.exists(POINTS_CSV):
        print(f"no {POINTS_CSV}, run calibrate.py first")
        return

    indices, pixels, base = load_points()
    print(f"{len(indices)} points")

    spread = np.ptp(base[:, 2])
    print(f"z spread: {spread:.2f} mm" + ("" if spread < 20 else
          "  (points are at different heights, the 3D fit may be better)"))

    if args.ransac:
        H, inliers = ransac_homography(pixels, base[:, :2],
                                       threshold_mm=args.ransac, random_state=0)
        rejected = [indices[i] for i in np.where(~inliers)[0]]
        print(f"\nRANSAC {args.ransac} mm: kept {inliers.sum()}/{len(indices)}"
              + (f", rejected {rejected}" if rejected else ""))
    else:
        H = estimate_homography(pixels, base[:, :2])
        inliers = np.ones(len(indices), bool)

    z_coefficients = estimate_z_plane(pixels[inliers], base[inliers, 2])

    xy_error = np.linalg.norm(apply_homography(H, pixels) - base[:, :2], axis=1)
    z_error = apply_z_plane(z_coefficients, pixels) - base[:, 2]

    print("\nper-point error (mm):")
    for index, exy, ez, keep in zip(indices, xy_error, z_error, inliers):
        flag = "" if keep else "   (rejected)"
        print(f"  point {index:>3}:  XY {exy:5.2f}   Z {ez:+5.2f}{flag}")

    kept_xy = xy_error[inliers]
    xy_rms = np.sqrt((kept_xy ** 2).mean())
    print(f"\nin-sample  XY RMS {xy_rms:.2f} mm   max {kept_xy.max():.2f} mm")
    print(f"           Z  RMS {np.sqrt((z_error[inliers] ** 2).mean()):.2f} mm")

    if inliers.sum() >= 6:
        loo = loocv_error(pixels[inliers], base[inliers, :2])
        print(f"leave-one-out XY RMS {np.sqrt((loo ** 2).mean()):.2f} mm   max {loo.max():.2f} mm")

    with open(OUT_PATH, "w") as f:
        f.write("# from testing/calibration/fit_planar.py\n")
        f.write(f"# {int(inliers.sum())} points, in-sample XY RMS {xy_rms:.2f} mm\n\n")
        f.write("import numpy as np\n\n")
        f.write("PIXEL_TO_BASE_H = np.array([\n")
        for row in H:
            f.write(f"    [{row[0]: .10e}, {row[1]: .10e}, {row[2]: .10e}],\n")
        f.write("])\n\n")
        f.write(f"BASE_Z_PLANE = np.array([{z_coefficients[0]: .10e},"
                f"{z_coefficients[1]: .10e},{z_coefficients[2]: .10e}])\n")

    print(f"\nwritten to {OUT_PATH}, copy into calibration.py")


if __name__ == "__main__":
    main()
