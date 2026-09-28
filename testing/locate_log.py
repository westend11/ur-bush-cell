"""Locate the bush, then type in where the robot actually touched it.
At the end it splits the error into a constant offset and the spread
around it, and checks whether the error grows with distance from the
camera (which would mean the bush height is off).

usage: python testing/locate_log.py --bush small [--append]
keys:  SPACE/c capture and locate, q finish
"""

import argparse
import csv
import os
import sys
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from camera import RealSenseCamera
from vision import locate_bush
from bush_centre import find_centre, estimate_radius
from calibration import CALIBRATION_ID, WORK_AREA_PX, BUSH_PROFILES, CAMERA_CENTRE_MM
from pickers import display_scale, to_display

RESULTS = os.path.join(ROOT, "testing", "results")
OUT_CSV = os.path.join(RESULTS, "locate_log.csv")


def ask_measured(index, window=None):
    # input() blocks the gui loop, close the window so it doesn't hang
    if window is not None:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
    while True:
        raw = input(f"  [{index}] touched X Y in mm (or 'skip'): ").strip()
        if raw.lower() in ("skip", "s"):
            return None
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("      need two numbers, e.g. '-124.83 1056.20'")
            continue
        try:
            return np.array([float(p) for p in parts])
        except ValueError:
            print("      could not parse those as numbers")


def capture_and_locate(camera, bush):
    color_image, depth_map, intrinsics, filename = camera.capture()
    pose = locate_bush(color_image, depth_map, intrinsics, bush=bush)

    x0, y0, x1, y1 = WORK_AREA_PX
    crop = color_image[y0:y1, x0:x1]
    detection = find_centre(crop)
    if detection is None or pose is None:
        print("  no usable detection")
        return color_image, None
    cx, cy, confidence = detection
    radius = estimate_radius(crop, cx, cy)
    centre = (cx + x0, cy + y0)

    r_str = f"  r {radius:5.1f}px" if radius else ""
    print(f"  pixel ({centre[0]:7.1f},{centre[1]:7.1f})  conf {confidence:5.1f}{r_str}")
    print(f"  predicted  x={pose[0]:8.2f}  y={pose[1]:8.2f}  z={pose[2]:6.1f}")

    return color_image, {
        "file": os.path.basename(filename),
        "u": centre[0], "v": centre[1],
        "confidence": confidence,
        "radius": radius if radius else float("nan"),
        "predicted": np.array(pose[:2]),
        "z": pose[2],
    }


def report(rows, bush):
    errors = np.array([r["measured"] - r["predicted"] for r in rows])
    magnitudes = np.linalg.norm(errors, axis=1)
    mean = errors.mean(axis=0)
    spread = np.linalg.norm(errors - mean, axis=1)
    spread_rms = np.sqrt((spread ** 2).mean())

    print(f"\n{'=' * 66}")
    print(f"{len(rows)} points, bush = {bush or 'none'}")
    print(f"  raw error       mean {magnitudes.mean():5.2f} mm   max {magnitudes.max():5.2f} mm")
    print(f"  constant offset ({mean[0]:+6.2f}, {mean[1]:+6.2f}) mm   "
          f"magnitude {np.linalg.norm(mean):.2f} mm")
    print(f"  spread about it RMS {spread_rms:5.2f} mm   max {spread.max():5.2f} mm")
    print(f"  per-axis sigma  x {errors[:, 0].std():.2f}   y {errors[:, 1].std():.2f}")

    # with the right height, error shouldn't depend on distance from the camera
    if len(rows) >= 4:
        cam_xy = CAMERA_CENTRE_MM[:2]
        lever = np.array([np.linalg.norm(r["predicted"] - cam_xy) for r in rows])
        toward = []
        for r in rows:
            to_cam = cam_xy - r["predicted"]
            to_cam /= max(np.linalg.norm(to_cam), 1e-9)
            toward.append(float((r["measured"] - r["predicted"]) @ to_cam))
        if lever.std() > 20:
            slope = np.polyfit(lever, np.array(toward), 1)[0]
            implied = slope * (CAMERA_CENTRE_MM[2] - np.mean([r["z"] for r in rows]))
            print(f"\n  error vs distance slope {slope:+.5f} "
                  f"-> {implied:+.1f} mm of height error")
            if abs(implied) > 3:
                print(f"    adjust this bush's height_mm by about {implied:+.1f} mm")
            else:
                print("    height looks right")
        else:
            print("\n  points are all at a similar distance, add some near the edges")

    if np.linalg.norm(mean) > 3 * max(spread_rms, 0.3):
        print("\n  -> mostly a constant offset")
    else:
        print("\n  -> no dominant offset, just scatter")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bush", choices=sorted(BUSH_PROFILES))
    parser.add_argument("--append", action="store_true",
                        help="append to the existing log")
    args = parser.parse_args()

    print(f"calibration: {CALIBRATION_ID}")
    print(f"bush       : {args.bush or 'none'}")
    print("SPACE/c = capture and locate,  q = finish\n")

    os.makedirs(RESULTS, exist_ok=True)
    camera = RealSenseCamera()
    window = "SPACE/c = capture,  q = finish"
    cv2.namedWindow(window)
    rows = []

    # write every row straight away so nothing is lost if it crashes
    mode = "a" if args.append and os.path.exists(OUT_CSV) else "w"
    log = open(OUT_CSV, mode, newline="")
    writer = csv.writer(log)
    if mode == "w":
        writer.writerow(["timestamp", "file", "bush", "u", "v", "confidence",
                         "radius_px", "pred_x", "pred_y", "pred_z",
                         "meas_x", "meas_y", "err_x", "err_y"])
        log.flush()

    try:
        first = camera.capture()[0]
        display = to_display(first, display_scale(first))

        while True:
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key not in (ord("c"), 32):
                continue

            print(f"\n--- capture {len(rows) + 1} ---")
            try:
                color_image, result = capture_and_locate(camera, args.bush)
                display = to_display(color_image, display_scale(color_image))
            except Exception as exc:
                print(f"  capture failed: {exc!r}")
                continue
            if result is None:
                continue

            measured = ask_measured(len(rows) + 1, window)
            cv2.namedWindow(window)
            if measured is None:
                print("  skipped")
                continue

            result["measured"] = measured
            error = measured - result["predicted"]
            print(f"  error      ({error[0]:+6.2f}, {error[1]:+6.2f}) mm   "
                  f"magnitude {np.linalg.norm(error):.2f} mm")
            rows.append(result)

            writer.writerow([f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                             result["file"], args.bush or "",
                             f"{result['u']:.1f}", f"{result['v']:.1f}",
                             f"{result['confidence']:.1f}", f"{result['radius']:.1f}",
                             f"{result['predicted'][0]:.2f}",
                             f"{result['predicted'][1]:.2f}", f"{result['z']:.1f}",
                             f"{measured[0]:.2f}", f"{measured[1]:.2f}",
                             f"{error[0]:.2f}", f"{error[1]:.2f}"])
            log.flush()
    finally:
        log.close()
        camera.close()
        cv2.destroyAllWindows()

    if not rows:
        print("\nnothing logged")
        return

    report(rows, args.bush)
    print(f"\nlogged to {OUT_CSV}")


if __name__ == "__main__":
    main()
