"""Run the full pipeline from the keyboard instead of the robot, and
optionally serve the result over Modbus so the pendant can read it.

usage: python testing/locate.py [--bush small|big] [--no-modbus]
keys:  SPACE/c capture and locate, q quit
"""

import argparse
import os
import sys
import threading
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import cv2
from pymodbus.server import StartTcpServer

from camera import RealSenseCamera
from vision import locate_bush, MIN_CONFIDENCE
from bush_centre import find_centre, estimate_radius
from calibration import CALIBRATION_ID, USE_PLANAR, WORK_AREA_PX, BUSH_PROFILES
from modbus_io import (encode_pose, decode_pose, build_context, write_pose,
                       HOST, PORT, UNIT_ID, POSE_X, POSE_Y, POS_SCALE)
from pickers import display_scale, to_display

OUT_DIR = os.path.join(ROOT, "testing", "results", "locate")


def start_server():
    ctx = build_context(lambda: None)
    threading.Thread(
        target=StartTcpServer,
        kwargs={"context": ctx, "address": (HOST, PORT)},
        daemon=True,
    ).start()
    print(f"modbus : serving unit {UNIT_ID} on {HOST}:{PORT}")
    return ctx


def report_registers(ctx):
    x_reg = ctx[UNIT_ID].getValues(3, POSE_X, count=1)[0]
    y_reg = ctx[UNIT_ID].getValues(3, POSE_Y, count=1)[0]

    print(f"  published  : 30002 = {x_reg:>5}    30003 = {y_reg:>5}")
    for label, reg in (("30002 (X)", x_reg), ("30003 (Y)", y_reg)):
        signed = reg - 65536 if reg > 32767 else reg
        note = "  <- negative, unwrap on the pendant" if reg > 32767 else ""
        print(f"    {label}: raw {reg:>5}  -> signed {signed:>6}  "
              f"-> {signed / POS_SCALE:8.1f} mm{note}")


def annotate(color_image, centre, radius, pose):
    vis = color_image.copy()
    x0, y0, x1, y1 = WORK_AREA_PX
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)
    if centre is not None:
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        if radius is not None:
            cv2.circle(vis, (cx, cy), int(round(radius)), (0, 255, 255), 3)
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        label = (f"({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}) mm"
                 if pose else "no pose")
        cv2.putText(vis, label, (cx - 200, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
    else:
        cv2.putText(vis, "NO BUSH FOUND", (60, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    return vis


def locate_once(camera, ctx=None, bush=None):
    color_image, depth_map, intrinsics, filename = camera.capture()
    print(f"\n{'=' * 60}\ncaptured {os.path.basename(filename)}")

    pose = locate_bush(color_image, depth_map, intrinsics, bush=bush)

    # detect again just for drawing / confidence, with the same crop
    x0, y0, x1, y1 = WORK_AREA_PX
    crop = color_image[y0:y1, x0:x1]
    detection = find_centre(crop)
    centre = radius = None
    if detection is not None:
        cx, cy, confidence = detection
        radius = estimate_radius(crop, cx, cy)
        centre = (cx + x0, cy + y0)
        weak = "   WEAK, rejected" if confidence < MIN_CONFIDENCE else ""
        print(f"  centre     : ({centre[0]:.1f}, {centre[1]:.1f}) px   "
              f"confidence {confidence:.1f}{weak}")
        print("  radius     : " + (f"{radius:.1f} px" if radius else "not estimated"))

        if bush and radius:
            low, high = BUSH_PROFILES[bush]["radius_px"]
            if not low <= radius <= high:
                print(f"  !! radius {radius:.0f}px outside {low}-{high}px "
                      f"for '{bush}', wrong bush selected?")
    else:
        print("  no bush detected")

    if pose is None:
        print("  -> no pose")
    else:
        x, y, z, rx, ry, rz = pose
        print(f"  pose (mm)  : x={x:.1f}  y={y:.1f}  z={z:.1f}")
        print(f"  orientation: rx={rx:.4f}  ry={ry:.4f}  rz={rz:.4f} rad")

        registers = encode_pose(pose)
        rounded = decode_pose(registers)
        loss = max(abs(a - b) for a, b in zip(pose[:3], rounded[:3]))
        print(f"  registers  : {registers}")
        print(f"  after round-trip: x={rounded[0]:.1f}  y={rounded[1]:.1f}  "
              f"z={rounded[2]:.1f}   (max loss {loss:.2f} mm)")

        if ctx is not None:
            write_pose(ctx, pose)
            report_registers(ctx)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{datetime.now():%Y%m%d_%H%M%S}.png")
    cv2.imwrite(out_path, annotate(color_image, centre, radius, pose))

    return color_image, centre, radius, pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bush", choices=sorted(BUSH_PROFILES),
                        help="bush type, needed for the height correction")
    parser.add_argument("--no-modbus", action="store_true")
    args = parser.parse_args()

    print(f"calibration: {CALIBRATION_ID}")
    print(f"mode       : {'planar' if USE_PLANAR else '3D'}")
    if args.bush:
        profile = BUSH_PROFILES[args.bush]
        print(f"bush       : {args.bush}  height {profile['height_mm']} mm")
    else:
        print("bush       : none, no height correction")

    ctx = None if args.no_modbus else start_server()
    camera = RealSenseCamera()
    window = "SPACE/c = capture and locate,  q = quit"
    cv2.namedWindow(window)

    try:
        # no live preview, every capture() writes ~8 MB to disk
        first = camera.capture()[0]
        display = to_display(first, display_scale(first))

        while True:
            cv2.imshow(window, display)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("c"), 32):
                color_image, centre, radius, pose = locate_once(
                    camera, ctx, bush=args.bush)
                vis = annotate(color_image, centre, radius, pose)
                display = to_display(vis, display_scale(vis))
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
