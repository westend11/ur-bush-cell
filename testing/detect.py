"""Run the detector on saved captures and write annotated copies.

usage: python testing/detect.py ["captures/*.png"]
"""

import glob
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import cv2

from bush_centre import find_centre, estimate_radius
from calibration import WORK_AREA_PX
from vision import MIN_CONFIDENCE

OUT_DIR = os.path.join(ROOT, "testing", "results", "detect")
DEFAULT_PATTERN = os.path.join(ROOT, "captures", "*.png")


def main(pattern: str) -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files match: {pattern}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    x0, y0, x1, y1 = WORK_AREA_PX
    for f in files:
        im = cv2.imread(f)
        n = os.path.basename(f)
        if im is None:
            print(f"{n:24s} COULD NOT READ")
            continue

        # same crop as locate_bush()
        crop = im[y0:y1, x0:x1]
        res = find_centre(crop)
        if res is None:
            print(f"{n:24s} NO PART")
            continue
        cx, cy, conf = res
        R = estimate_radius(crop, cx, cy)
        cx, cy = cx + x0, cy + y0

        weak = "  WEAK" if conf < MIN_CONFIDENCE else ""
        vis = im.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)
        if R is not None:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(R)), (0, 255, 255), 3)
        cv2.drawMarker(vis, (int(round(cx)), int(round(cy))), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(vis, f"({cx:.0f}, {cy:.0f})  conf {conf:.0f}",
                    (int(round(cx)) - 120, int(round(cy)) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 3)
        cv2.imwrite(os.path.join(OUT_DIR, n), vis)

        rstr = f"R={R:6.1f}" if R is not None else "R=  NONE"
        print(f"{n:24s} c=({cx:7.2f},{cy:7.2f}) {rstr} conf={conf:6.1f}{weak}")

    print(f"\nwritten to {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATTERN)
