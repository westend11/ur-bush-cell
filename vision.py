"""Colour image + aligned depth -> bush pose in the robot base frame."""

import numpy as np

from bush_centre import find_centre
from calibration import (deproject, cam_to_base, pixel_to_base_planar,
                         USE_PLANAR, WORK_AREA_PX, BUSH_PROFILES)

# tool pointing straight down; a round bush has no usable yaw
FIXED_ORIENTATION = (0.0, 3.1416, 0.0)

DEPTH_SAMPLE_RADIUS = 3  # px

# real bushes score 35-60 inside the work area, background junk ~9
# TODO: check against an empty table
MIN_CONFIDENCE = 15.0


def _median_depth(depth_m, cx: int, cy: int, r: int = DEPTH_SAMPLE_RADIUS) -> float:
    """Median of the non-zero depths in a (2r+1) window, 0.0 if all are dropouts."""
    h, w = depth_m.shape
    window = depth_m[max(cy - r, 0):min(cy + r + 1, h),
                     max(cx - r, 0):min(cx + r + 1, w)]
    valid = window[window > 0]
    return float(np.median(valid)) if valid.size else 0.0


def locate_bush(color_image, depth_m, intrinsics, bush=None):
    """[x, y, z, rx, ry, rz] in the base frame (mm, rad), or None."""
    x0, y0, x1, y1 = WORK_AREA_PX
    result = find_centre(color_image[y0:y1, x0:x1])
    if result is None:
        return None
    cx, cy, conf = result
    cx, cy = cx + x0, cy + y0

    if conf < MIN_CONFIDENCE:
        return None

    if USE_PLANAR:
        # no bush type -> no height correction, ~19 mm off at the edges
        height = BUSH_PROFILES[bush]["height_mm"] if bush else None
        x, y, z = pixel_to_base_planar(cx, cy, height_mm=height)
        return [x, y, z, *FIXED_ORIENTATION]

    depth = _median_depth(depth_m, int(round(cx)), int(round(cy)))
    if depth <= 0:
        return None

    point_cam = deproject(intrinsics, cx, cy, depth)
    x, y, z = cam_to_base(point_cam) * 1000.0

    return [x, y, z, *FIXED_ORIENTATION]
