"""
Camera -> robot base transform for the bird's-eye pose.

Two options, picked with USE_PLANAR:
  planar - homography for XY and a fitted plane for Z, no depth used.
           Default, since everything sits on one table and the D435 depth
           on this surface is noisy (1-4 mm steps, holes).
  3D     - deproject pixel + depth, then rigid transform to base.

Calibrated 2026-08-25 with 9 TCP-touched points.
planar: 0.40 mm in-sample, 0.92 mm leave-one-out. 3D: 1.90 mm.

Everything here is only valid for the current bird's-eye waypoint, TCP and
camera mount. Change any of them and you have to recalibrate.
"""

import numpy as np
import pyrealsense2 as rs

from planar import apply_homography, apply_z_plane

USE_PLANAR = True

# printed by the tools so it's obvious which calibration is loaded
CALIBRATION_ID = "2026-08-25 planar, 9 points, LOOCV 0.92 mm"


# --- planar (from testing/calibration/fit_planar.py) ---

PIXEL_TO_BASE_H = np.array([
    [-4.5575904423e-01,  9.8028597479e-03,  2.7245140080e+02],
    [ 8.3165586073e-03,  4.3862197442e-01,  6.2149007935e+02],
    [ 1.2270026161e-06, -1.7547707292e-05,  1.0000000000e+00],
])

BASE_Z_PLANE = np.array([-1.0172603029e-04, 4.1759276933e-03, 1.0338623631e+02])

# bounding box of the calibration points (x0, y0, x1, y1) in px.
# Searching the whole frame doesn't work, the detector locks onto the dark
# machinery in the top right.
CALIBRATED_BOX_PX = (465, 151, 1323, 1012)

# padding so bushes on the edge of the box aren't cut off.
# 0 -> 10.4 mm RMS, 40 -> 3.4 mm, 80 -> clutter gets back in and it breaks
SEARCH_PAD_PX = 40
WORK_AREA_PX = (CALIBRATED_BOX_PX[0] - SEARCH_PAD_PX,
                CALIBRATED_BOX_PX[1] - SEARCH_PAD_PX,
                CALIBRATED_BOX_PX[2] + SEARCH_PAD_PX,
                CALIBRATED_BOX_PX[3] + SEARCH_PAD_PX)

# camera optical centre in base frame (mm), from the 3D fit.
# only used for the height correction below
CAMERA_CENTRE_MM = np.array([-132.46, 868.65, 742.80])

# constant bias measured over 15 logged points (testing/locate_log.py),
# brings 2.61 mm RMS down to 1.24 mm. Probably TCP. Re-measure after recalibrating.
SYSTEMATIC_OFFSET_MM = np.array([-2.59, 0.32])

# default height above the calibrated plane when no bush type is given
OBJECT_HEIGHT_MM = 0.0

# measured 2026-08-26
# height_mm: height of the rim the detector sees above the calibrated plane
# radius_px: observed small 47-52 px, big 84-96 px
BUSH_PROFILES = {
    "small": {
        "height_mm": 63.5,
        "insert_z_mm": 162.9,
        "safe_z_mm": 170.0,
        "radius_px": (38, 64),
    },
    "big": {
        # only fitted on 4 points, could use a few more near the corners
        "height_mm": 56.0,
        "insert_z_mm": 158.8,
        "safe_z_mm": 170.0,
        "radius_px": (74, 110),
    },
}

# radius inside the band -> not sure which bush, don't guess
BUSH_RADIUS_SPLIT = 68.0
BUSH_RADIUS_BAND = (62.0, 76.0)


def classify_bush(radius_px):
    """'small', 'big' or None if unclear."""
    if radius_px is None:
        return None
    if BUSH_RADIUS_BAND[0] <= radius_px <= BUSH_RADIUS_BAND[1]:
        return None
    return "small" if radius_px < BUSH_RADIUS_SPLIT else "big"


# --- 3D (from testing/calibration/refit_3d.py) ---

CAM_TO_BASE_R = np.array([
    [-0.99958246,  0.01572095, -0.02424380],
    [ 0.01490976,  0.99933469,  0.03328481],
    [ 0.02475094,  0.03290944, -0.99915182],
])

CAM_TO_BASE_T = np.array([-0.14148189,  0.89909160,  0.74169318])


def deproject(intrinsics, u: float, v: float, depth_m: float) -> np.ndarray:
    """Pixel + depth (m) -> point in camera frame (m)."""
    return np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_m))


def cam_to_base(point_cam: np.ndarray) -> np.ndarray:
    return CAM_TO_BASE_R @ point_cam + CAM_TO_BASE_T


def pixel_to_base_planar(u: float, v: float, height_mm=None) -> np.ndarray:
    """Pixel -> [x, y, z] in base frame (mm).

    With a height the point is moved from the plane up to the top of the
    object, so z is the top face (not the insertion depth).
    """
    x, y = apply_homography(PIXEL_TO_BASE_H, [[u, v]])[0]
    z = apply_z_plane(BASE_Z_PLANE, [[u, v]])[0]
    height = OBJECT_HEIGHT_MM if height_mm is None else height_mm

    if height:
        # parallax: the homography gives where the camera ray hits the plane (P).
        # The top of the bush is on the same ray, h higher, so move back
        # towards the camera C:  T = P - (h/d) * (P - C)
        d = CAMERA_CENTRE_MM[2] - z
        fraction = height / d
        x -= fraction * (x - CAMERA_CENTRE_MM[0])
        y -= fraction * (y - CAMERA_CENTRE_MM[1])
        z += height

    x += SYSTEMATIC_OFFSET_MM[0]
    y += SYSTEMATIC_OFFSET_MM[1]
    return np.array([x, y, z])
