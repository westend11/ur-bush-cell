"""Pixel -> base frame mapping for a fixed camera looking at a flat table.

XY comes from a homography, Z from a fitted (tilted) plane. Note that a
homography maps to a single plane, so anything above it needs the height
correction in calibration.py.
"""

import numpy as np


def _normalise(points):
    # Hartley normalisation, centroid at origin and mean distance sqrt(2)
    mean = points.mean(axis=0)
    shifted = points - mean
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array([[s, 0, -s * mean[0]],
                  [0, s, -s * mean[1]],
                  [0, 0, 1.0]])
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalised = (T @ homogeneous.T).T
    return T, normalised[:, :2] / normalised[:, 2:3]


def estimate_homography(pixels, world_xy):
    """Normalised DLT, pixels -> world XY (mm)."""
    if len(pixels) < 4:
        raise ValueError("a homography needs at least 4 points")

    T_px, px_n = _normalise(np.asarray(pixels, float))
    T_w, w_n = _normalise(np.asarray(world_xy, float))

    rows = []
    for (x, y), (X, Y) in zip(px_n, w_n):
        rows.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        rows.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    _, _, Vt = np.linalg.svd(np.asarray(rows, float))

    H = np.linalg.inv(T_w) @ Vt[-1].reshape(3, 3) @ T_px
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def apply_homography(H, pixels):
    pixels = np.atleast_2d(np.asarray(pixels, float))
    projected = np.column_stack([pixels, np.ones(len(pixels))]) @ H.T
    return projected[:, :2] / projected[:, 2:3]


def invert_homography(H, world_xy):
    world_xy = np.atleast_2d(np.asarray(world_xy, float))
    projected = np.column_stack([world_xy, np.ones(len(world_xy))]) @ np.linalg.inv(H).T
    return projected[:, :2] / projected[:, 2:3]


def estimate_z_plane(pixels, z_mm):
    """Least squares fit of z = a*u + b*v + c, returns (a, b, c)."""
    pixels = np.asarray(pixels, float)
    A = np.column_stack([pixels[:, 0], pixels[:, 1], np.ones(len(pixels))])
    coefficients, *_ = np.linalg.lstsq(A, np.asarray(z_mm, float), rcond=None)
    return coefficients


def apply_z_plane(coefficients, pixels):
    pixels = np.atleast_2d(np.asarray(pixels, float))
    a, b, c = coefficients
    return a * pixels[:, 0] + b * pixels[:, 1] + c


def ransac_homography(pixels, world_xy, threshold_mm=2.0,
                      max_iters=2000, confidence=0.99, random_state=None):
    """Returns (H, inlier_mask)."""
    pixels = np.asarray(pixels, float)
    world_xy = np.asarray(world_xy, float)
    n = len(pixels)
    if n < 4:
        raise ValueError("a homography needs at least 4 points")

    rng = np.random.default_rng(random_state)
    best_mask = np.zeros(n, bool)
    best_count = 0
    iters_needed = max_iters
    i = 0

    while i < iters_needed and i < max_iters:
        i += 1
        try:
            sample = rng.choice(n, size=4, replace=False)
            H = estimate_homography(pixels[sample], world_xy[sample])
            errors = np.linalg.norm(apply_homography(H, pixels) - world_xy, axis=1)
        except (ValueError, np.linalg.LinAlgError):
            continue

        mask = errors <= threshold_mm
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
            ratio = min(max(count / n, 1e-6), 1 - 1e-6)
            denominator = 1.0 - ratio ** 4
            if denominator > 1e-12:
                iters_needed = min(max_iters, int(np.ceil(
                    np.log(1 - confidence) / np.log(denominator))))
        if count == n:
            break

    if best_count < 4:
        return estimate_homography(pixels, world_xy), np.ones(n, bool)

    # refit on all inliers
    return estimate_homography(pixels[best_mask], world_xy[best_mask]), best_mask


def loocv_error(pixels, world_xy, fit=estimate_homography, apply=apply_homography):
    """Leave-one-out error per point. More honest than the in-sample RMS."""
    pixels = np.asarray(pixels, float)
    world_xy = np.asarray(world_xy, float)
    errors = []
    for i in range(len(pixels)):
        keep = np.arange(len(pixels)) != i
        model = fit(pixels[keep], world_xy[keep])
        errors.append(np.linalg.norm(apply(model, pixels[i:i + 1])[0] - world_xy[i]))
    return np.array(errors)
