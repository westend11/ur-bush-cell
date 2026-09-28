"""Bush centre detection with fast radial symmetry.

usage: python bush_centre.py "captures/*.png"
"""

import sys, glob, os
import cv2
import numpy as np


def radial_symmetry(g: np.ndarray,
                    rmin: int,
                    rmax: int,
                    n_radii: int = 12,
                    grad_pct: float = 92.0) -> np.ndarray:
    H, W = g.shape
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ys, xs = np.nonzero(mag > np.percentile(mag, grad_pct))
    if len(xs) < 32:
        return np.zeros((H, W), np.float32)
    ux = gx[ys, xs] / (mag[ys, xs] + 1e-6)
    uy = gy[ys, xs] / (mag[ys, xs] + 1e-6)

    acc = np.zeros((H, W), np.float32)
    for n in range(rmin, rmax + 1, max(1, (rmax - rmin) // n_radii)):
        O = np.zeros((H, W), np.float32)
        px = (xs - n * ux).astype(int)
        py = (ys - n * uy).astype(int)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        np.add.at(O, (py[ok], px[ok]), 1.0)
        acc += cv2.GaussianBlur(O, (0, 0), n * 0.3 + 1)
    return acc


def _subpixel(acc: np.ndarray, x: int, y: int, w: int = 5) -> tuple[float, float]:
    H, W = acc.shape
    y0, y1 = max(y - w, 0), min(y + w + 1, H)
    x0, x1 = max(x - w, 0), min(x + w + 1, W)
    P = acc[y0:y1, x0:x1].astype(np.float64)
    P = P - P.min()
    if P.sum() <= 0:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((P * xx).sum() / P.sum()), float((P * yy).sum() / P.sum())


def find_centre(im: np.ndarray,
                r_frac: tuple[float, float] = (0.012, 0.075),
                grad_pct: float = 92.0,
                margin: int = 12,
                min_conf: float = 0.0) -> tuple[float, float, float] | None:
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    H, W = g.shape
    diag = np.hypot(H, W)
    rmin = max(4, int(r_frac[0] * diag))
    rmax = max(rmin + 2, int(r_frac[1] * diag))

    acc = radial_symmetry(g, rmin, rmax, grad_pct=grad_pct)

    dark = cv2.GaussianBlur(g, (0, 0), rmax * 0.6)
    acc *= np.clip((np.percentile(g, 60) - dark) / max(g.std(), 1.0), 0, None)

    if margin > 0:
        acc[:margin] = 0
        acc[-margin:] = 0
        acc[:, :margin] = 0
        acc[:, -margin:] = 0

    if not np.isfinite(acc).any() or acc.max() <= 0:
        return None

    _, peak, _, (x, y) = cv2.minMaxLoc(acc)
    conf = float(peak / (acc.mean() + 1e-9))
    if conf < min_conf:
        return None
    cx, cy = _subpixel(acc, x, y)
    return cx, cy, conf


def estimate_radius(im: np.ndarray,
                    cx: float,
                    cy: float,
                    r_frac: tuple[float, float] = (0.03, 0.22),
                    n_th: int = 720,
                    trim: float = 0.25) -> float | None:
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    diag = np.hypot(*g.shape)
    rr = np.arange(max(r_frac[0] * diag, 3.0), r_frac[1] * diag, 0.5)
    if len(rr) < 5:
        return None
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    X = (cx + np.outer(rr, np.cos(th))).astype(np.float32)
    Y = (cy + np.outer(rr, np.sin(th))).astype(np.float32)
    P = cv2.remap(g, X, Y, cv2.INTER_LINEAR)

    prof = np.median(P, axis=1)
    sm = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 2.0).ravel()
    d = np.gradient(sm)
    if not np.isfinite(d).any() or d.max() <= 0:
        return None
    seed = rr[int(d.argmax())]

    band = (rr > seed * 0.75) & (rr < seed * 1.25)
    if band.sum() < 5:
        return float(seed)
    Pb, rb = P[band], rr[band]
    lo_r = np.percentile(Pb[:max(len(rb) // 4, 1)], 20, axis=0)
    hi_r = np.percentile(Pb[-max(len(rb) // 4, 1):], 80, axis=0)
    above = Pb >= lo_r + 0.5 * (hi_r - lo_r)
    rh = rb[np.where(above.any(0), above.argmax(0), len(rb) - 1)]
    ok = (hi_r - lo_r > 8) & (np.abs(rh - np.median(rh)) < trim * np.median(rh))
    if ok.sum() < 30:
        return float(seed)
    return float(np.median(rh[ok]))


def main(pattern: str) -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files match: {pattern}")
        return

    os.makedirs("out_centre", exist_ok=True)
    for f in files:
        im = cv2.imread(f)
        n = os.path.basename(f)
        if im is None:
            print(f"{n:24s} COULD NOT READ")
            continue

        res = find_centre(im)
        if res is None:
            print(f"{n:24s} NO PART")
            continue
        cx, cy, conf = res
        R = estimate_radius(im, cx, cy)

        vis = im.copy()
        if R is not None:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(R)), (0, 255, 255), 2)
        cv2.drawMarker(vis, (int(round(cx)), int(round(cy))), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis, f"({cx:.0f}, {cy:.0f})",
                    (int(round(cx)) - 40, int(round(cy)) - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.imwrite(f"out_centre/{n}", vis)

        rstr = f"R={R:6.1f}" if R is not None else "R=  NONE"
        print(f"{n:24s} c=({cx:7.2f},{cy:7.2f}) {rstr} conf={conf:6.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "*.png")
