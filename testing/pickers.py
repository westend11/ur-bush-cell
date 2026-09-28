"""Click helpers. The 1080p frame doesn't fit on screen, so it's shown scaled
and clicks are mapped back, then refined on a zoomed crop."""

import cv2

MAX_DISPLAY = (1500, 820)
ZOOM_HALF = 90
ZOOM_FACTOR = 5


def display_scale(image, max_display=MAX_DISPLAY):
    h, w = image.shape[:2]
    return min(1.0, max_display[0] / w, max_display[1] / h)


def to_display(image, scale):
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def refine_click(image, cx, cy, window="refine: click the exact point  |  r=redo"):
    """Zoomed second click around (cx, cy). Returns (x, y) in full-res image
    coords, or None if 'r' was pressed."""
    h, w = image.shape[:2]
    x0 = max(0, min(cx - ZOOM_HALF, w - 2 * ZOOM_HALF))
    y0 = max(0, min(cy - ZOOM_HALF, h - 2 * ZOOM_HALF))
    crop = image[y0:y0 + 2 * ZOOM_HALF, x0:x0 + 2 * ZOOM_HALF]

    zoom = cv2.resize(crop, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR,
                      interpolation=cv2.INTER_LINEAR)
    result = {}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            result["xy"] = (x0 + x / ZOOM_FACTOR, y0 + y / ZOOM_FACTOR)

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    try:
        while True:
            vis = zoom.copy()
            mid = ZOOM_HALF * ZOOM_FACTOR
            cv2.line(vis, (mid, 0), (mid, vis.shape[0]), (80, 80, 80), 1)
            cv2.line(vis, (0, mid), (vis.shape[1], mid), (80, 80, 80), 1)
            if "xy" in result:
                px = int((result["xy"][0] - x0) * ZOOM_FACTOR)
                py = int((result["xy"][1] - y0) * ZOOM_FACTOR)
                cv2.drawMarker(vis, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.imshow(window, vis)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("r"):
                return None
            if key in (13, 32) and "xy" in result:  # enter / space
                return result["xy"]
    finally:
        cv2.destroyWindow(window)
