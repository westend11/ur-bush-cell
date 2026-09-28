"""OpenCV display for the demo: annotated capture on the left, pose,
registers and handshake state on the right."""

import cv2
import numpy as np

PANEL_W = 560
BG = (24, 24, 28)
FG = (235, 235, 235)
DIM = (150, 150, 155)
OK = (120, 220, 130)
WARN = (80, 200, 250)
BAD = (90, 90, 240)
ACCENT = (250, 200, 90)

FONT = cv2.FONT_HERSHEY_SIMPLEX

STATUS_TEXT = {
    0: ("IDLE", DIM),
    1: ("WORKING", WARN),
    2: ("POSE SENT", OK),
    3: ("NO BUSH FOUND", BAD),
    4: ("LOW CONFIDENCE", BAD),
    5: ("VISION ERROR", BAD),
}


def annotate_capture(image, centre, radius, work_area, bush, confidence):
    vis = image.copy()
    x0, y0, x1, y1 = work_area
    cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 160, 60), 2)

    if centre is not None:
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        if radius:
            cv2.circle(vis, (cx, cy), int(round(radius)), (60, 230, 240), 3)
        cv2.drawMarker(vis, (cx, cy), (60, 60, 240), cv2.MARKER_CROSS, 46, 3)
        label = f"{bush or '?'}  r={radius or 0:.0f}px  conf={confidence:.0f}"
        cv2.putText(vis, label, (cx - 150, cy - 40), FONT, 1.0, (240, 120, 240), 3)
    else:
        cv2.putText(vis, "NO DETECTION", (x0 + 30, y0 + 80), FONT, 2.0, BAD, 4)
    return vis


def _text(canvas, x, y, text, colour=FG, scale=0.62, thick=1):
    cv2.putText(canvas, text, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def _panel(height, state):
    panel = np.full((height, PANEL_W, 3), BG, np.uint8)
    y = 46
    _text(panel, 24, y, "BUSH PICKING CELL", ACCENT, 0.86, 2); y += 40
    _text(panel, 24, y, state.get("calibration", ""), DIM, 0.5); y += 34

    countdown = state.get("countdown")
    if countdown is not None:
        cv2.rectangle(panel, (20, y - 26), (PANEL_W - 20, y + 12), (30, 30, 70), -1)
        _text(panel, 32, y, f"RELEASING IN {countdown:.1f}s", BAD, 0.9, 2)
    else:
        label, colour = STATUS_TEXT.get(state.get("status", 0), ("?", DIM))
        if state.get("armed") and state.get("status") == 0:
            label, colour = "ARMED - press c", WARN
        cv2.rectangle(panel, (20, y - 26), (PANEL_W - 20, y + 12), (40, 40, 46), -1)
        _text(panel, 32, y, label, colour, 0.9, 2)
    y += 52

    mode = "AUTO" if state.get("select_mode") == 0 else "MANUAL"
    _text(panel, 24, y, f"bush     {state.get('bush') or '--'}   ({mode})",
          FG, 0.68); y += 30
    _text(panel, 24, y, f"cycle    {state.get('cycles', 0)}"
                        f"    {state.get('elapsed_ms', 0):.0f} ms", DIM); y += 40

    _text(panel, 24, y, "PUBLISHED POSE", ACCENT, 0.6); y += 30
    pose = state.get("pose")
    if pose:
        _text(panel, 32, y, f"X  {pose[0]:9.1f} mm", FG, 0.72); y += 28
        _text(panel, 32, y, f"Y  {pose[1]:9.1f} mm", FG, 0.72); y += 28
        _text(panel, 32, y, f"Z  {pose[2]:9.1f} mm  (top face)", DIM, 0.6); y += 34
    else:
        _text(panel, 32, y, "-- none --", DIM); y += 34

    # raw values next to the decoded mm, handy for spotting sign/scale bugs on the pendant
    _text(panel, 24, y, "REGISTERS (what the robot reads)", ACCENT, 0.6); y += 28
    for addr, name in ((2, "30002 X"), (3, "30003 Y")):
        raw = state.get("registers", {}).get(addr)
        if raw is None:
            continue
        signed = raw - 65536 if raw > 32767 else raw
        note = "  (unwrap!)" if raw > 32767 else ""
        _text(panel, 32, y, f"{name}  {raw:>6}  ->{signed:>7} ->{signed/10:8.1f} mm"
                            f"{note}", DIM, 0.52); y += 24
    y += 14

    _text(panel, 24, y, "HANDSHAKE", ACCENT, 0.6); y += 28
    for name, addr in (("ARRIVED", 0), ("POSE_READY", 1), ("GO", 11),
                       ("BUSH_SELECT", 8), ("STATUS", 9), ("BUSH_DETECTED", 10)):
        value = state.get("registers", {}).get(addr, 0)
        colour = OK if value else DIM
        _text(panel, 32, y, f"{name:<14}{value}", colour, 0.55); y += 24

    y += 16
    _text(panel, 24, y, "HISTORY", ACCENT, 0.6); y += 26
    for line in state.get("history", [])[-7:]:
        _text(panel, 32, y, line[:58], DIM, 0.48); y += 22

    _text(panel, 24, height - 22,
          "c = capture / recapture    SPACE = release    q = quit", DIM, 0.5)
    return panel


def render(capture_vis, state, max_height=900):
    if capture_vis is None:
        capture_vis = np.full((720, 1280, 3), BG, np.uint8)
        _text(capture_vis, 60, 360, "waiting for first capture...", DIM, 1.0, 2)

    h, w = capture_vis.shape[:2]
    scale = min(max_height / h, (1600 - PANEL_W) / w, 1.0)
    left = cv2.resize(capture_vis, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)

    # panel needs ~760 px, pad the image if it's shorter
    height = max(left.shape[0], 760)
    if left.shape[0] < height:
        pad = np.full((height - left.shape[0], left.shape[1], 3), BG, np.uint8)
        left = np.vstack([left, pad])

    return np.hstack([left, _panel(height, state)])
