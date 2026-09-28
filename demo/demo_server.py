"""
Demo version of the cell. The pendant runs the robot, the laptop does the
vision and shows what it found. Nothing moves until the operator releases it.

  robot sets ARRIVED=1   -> laptop armed
  'c'                    -> capture, detect, publish pose (repeat to retake)
  SPACE                  -> 3 s countdown, then GO=1
  robot sees GO, reads X/Y, moves, clears ARRIVED

BUSH_SELECT from the pendant: 0 = auto (by radius), 1 = small, 2 = big.

usage:
  python demo/demo_server.py
  python demo/demo_server.py --no-robot    camera + display only
  python demo/demo_server.py --auto        capture on ARRIVED, no keys
"""

import argparse
import os
import queue
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
from pymodbus.server import StartTcpServer

from camera import RealSenseCamera
from vision import locate_bush, MIN_CONFIDENCE
from bush_centre import find_centre, estimate_radius
from calibration import (CALIBRATION_ID, WORK_AREA_PX, BUSH_PROFILES,
                         classify_bush, BUSH_RADIUS_BAND)
from modbus_io import (HOST, PORT, UNIT_ID, build_context, encode_pose,
                       ARRIVED, POSE_READY, POSE_X, BUSH_SELECT, STATUS,
                       BUSH_DETECTED, GO, BUSH_CODES, BUSH_NUMBERS,
                       STATUS_IDLE, STATUS_WORKING, STATUS_OK, STATUS_NO_BUSH,
                       STATUS_LOW_CONF, STATUS_ERROR)

import dashboard

WINDOW = "Bush picking cell"


def start_server(on_arrived):
    ctx = build_context(on_arrived)
    threading.Thread(
        target=StartTcpServer,
        kwargs={"context": ctx, "address": (HOST, PORT)},
        daemon=True,
    ).start()
    return ctx


def read(ctx, addr):
    return ctx[UNIT_ID].getValues(3, addr, count=1)[0]


def write(ctx, addr, value):
    ctx[UNIT_ID].setValues(3, addr, [int(value)])


def publish(ctx, pose, bush):
    # POSE_READY goes last so the robot never sees it over old values
    ctx[UNIT_ID].setValues(3, POSE_X, encode_pose(pose))
    write(ctx, BUSH_DETECTED, BUSH_NUMBERS[bush])
    write(ctx, STATUS, STATUS_OK)
    write(ctx, POSE_READY, 1)


def log(state, msg):
    state["history"].append(f"{datetime.now():%H:%M:%S} {msg}")


def run_vision(camera, ctx, state):
    """Capture -> detect -> pose. Returns (annotated image, pose, bush)."""
    started = time.time()
    write(ctx, STATUS, STATUS_WORKING)

    colour, depth_map, intrinsics, filename = camera.capture()
    x0, y0, x1, y1 = WORK_AREA_PX
    crop = colour[y0:y1, x0:x1]

    detection = find_centre(crop)
    if detection is None:
        write(ctx, STATUS, STATUS_NO_BUSH)
        log(state, "no detection")
        return dashboard.annotate_capture(colour, None, None, WORK_AREA_PX,
                                          None, 0.0), None, None
    cx, cy, confidence = detection
    radius = estimate_radius(crop, cx, cy)
    centre = (cx + x0, cy + y0)

    requested = read(ctx, BUSH_SELECT)
    state["select_mode"] = requested
    if requested in BUSH_CODES:
        bush = BUSH_CODES[requested]
        low, high = BUSH_PROFILES[bush]["radius_px"]
        if radius and not low <= radius <= high:
            log(state, f"r={radius:.0f}px contradicts '{bush}'")
    else:
        bush = classify_bush(radius)
        if bush is None:
            write(ctx, STATUS, STATUS_LOW_CONF)
            log(state, f"r={radius or 0:.0f}px ambiguous "
                       f"({BUSH_RADIUS_BAND[0]:.0f}-{BUSH_RADIUS_BAND[1]:.0f})")
            return dashboard.annotate_capture(colour, centre, radius,
                                              WORK_AREA_PX, None,
                                              confidence), None, None

    if confidence < MIN_CONFIDENCE:
        write(ctx, STATUS, STATUS_LOW_CONF)
        log(state, f"conf {confidence:.0f} below {MIN_CONFIDENCE:.0f}")
        return dashboard.annotate_capture(colour, centre, radius, WORK_AREA_PX,
                                          bush, confidence), None, bush

    pose = locate_bush(colour, depth_map, intrinsics, bush=bush)
    if pose is None:
        write(ctx, STATUS, STATUS_NO_BUSH)
        return dashboard.annotate_capture(colour, centre, radius, WORK_AREA_PX,
                                          bush, confidence), None, bush

    publish(ctx, pose, bush)
    state["elapsed_ms"] = (time.time() - started) * 1000.0
    state["cycles"] += 1
    log(state, f"{bush:<5} ({pose[0]:7.1f},{pose[1]:7.1f}) {state['elapsed_ms']:.0f}ms")
    return (dashboard.annotate_capture(colour, centre, radius, WORK_AREA_PX,
                                       bush, confidence), pose, bush)


class OfflineCtx:
    """Fake register store for --no-robot."""

    def __init__(self):
        self._values = {}

    def __getitem__(self, _unit):
        return self

    def getValues(self, _fc, addr, count=1):
        return [self._values.get(addr + i, 0) for i in range(count)]

    def setValues(self, _fc, addr, values):
        for i, value in enumerate(values):
            self._values[addr + i] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-robot", action="store_true",
                        help="skip the Modbus server; capture with 'c' only")
    parser.add_argument("--auto", action="store_true",
                        help="capture on ARRIVED and release without SPACE")
    parser.add_argument("--countdown", type=float, default=3.0,
                        help="seconds between SPACE and GO")
    args = parser.parse_args()

    state = {"calibration": CALIBRATION_ID, "status": STATUS_IDLE,
             "cycles": 0, "elapsed_ms": 0.0, "history": [], "registers": {},
             "pose": None, "bush": None, "select_mode": 0,
             "armed": False, "countdown": None}

    # the modbus callback only queues a token, all the work stays on this thread
    triggers = queue.Queue()
    ctx = None if args.no_robot else start_server(
        lambda: triggers.put_nowait(True))
    if ctx is not None:
        write(ctx, STATUS, STATUS_IDLE)
        print(f"serving on {HOST}:{PORT} (unit {UNIT_ID}) -- waiting for ARRIVED")
    print(f"calibration: {CALIBRATION_ID}")

    camera = RealSenseCamera()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1600, 900)
    capture_vis = None
    release_at = None
    offline = OfflineCtx()

    try:
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break

            arrived_edge = False
            try:
                triggers.get_nowait()
                arrived_edge = True
                state["armed"] = True
                log(state, "robot at capture position")
            except queue.Empty:
                pass

            triggered = key == ord("c") or (arrived_edge and args.auto)

            # only release if there is actually a pose published
            if key == 32:
                if ctx is not None and read(ctx, POSE_READY) == 1:
                    release_at = time.time() + args.countdown
                    log(state, f"released in {args.countdown:.0f}s")
                else:
                    log(state, "SPACE ignored: no pose yet")

            if release_at is not None:
                remaining = release_at - time.time()
                state["countdown"] = max(remaining, 0.0)
                if remaining <= 0:
                    release_at = None
                    state["countdown"] = None
                    if ctx is not None:
                        write(ctx, GO, 1)
                    log(state, "GO")

            if triggered:
                try:
                    capture_vis, pose, bush = run_vision(camera, ctx or offline, state)
                    state["pose"] = pose
                    state["bush"] = bush
                    if args.auto and pose is not None and ctx is not None:
                        write(ctx, GO, 1)
                except Exception as exc:
                    if ctx is not None:
                        write(ctx, STATUS, STATUS_ERROR)
                    log(state, f"ERROR {exc!r}"[:60])
                    print(f"vision error: {exc!r}")

            if ctx is not None:
                state["registers"] = {
                    addr: read(ctx, addr)
                    for addr in (ARRIVED, POSE_READY, POSE_X, POSE_X + 1,
                                 BUSH_SELECT, STATUS, BUSH_DETECTED, GO)}
                state["status"] = state["registers"].get(STATUS, 0)

                # robot dropped ARRIVED -> it's done with the pose, reset.
                # The PC clears these because a UR register output keeps
                # re-sending its value and would overwrite us.
                regs = state["registers"]
                if regs[ARRIVED] == 0 and (regs[POSE_READY] == 1 or regs[GO] == 1):
                    write(ctx, POSE_READY, 0)
                    write(ctx, GO, 0)
                    write(ctx, STATUS, STATUS_IDLE)
                    state["armed"] = False
                    release_at = None
                    state["countdown"] = None

            cv2.imshow(WINDOW, dashboard.render(capture_vis, state))
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
