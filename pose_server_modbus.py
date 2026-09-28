"""
Modbus TCP server + vision loop for the bush-picking cell.

The robot moves to the bird's-eye pose and sets ARRIVED=1. We capture a
frame, locate the bush, write the pose and raise POSE_READY. All motion is
in the pendant program. Register map is in modbus_io.py.
"""

import queue
import threading

from pymodbus.server import StartTcpServer

from modbus_io import HOST, PORT, UNIT_ID, build_context, write_pose
from camera import RealSenseCamera
from vision import locate_bush

_arrivals = queue.Queue()


def on_arrived():
    # called from the modbus thread, so just queue it
    _arrivals.put_nowait(True)


def worker(camera: RealSenseCamera, ctx) -> None:
    while True:
        _arrivals.get()
        try:
            color_image, depth_map, intrinsics, filename = camera.capture()
            print(f"captured {filename}")

            pose = locate_bush(color_image, depth_map, intrinsics)
            if pose is None:
                print("no bush found in frame, skipping")
                continue

            write_pose(ctx, pose)
            print(f"pose sent: {pose}")
        except Exception as exc:
            print(f"vision pipeline error: {exc!r}")


def main() -> None:
    camera = RealSenseCamera()
    ctx = build_context(on_arrived)

    threading.Thread(target=worker, args=(camera, ctx), daemon=True).start()

    print(f"serving unit {UNIT_ID} on {HOST}:{PORT}, waiting for ARRIVED signal")
    StartTcpServer(context=ctx, address=(HOST, PORT))


if __name__ == "__main__":
    main()
