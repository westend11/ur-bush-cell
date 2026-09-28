"""Poll the server's registers and print whenever something changes.
Start pose_server_modbus.py (or the demo) first.

usage: python testing/modbus_monitor.py
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ConnectionException

from modbus_io import PORT, UNIT_ID, ARRIVED, POSE_X, POSE_RZ, GO, decode_pose

HOST = "127.0.0.1"
POLL_INTERVAL = 0.2
NAMES = ["ARRIVED", "POSE_READY", "X", "Y", "Z", "RX", "RY", "RZ",
         "SELECT", "STATUS", "DETECTED", "GO"]


def main() -> None:
    client = ModbusTcpClient(HOST, port=PORT)
    if not client.connect():
        print(f"could not connect to {HOST}:{PORT}, is the server running?")
        return

    print(f"polling every {POLL_INTERVAL}s, ctrl-c to stop\n")

    last = None
    try:
        while True:
            try:
                rr = client.read_holding_registers(ARRIVED, count=GO + 1, slave=UNIT_ID)
            except ConnectionException:
                print("server unreachable, retrying...")
                time.sleep(1.0)
                client.connect()
                continue
            if rr.isError():
                print(f"read error: {rr}")
                time.sleep(POLL_INTERVAL)
                continue

            values = rr.registers
            if values != last:
                stamp = f"{datetime.now():%H:%M:%S.%f}"[:-3]
                print(f"[{stamp}] " + "  ".join(f"{n}={v}" for n, v in zip(NAMES, values)))
                pose = decode_pose(values[POSE_X:POSE_RZ + 1])
                print(f"           pose: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} mm  "
                      f"rx={pose[3]:.4f} ry={pose[4]:.4f} rz={pose[5]:.4f} rad")
                last = values

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
