"""Power on the arm and release the brakes through the dashboard server."""

import socket
import time


class Dashboard:
    def __init__(self, host="192.168.10.220", port=29999, timeout=10):
        self.s = socket.create_connection((host, port), timeout=timeout)
        self.s.recv(4096)  # greeting

    def cmd(self, c: str) -> str:
        self.s.sendall((c + "\n").encode())
        return self.s.recv(4096).decode().strip()

    def close(self):
        self.s.close()


if __name__ == "__main__":
    d = Dashboard()
    print(d.cmd("power on"))
    print(d.cmd("brake release"))

    for _ in range(30):
        mode = d.cmd("robotmode")
        print(mode)
        if "RUNNING" in mode:
            break
        time.sleep(1)
    d.close()
