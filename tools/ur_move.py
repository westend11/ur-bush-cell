"""Send raw URScript to the secondary interface (port 30002). Manual use only."""

import socket

HOST, SECONDARY = "192.168.10.220", 30002

# joint angles in rad: base, shoulder, elbow, wrist1, wrist2, wrist3
DEMO_MOVE = "movej([0, -1.57, 0, -1.57, 0, 0], a=1.0, v=0.5)"


def urscript(script: str) -> None:
    # no ack on 30002, a bad script just fails silently on the controller
    with socket.create_connection((HOST, SECONDARY), timeout=5) as s:
        s.sendall((script + "\n").encode())


if __name__ == "__main__":
    print(f"about to send to {HOST}:\n  {DEMO_MOVE}")
    if input("the arm will move. type 'yes' to continue: ").strip().lower() == "yes":
        urscript(DEMO_MOVE)
        print("sent")
    else:
        print("cancelled")
