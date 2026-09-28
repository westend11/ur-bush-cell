# UR bush picking cell

A UR arm moves to a fixed bird's-eye pose and signals the PC over Modbus TCP.
The PC grabs a RealSense frame, finds the bush and sends back its position in
the robot base frame. All motion is done by the pendant program.

```
robot:  home -> bird's-eye pose -> ARRIVED=1
PC:     capture -> locate_bush -> write pose -> POSE_READY=1
robot:  read pose, move, clear ARRIVED
```

## Running

```
pip install -r requirements.txt
python pose_server_modbus.py      # headless version
python demo/demo_server.py        # with display and operator release (c / SPACE)
```

Start the PC side first, then the pendant program. Register map is in
`modbus_io.py`, pendant setup in [docs/pendant.md](docs/pendant.md).

## Files

| | |
|---|---|
| `pose_server_modbus.py` | Modbus server + capture/vision loop |
| `demo/` | demo server with an OpenCV dashboard |
| `modbus_io.py` | register map, pose encoding, ARRIVED handshake |
| `camera.py` | RealSense capture (1080p colour, aligned depth) |
| `bush_centre.py` | bush detection using radial symmetry |
| `vision.py` | detection + calibration -> base frame pose |
| `calibration.py` | calibration constants and pixel -> base transform |
| `planar.py` | homography / plane fitting |
| `testing/` | detection, locating and calibration scripts |
| `tools/` | dashboard power-on and raw URScript helpers |

## Calibration

1. Set the final TCP first.
2. Move to the bird's-eye pose.
3. `python testing/calibration/calibrate.py` - click points, touch them with the TCP.
4. `python testing/calibration/fit_planar.py` and copy the result into `calibration.py`.
5. `python testing/calibration/validate.py` on a few new points.

Accuracy right now is about 1 mm (planar, leave-one-out). The calibration is
only valid for the current bird's-eye waypoint, TCP and camera mount; if any
of them change the poses will be wrong without any error.

## Network

PC `192.168.10.200`, robot `192.168.10.220`, mask `255.255.255.0`.
The server listens on `0.0.0.0:502` and answers on any unit id.
