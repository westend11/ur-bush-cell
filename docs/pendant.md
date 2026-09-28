# Pendant setup

## MODBUS client

Installation > Fieldbus > MODBUS client, add a unit at `192.168.10.200`.
Enable sequential mode under advanced options.

| Addr | Name | Type | |
|---:|---|---|---|
| 0 | `mb_arrived` | Register Output | 1 = at bird's-eye pose |
| 1 | `mb_pose_ready` | Register Input | set/cleared by the PC |
| 2 | `mb_pose_x` | Register Input | mm x 10, signed |
| 3 | `mb_pose_y` | Register Input | mm x 10, signed |
| 4 | `mb_pose_z` | Register Input | top face of the bush, not used for motion |
| 5-7 | `mb_pose_rx/ry/rz` | Register Input | rad x 10000 |
| 8 | `mb_bush_select` | Register Output | 0 auto, 1 small, 2 big |
| 9 | `mb_status` | Register Input | see below |
| 10 | `mb_bush_detected` | Register Input | 1 small, 2 big |
| 11 | `mb_go` | Register Input | 1 = operator released the robot |

The 3xxxx addresses (e.g. 30002) also work.

Each register is written by only one side. A UR register output keeps
re-sending its value, so if `mb_pose_ready` were an output the robot would
keep overwriting the PC. The PC clears `mb_pose_ready` and `mb_go` when it
sees `mb_arrived` go back to 0.

### Status

| | |
|---:|---|
| 0 | idle |
| 1 | working |
| 2 | pose ok |
| 3 | no bush found |
| 4 | low confidence / unclear size |
| 5 | vision error |

## Reading the pose

Registers are unsigned, and X is negative everywhere in the cell, so unwrap:

```
raw_x = mb_pose_x
if raw_x > 32767:
    raw_x = raw_x - 65536
end
x = raw_x / 10000.0     # metres
```

Same for Y.

## Depths

Z comes from program constants, not from vision:

| Bush | insert_z | safe_z |
|---|---:|---:|
| small | 0.1629 | 0.1750 |
| big | 0.1588 | 0.1700 |

Orientation is taken from the TCP pose at the bird's-eye waypoint.

## Program

```
BeforeStart
    mb_bush_select = bush_mode
    mb_arrived = 0          # otherwise the next 0->1 edge is missed
    MoveJ Home

Loop
    MoveJ Home
    MoveJ BirdsEye
    Wait 0.3
    pos_snap := get_actual_tcp_pose()
    mb_arrived = 1

    Loop mb_go != 1
        If mb_status >= 3: Popup "Vision: check the laptop"
        Wait 0.1

    x, y := unwrapped mb_pose_x / mb_pose_y
    pick z_safe / z_ins from mb_bush_detected

    MoveL p[x, y, z_safe, pos_snap[3], pos_snap[4], pos_snap[5]]
    MoveL p[x, y, z_ins,  pos_snap[3], pos_snap[4], pos_snap[5]]   # slow
    # insert
    MoveL p[x, y, z_safe, ...]

    mb_arrived = 0
    Wait until mb_pose_ready == 0
```

Use MoveL after the capture pose, MoveJ can pick a different IK solution and
flip the wrist. Keep the last approach under 50 mm/s.

## Troubleshooting

| Symptom | Fix |
|---|---|
| grey signal dots, reconnects climbing | start the PC server first |
| E2 / illegal address | check the addresses above |
| huge X | sign not unwrapped |
| off by x1000 | divide by 10000 for metres |
| nothing happens after `mb_arrived = 1` | it was left at 1, clear it in BeforeStart |
| `mb_pose_ready` never goes high | it's set as an output, make it an input |
| robot waits at capture pose | nobody pressed SPACE on the laptop |
| off by a few mm near the edges | wrong bush type |
| everything consistently wrong | waypoint or camera moved, recalibrate |
