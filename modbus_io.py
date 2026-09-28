"""
Modbus TCP register map and handshake.

Holding registers (set these up on the pendant as MODBUS client signals):

  0  ARRIVED        robot -> PC  1 = at bird's-eye pose. Robot clears it.
  1  POSE_READY     PC -> robot  1 = pose below is fresh
  2-7 POSE_X..RZ    PC -> robot  signed 16 bit, mm*10 and rad*10000
  8  BUSH_SELECT    robot -> PC  0 auto, 1 small, 2 big
  9  STATUS         PC -> robot  see STATUS_*
  10 BUSH_DETECTED  PC -> robot  1 small, 2 big
  11 GO             PC -> robot  1 = operator released the robot
"""

from pymodbus.datastore import (
    ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext,
)
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder
from pymodbus.constants import Endian

HOST = "0.0.0.0"
PORT = 502
UNIT_ID = 128

ARRIVED    = 0
POSE_READY = 1
POSE_X, POSE_Y, POSE_Z, POSE_RX, POSE_RY, POSE_RZ = range(2, 8)

BUSH_SELECT   = 8
STATUS        = 9
BUSH_DETECTED = 10
GO            = 11

STATUS_IDLE     = 0
STATUS_WORKING  = 1
STATUS_OK       = 2
STATUS_NO_BUSH  = 3
STATUS_LOW_CONF = 4  # weak detection or ambiguous size
STATUS_ERROR    = 5

BUSH_CODES = {1: "small", 2: "big"}
BUSH_NUMBERS = {"small": 1, "big": 2}

POS_SCALE = 10       # 0.1 mm
ROT_SCALE = 10000    # 1e-4 rad
_SCALES = [POS_SCALE, POS_SCALE, POS_SCALE, ROT_SCALE, ROT_SCALE, ROT_SCALE]


def encode_pose(pose):
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG)
    for value, scale in zip(pose, _SCALES):
        builder.add_16bit_int(round(value * scale))
    return builder.to_registers()


def decode_pose(registers):
    decoder = BinaryPayloadDecoder.fromRegisters(registers, byteorder=Endian.BIG)
    return [decoder.decode_16bit_int() / scale for scale in _SCALES]


REFLECT_SPAN = 10000  # UR also accepts x as 10000+x, 20000+x, ...


class WatchedBlock(ModbusSequentialDataBlock):
    """Data block that calls on_write(addr, values) when the robot writes,
    and maps UR's reflected addresses (e.g. 30002) back to the base address."""

    def __init__(self, address, values, on_write=None):
        super().__init__(address, values)
        self.on_write = on_write

    @staticmethod
    def _reflect(address):
        # address comes in with the slave context's +1 already added
        protocol_addr = address - 1
        if protocol_addr >= REFLECT_SPAN:
            protocol_addr %= REFLECT_SPAN
        return protocol_addr + 1

    def validate(self, address, count=1):
        return super().validate(self._reflect(address), count)

    def getValues(self, address, count=1):
        return super().getValues(self._reflect(address), count)

    def setValues(self, address, values):
        address = self._reflect(address)
        super().setValues(address, values)
        if self.on_write:
            self.on_write(address - 1, values)


def build_context(on_arrived):
    """on_arrived() fires on the 0 -> 1 edge of ARRIVED only, since the
    pendant keeps re-sending the same value."""
    state = {"arrived": 0}

    def _on_write(protocol_addr, values):
        if protocol_addr != ARRIVED or not values:
            return
        value, previous = values[0], state["arrived"]
        state["arrived"] = value
        if value == 1 and previous != 1:
            on_arrived()

    block = WatchedBlock(0, [0] * 100, on_write=_on_write)
    store = ModbusSlaveContext(hr=block, ir=block)
    # single=True answers on any unit id (polyscope defaults to 255)
    return ModbusServerContext(slaves=store, single=True)


def write_pose(ctx, pose):
    """Write the pose, then raise POSE_READY. ARRIVED is left to the robot."""
    ctx[UNIT_ID].setValues(3, POSE_X, encode_pose(pose))
    ctx[UNIT_ID].setValues(3, POSE_READY, [1])
