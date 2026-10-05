"""Standalone RS232 check for the KDC-24V60W-4T LED controller: send raw
command strings (e.g. SA0200#, SA0255#) and print what comes back.

Uses the same serial settings as core/led/serial_led_client.py
(8 data bits, no parity, 1 stop bit, no flow control) and reads the reply
until the "!" ack arrives or the timeout expires.

Run directly (no project imports needed):

    python development_files/led_serial_script.py                    # sends COMMANDS below
    python development_files/led_serial_script.py SA0200# SB0100#    # sends these instead
"""

from __future__ import annotations

import sys
import time

import serial

PORT = "COM3"
BAUD_RATE = 19200
TIMEOUT_S = 1.0
DELAY_BETWEEN_S = 0.5

# COMMANDS = ["SA0200#","SB0200#","SC0200#","SD0200#"]
COMMANDS = ["SA0150#","SB0150#","SC0150#","SD0150#"]
# COMMANDS = ["SA0000#","SB0000#","SC0000#","SD0000#"]
# COMMANDS = ["SB0000#","SB0200#","SC0200#","SD0200#","SA0000#","SB0000#","SC0000#","SD0000#"]*100
# COMMANDS = ["SC0000#SB0000#","S200T200T200T200TC#"]

ACK = b"!"


def read_response(ser: serial.Serial) -> bytes:
    """Collect bytes until the ack is seen or TIMEOUT_S elapses."""
    buf = bytearray()
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            buf += ser.read(waiting)
            if ACK in buf:
                break
        else:
            time.sleep(0.01)
    return bytes(buf)


def main() -> None:
    commands = sys.argv[1:] or COMMANDS

    try:
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT_S,
            write_timeout=TIMEOUT_S,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
    except serial.SerialException as exc:
        print(f"Cannot open {PORT}: {exc}")
        return

    print(f"Opened {PORT} at {BAUD_RATE} baud")
    try:
        for command in commands:
            ser.reset_input_buffer()
            ser.write(command.encode("ascii"))
            ser.flush()
            print(f"TX: {command}")

            response = read_response(ser)
            if response:
                print(f"RX: {response.decode('ascii', errors='replace')!r}  (hex: {response.hex(' ')})")
            else:
                print(f"RX: <no response within {TIMEOUT_S * 1000:.0f} ms>")

            time.sleep(DELAY_BETWEEN_S)
    finally:
        ser.close()
        print(f"Closed {PORT}")


if __name__ == "__main__":
    main()
