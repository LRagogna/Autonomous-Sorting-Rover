"""Main Raspberry Pi entry point for driving the rover over Arduino serial.

Example:
    python src/main.py FFBB --port /dev/ttyACM0

The sequence FFBB sends:
    F for 1 second, stop
    F for 1 second, stop
    B for 1 second, stop
    B for 1 second, stop
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Protocol


DEFAULT_BAUD_RATE = 9600
DEFAULT_MOVE_SECONDS = 1.0
DEFAULT_COMMAND_PAUSE_SECONDS = 0.05


class SerialConnection(Protocol):
    def write(self, data: bytes) -> int:
        ...

    def flush(self) -> None:
        ...


@dataclass(frozen=True)
class DriveStep:
    command: str
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rover drive commands through the Arduino serial controller."
    )
    parser.add_argument(
        "sequence",
        help="Drive sequence using F and B, such as FFBB.",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Arduino serial port. Defaults to /dev/ttyACM0 on Raspberry Pi.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD_RATE,
        help=f"Serial baud rate. Defaults to {DEFAULT_BAUD_RATE}.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_MOVE_SECONDS,
        help=f"Seconds to run each F or B command. Defaults to {DEFAULT_MOVE_SECONDS}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without opening the serial port.",
    )
    return parser.parse_args()


def build_drive_steps(sequence: str, seconds: float) -> list[DriveStep]:
    if seconds <= 0:
        raise ValueError("--seconds must be greater than 0.")

    normalized = "".join(sequence.upper().split())
    if not normalized:
        raise ValueError("Sequence cannot be empty.")

    invalid = sorted({command for command in normalized if command not in {"F", "B"}})
    if invalid:
        raise ValueError(
            "Sequence can only contain F and B commands. "
            f"Invalid command(s): {', '.join(invalid)}"
        )

    return [DriveStep(command=command, seconds=seconds) for command in normalized]


def send_command(connection: SerialConnection, command: str) -> None:
    connection.write(f"{command}\n".encode("ascii"))
    connection.flush()


def run_drive_sequence(
    connection: SerialConnection,
    steps: Iterable[DriveStep],
    pause_seconds: float = DEFAULT_COMMAND_PAUSE_SECONDS,
) -> None:
    for step in steps:
        send_command(connection, step.command)
        time.sleep(step.seconds)
        send_command(connection, "S")
        time.sleep(pause_seconds)


def open_serial_connection(port: str, baud: int) -> SerialConnection:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "Missing pyserial. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return serial.Serial(port=port, baudrate=baud, timeout=1)


def main() -> int:
    args = parse_args()

    try:
        steps = build_drive_steps(args.sequence, args.seconds)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for step in steps:
            print(f"{step.command} for {step.seconds:g}s, then S")
        return 0

    try:
        with open_serial_connection(args.port, args.baud) as connection:
            time.sleep(2.0)
            run_drive_sequence(connection, steps)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error opening serial port {args.port}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
