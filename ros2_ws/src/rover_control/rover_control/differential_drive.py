"""Differential-drive kinematics: Twist (v, omega) -> left/right motor commands.

This module is intentionally free of ROS and hardware dependencies so it can be
reused unchanged by:
  * fake_motor_node  (prints the motor values)
  * a future arduino_bridge_node (writes the same values over serial)

Keeping the conversion here is what makes the fake node swappable: only the
"what do I do with (left, right)" step differs between simulation and hardware.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DriveConfig:
    """Physical / scaling parameters for the differential-drive conversion."""
    wheel_base: float = 0.30        # distance between wheels [m]
    max_linear: float = 0.50        # linear speed that maps to full throttle [m/s]
    max_angular: float = 1.50       # angular speed that maps to full throttle [rad/s]
    max_motor: float = 255.0        # output scale (e.g. PWM range for Arduino)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def twist_to_motors(linear_x: float, angular_z: float, cfg: DriveConfig):
    """Convert a body-frame Twist into (left, right) motor commands.

    Standard differential-drive model:
        v_left  = v - omega * (wheel_base / 2)
        v_right = v + omega * (wheel_base / 2)

    The wheel speeds (in m/s) are then normalised against the configured maxima
    and scaled to the motor output range, clamped to +/- max_motor.

    Returns:
        (left, right) as floats in the range [-max_motor, max_motor].
    """
    half_base = cfg.wheel_base / 2.0

    v_left = linear_x - angular_z * half_base
    v_right = linear_x + angular_z * half_base

    # Reference speed used to normalise wheel velocity into a 0..1 throttle.
    # It accounts for the extra speed a wheel reaches during a full-rate turn.
    reference = cfg.max_linear + cfg.max_angular * half_base
    if reference <= 0.0:
        reference = 1.0

    left = _clamp(v_left / reference * cfg.max_motor, cfg.max_motor)
    right = _clamp(v_right / reference * cfg.max_motor, cfg.max_motor)

    return left, right
