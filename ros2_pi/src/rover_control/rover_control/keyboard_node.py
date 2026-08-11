"""keyboard_node: read keystrokes and publish geometry_msgs/Twist on /cmd_vel.

Controls (no Enter needed):
    w / s : increase / decrease forward speed
    a / d : increase / decrease turn (left / right)
    space : stop (zero linear and angular)
    x     : same as space
    q     : quit

The node publishes the current Twist continuously at a fixed rate so that any
subscriber (fake_motor_node today, an Arduino bridge tomorrow) always receives a
steady command stream. It is agnostic to what consumes /cmd_vel.

Key reading is cross-platform: macOS/Linux use termios+select, Windows uses
msvcrt. The rest of the node is identical on every platform.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

_IS_WINDOWS = sys.platform == 'win32'

HELP = """
rover keyboard teleop  ->  publishing geometry_msgs/Twist on /cmd_vel
------------------------------------------------------------------
   w/s : forward speed  +/-        a/d : turn left/right  +/-
   space or x : stop            q : quit
------------------------------------------------------------------
"""

LIN_STEP = 0.05   # m/s per keypress
ANG_STEP = 0.20   # rad/s per keypress
LIN_MAX = 0.50
ANG_MAX = 1.50


def _clamp(value, limit):
    return max(-limit, min(limit, value))


class KeyReader:
    """Read single keystrokes without Enter, on any OS.

    Use as a context manager so terminal state is always restored:
        with KeyReader() as keys:
            key = keys.get(timeout=0.05)
    """

    def __enter__(self):
        if _IS_WINDOWS:
            import msvcrt
            self._msvcrt = msvcrt
        else:
            import termios
            import tty
            self._termios = termios
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if not _IS_WINDOWS:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)

    def get(self, timeout):
        """Return one character if pressed within timeout seconds, else ''."""
        if _IS_WINDOWS:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self._msvcrt.kbhit():
                    return self._msvcrt.getwch()
                time.sleep(0.005)
            return ''
        import select
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return ''


class KeyboardNode(Node):
    def __init__(self):
        super().__init__('keyboard_node')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.linear_x = 0.0
        self.angular_z = 0.0
        # Publish at 10 Hz regardless of typing, so the command is held.
        self.timer = self.create_timer(0.1, self._publish)
        self.get_logger().info('keyboard_node started; publishing on /cmd_vel')

    def apply_key(self, key):
        """Update the target velocity from a single keystroke. Returns False to quit."""
        if key == 'w':
            self.linear_x = _clamp(self.linear_x + LIN_STEP, LIN_MAX)
        elif key == 's':
            self.linear_x = _clamp(self.linear_x - LIN_STEP, LIN_MAX)
        elif key == 'a':
            self.angular_z = _clamp(self.angular_z + ANG_STEP, ANG_MAX)
        elif key == 'd':
            self.angular_z = _clamp(self.angular_z - ANG_STEP, ANG_MAX)
        elif key in (' ', 'x'):
            self.linear_x = 0.0
            self.angular_z = 0.0
        elif key in ('q', '\x03'):  # q or Ctrl-C
            return False
        return True

    def _publish(self):
        msg = Twist()
        msg.linear.x = float(self.linear_x)
        msg.angular.z = float(self.angular_z)
        self.publisher.publish(msg)
        # \r keeps the live status on one line without spamming scrollback.
        sys.stdout.write(
            '\r  cmd_vel  linear.x = {:+.2f} m/s   angular.z = {:+.2f} rad/s   '
            .format(self.linear_x, self.angular_z))
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardNode()

    print(HELP)
    try:
        with KeyReader() as keys:
            while rclpy.ok():
                key = keys.get(0.05)
                if key and not node.apply_key(key):
                    break
                # Service timers/callbacks without blocking on input.
                rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        # Command a full stop on the way out (terminal already restored).
        stop = Twist()
        node.publisher.publish(stop)
        print('\nstopping rover and shutting down keyboard_node.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
