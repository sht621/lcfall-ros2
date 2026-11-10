"""Shared QoS, input checks and console-only status helpers."""
import math
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def internal_qos():
    return QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                      reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.VOLATILE)


def parameter(node, name, default):
    # Configuration is applied at startup, avoiding partially updated transforms/buffers.
    return node.declare_parameter(
        name, default, ParameterDescriptor(read_only=True)).value


def positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def stamp_ns(stamp):
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
        raise ValueError('invalid sensor stamp')
    value = stamp.sec * 1_000_000_000 + stamp.nanosec
    if value == 0:
        raise ValueError('zero sensor stamp')
    return value


class Console:
    """Throttle repeated console messages without persistent history or file output."""
    def __init__(self, node):
        self.node = node
        self.last = {}

    def show(self, key, message, warning=False, interval=5.0):
        now = time.monotonic()
        if now - self.last.get(key, -math.inf) >= interval:
            self.last[key] = now
            log = self.node.get_logger()
            # rclpy caches severity per source call site; keep distinct lines.
            if warning:
                log.warning(message)
            else:
                log.info(message)


def steady_timer(node, period, callback):
    return node.create_timer(period, callback, clock=Clock(clock_type=ClockType.STEADY_TIME))


def run(factory, args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = factory()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
