"""Temporary nonblocking ROS topic -> EdgeRM-facing service bridge; never retry."""
from collections import deque
import time

from rclpy.node import Node
from fall_detection_msgs.msg import SynchronizedFrame
from fall_detection_msgs.srv import SendSynchronizedFrame
from .common import Console, internal_qos, parameter, positive, run, steady_timer


class TopicToServiceBridgeNode(Node):
    def __init__(self):
        super().__init__('topic_to_service_bridge_node')
        self.console = Console(self)
        self.maximum = int(positive(parameter(self, 'max_in_flight_requests', 2), 'max_in_flight_requests'))
        self.pending_size = int(positive(parameter(self, 'pending_buffer_size', 1), 'pending_buffer_size'))
        self.timeout = positive(parameter(self, 'request_timeout_sec', 0.5), 'request_timeout_sec')
        service_name = parameter(self, 'service_name', '/send_synchronized_frame')
        use_edgerm = parameter(self, 'use_edgerm', True)
        if use_edgerm:
            url = parameter(self, 'edgerm_url', '')
            runtime = parameter(self, 'edgerm_runtime_name', '')
            verbose = parameter(self, 'edgerm_verbose', False)
            if not url or not runtime:
                raise ValueError('EdgeRM URL/runtime must be configured; requires Pi verification')
            from fall_detection_edgerm import SendSynchronizedFrame as EdgeRMService
            self.middleware = EdgeRMService(url, runtime, verbose)
            self.client = self.create_client(self.middleware, service_name)
        else:
            # Explicit ordinary ROS service mode, never an automatic fallback.
            self.client = self.create_client(SendSynchronizedFrame, service_name)
        self.pending = deque(maxlen=self.pending_size)
        self.in_flight = {}
        self.subscription = self.create_subscription(
            SynchronizedFrame, '/synchronized_frame', self.receive, internal_qos())
        # Steady timer works even when /clock stalls; callbacks and bookkeeping all
        # run in the default mutually exclusive group / single-threaded executor.
        self.timer = steady_timer(self, min(0.02, self.timeout / 4), self.pump)
        self.get_logger().info(f'Temporary service bridge -> {service_name}; no retries')

    def receive(self, frame):
        # Reap before enqueueing so a fresh frame replaces any old pending frame
        # before freed slots are filled. The latest-only default is exactly one.
        self.reap()
        if len(self.pending) == self.pending.maxlen:
            self.console.show('overwrite', 'Unsent pending frame replaced by newer frame', True)
        self.pending.append(frame)
        self.send_pending()

    def reap(self):
        now = time.monotonic()
        for future, sent in list(self.in_flight.items()):
            expired = now - sent >= self.timeout
            if future.done():
                del self.in_flight[future]
                if expired:
                    self.console.show('timeout', 'Service deadline exceeded; receipt unknown; no retry', True)
                else:
                    try:
                        response = future.result()
                        if response is None or not response.accepted:
                            self.console.show('rejected', 'Server did not accept frame; no retry', True)
                        else:
                            self.console.show('accepted', 'Server accepted synchronized frame')
                    except Exception as exc:
                        self.console.show('failed', f'Service request failed; no retry: {exc}', True)
                # rclpy may already have removed a completed request; removal is idempotent.
                self.client.remove_pending_request(future)
            elif expired:
                del self.in_flight[future]
                self.client.remove_pending_request(future)
                future.cancel()
                self.console.show('timeout', 'Service timeout; receipt unknown; no retry', True)

    def send_pending(self):
        if not self.pending:
            return
        if not self.client.service_is_ready():
            self.console.show('unavailable', 'Service unavailable; bounded newest pending frames retained', True)
            return
        while self.pending and len(self.in_flight) < self.maximum:
            frame = self.pending.popleft()
            try:
                future = self.client.call_async(SendSynchronizedFrame.Request(frame=frame))
                self.in_flight[future] = time.monotonic()
            except Exception as exc:
                self.console.show('send', f'Async send failed; frame dropped, no retry: {exc}', True)

    def pump(self):
        self.reap()
        self.send_pending()

    def destroy_node(self):
        self.timer.cancel()
        self.pending.clear()
        for future in list(self.in_flight):
            self.client.remove_pending_request(future)
            future.cancel()
        self.in_flight.clear()
        return super().destroy_node()


def main(args=None):
    run(TopicToServiceBridgeNode, args)
