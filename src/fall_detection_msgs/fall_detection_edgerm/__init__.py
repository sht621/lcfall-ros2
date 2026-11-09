"""Optional binding for this one service; requires Pi verification of EdgeRM ABI."""
from rclpy.middleware import Middleware
from . import _service


class SendSynchronizedFrame(Middleware):
    __slots__ = ()

    def __init__(self, edgerm_url, runtime_name, verbose=False):
        self.handle = _service.SendSynchronizedFrame(edgerm_url, runtime_name, verbose)
        self.index = self.handle.index
