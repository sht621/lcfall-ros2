"""Bounded exact correlation; arrival times are monotonic, keys are integer ns."""
from collections import OrderedDict


class Collector:
    def __init__(self, limits, timeout, publish, warn, history_size=256):
        self.limits, self.timeout = limits, timeout
        self.publish, self.warn = publish, warn
        self.buffers = {kind: OrderedDict() for kind in limits}
        # Bounded key-only history suppresses repeated DDS/application deliveries.
        # Once the history fills, a retired low watermark rejects older replay.
        self.seen = {kind: set() for kind in limits}
        self.floors = {kind: None for kind in limits}
        self.history_size = history_size

    def add(self, kind, key, message, now):
        self.expire(now)
        floor = self.floors[kind]
        if key in self.seen[kind] or (floor is not None and key <= floor):
            self.warn('duplicate', f'Duplicate/retired {kind} stamp discarded')
            return
        if kind == 'pairs':
            rgb, lidar = key
            if any((rgb == r or lidar == l) and key != (r, l)
                   for r, l in self.seen['pairs']):
                self.warn('conflict', 'Conflicting PairInfo reuses a sensor stamp; discarded')
                return
        self.seen[kind].add(key)
        if len(self.seen[kind]) > self.history_size:
            retired = min(self.seen[kind])
            self.seen[kind].remove(retired)
            self.floors[kind] = retired
        self.buffers[kind][key] = (now, message)
        # Complete before capacity eviction: a just-completed pair is not pending.
        self.assemble()
        for name, buffer in self.buffers.items():
            while len(buffer) > self.limits[name]:
                old, _ = buffer.popitem(last=False)
                self.warn('capacity', f'Oldest incomplete {name} entry evicted: {old}')

    def assemble(self):
        pairs, poses, points = (self.buffers[k] for k in ('pairs', 'poses', 'points'))
        for key in list(pairs):
            rgb, lidar = key
            if rgb in poses and lidar in points:
                pose, cloud = poses[rgb][1], points[lidar][1]
                del pairs[key]
                del poses[rgb]
                del points[lidar]
                self.publish(pose, cloud)

    def expire(self, now):
        for name, buffer in self.buffers.items():
            while buffer:
                key, (arrival, _) = next(iter(buffer.items()))
                if now - arrival < self.timeout:
                    break
                del buffer[key]
                self.warn('expiry', f'Incomplete {name} entry expired: {key}')
