"""Explicitly configured YOLOv8 pose raw-head decoder (requires Pi verification).

No output name, threshold, HEF layout, or missing-joint sentinel is guessed.
Other HEF layouts use a model-specific decoder module, never a dummy fallback.
"""
import numpy as np


def sigmoid(values):
    # Equivalent sigmoid without overflow or clipping low-confidence logits.
    exp_negative_abs = np.exp(-np.abs(values))
    return np.where(values >= 0, 1.0 / (1.0 + exp_negative_abs),
                    exp_negative_abs / (1.0 + exp_negative_abs))


def nms(rows, threshold, max_detections):
    # Model Zoo dc402c7: outer score sort, then cython_nms uses inclusive
    # pixel areas (+1) and suppresses IoU >= threshold. Preserve its returned
    # outer-sort order, including the two argsort operations for tied scores.
    if len(rows) == 0:
        return rows
    rows = rows[rows[:, 4].argsort()[::-1]]
    order = rows[:, 4].argsort()[::-1]
    areas = np.prod(rows[:, 2:4] - rows[:, :2] + 1, axis=1)
    threshold = np.float32(threshold)
    kept = []
    while len(order):
        current, rest = order[0], order[1:]
        kept.append(current)
        a, b = rows[current, :4], rows[rest, :4]
        intersect = np.prod(np.maximum(0, np.minimum(a[2:], b[:, 2:])
                                      - np.maximum(a[:2], b[:, :2]) + 1), axis=1)
        union = areas[current] + areas[rest] - intersect
        iou = np.divide(intersect, union, out=np.zeros_like(intersect), where=union > 0)
        order = rest[iou < threshold]
    # No pre-NMS candidate cap. Match the official post-NMS limit; worst-case
    # work is still quadratic in up to 8400 candidates (requires Pi timing).
    return rows[np.sort(kept)[:max_detections]]


class RawYolov8PoseDecoder:
    def __init__(self, config, output_shapes, input_shape):
        self.config = config
        self.heads = config['heads']
        self.bins = int(config['reg_max'])
        self.score_threshold = float(config['person_score_threshold'])
        self.iou_threshold = float(config['nms_iou_threshold'])
        self.max_detections = config['max_detections']
        if (self.bins < 2 or not self.heads
                or type(self.max_detections) is not int or self.max_detections <= 0
                or not 0 <= self.score_threshold <= 1 or not 0 <= self.iou_threshold <= 1):
            raise ValueError('invalid model decoder configuration')
        if config['scores'] not in ('logits', 'probabilities'):
            raise ValueError('scores must be logits or probabilities')
        if config['keypoint_scores'] not in ('logits', 'probabilities'):
            raise ValueError('keypoint_scores must be logits or probabilities')
        if config['missing_joints'] != 'none':
            raise ValueError('raw decoder supports no explicit missing sentinel; provide custom decoder')
        height, width, _ = input_shape
        names = []
        for head in self.heads:
            stride = head['stride']
            if not isinstance(stride, int) or stride <= 0 or height % stride or width % stride:
                raise ValueError('invalid head stride/input size')
            for field, channels in (('boxes', 4 * self.bins), ('scores', 1), ('keypoints', 51)):
                name = head[field]
                names.append(name)
                expected = (height // stride, width // stride, channels)
                if tuple(output_shapes.get(name, ())) != expected:
                    raise ValueError(f'HEF output {name} does not match explicit HWC shape {expected}')
        if len(set(names)) != len(names) or set(names) != set(output_shapes):
            raise ValueError('raw decoder output names must match all HEF outputs exactly')

    def decode(self, outputs):
        candidates = []
        for head in self.heads:
            stride = head['stride']
            arrays = [np.asarray(outputs[head[name]], dtype=np.float32)
                      for name in ('boxes', 'scores', 'keypoints')]
            boxes, scores, keypoints = [a[0] for a in arrays]
            if any(a.shape[0] != 1 or not np.isfinite(a).all() for a in arrays):
                raise ValueError('raw HEF outputs must be finite, batch=1')
            height, width, _ = boxes.shape
            scores = scores.reshape(-1)
            if self.config['scores'] == 'logits':
                scores = sigmoid(scores)
            if ((scores < 0) | (scores > 1)).any():
                raise ValueError('person scores outside probability range')
            selected = scores > self.score_threshold
            if not selected.any():
                continue
            # DFL distances about cell-centre anchors, in model-input pixels.
            distribution = boxes.reshape(-1, 4, self.bins)[selected]
            distribution = np.exp(distribution - distribution.max(axis=2, keepdims=True))
            distribution /= distribution.sum(axis=2, keepdims=True)
            distances = (distribution * np.arange(self.bins)).sum(axis=2) * stride
            grid_y, grid_x = np.mgrid[:height, :width]
            grid = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)[selected]
            centres = (grid + 0.5) * stride
            bbox = np.concatenate((centres - distances[:, :2], centres + distances[:, 2:]), axis=1)
            joints = keypoints.reshape(-1, 17, 3)[selected].copy()
            # Ultralytics raw YOLOv8 pose: (2*raw + anchor - 0.5)*stride.
            joints[:, :, :2] = (2 * joints[:, :, :2] + grid[:, None, :]) * stride
            if self.config['keypoint_scores'] == 'logits':
                joints[:, :, 2] = sigmoid(joints[:, :, 2])
            if ((joints[:, :, 2] < 0) | (joints[:, :, 2] > 1)).any():
                raise ValueError('keypoint scores outside probability range')
            candidates.append(np.concatenate((bbox, scores[selected, None], joints.reshape(-1, 51)), axis=1))
        if not candidates:
            return np.empty((0, 56), dtype=np.float32)
        return nms(np.concatenate(candidates).astype(np.float32), self.iou_threshold,
                   self.max_detections)
