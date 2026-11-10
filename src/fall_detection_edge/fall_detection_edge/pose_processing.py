"""Image letterbox and conversion from model-input pixels to original RGB pixels."""
from dataclasses import dataclass

import cv2
import numpy as np
from fall_detection_msgs.msg import PersonPose


@dataclass(frozen=True)
class Letterbox:
    scale_x: float
    scale_y: float
    left: int
    top: int


def prepare(rgb, shape, config):
    height, width, channels = shape
    if channels != 3:
        raise ValueError('only HWC 3-channel model input is supported')
    source_h, source_w = rgb.shape[:2]
    scale = min(width / source_w, height / source_h)
    # Match the pinned Model Zoo pose_preprocessing.letterbox: truncate resized
    # dimensions, centred padding and area interpolation for downscaling.
    resized_w, resized_h = int(source_w * scale), int(source_h * scale)
    if resized_w < 1 or resized_h < 1:
        raise ValueError('source aspect ratio produces an empty resized image')
    left = round((width - resized_w) / 2 - 0.1)
    top = round((height - resized_h) / 2 - 0.1)
    canvas = np.full((height, width, 3), config['pad_value'], dtype=np.uint8)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    image = cv2.resize(rgb, (resized_w, resized_h), interpolation=interpolation)
    canvas[top:top + resized_h, left:left + resized_w] = image
    order = config['channel_order']
    if order == 'bgr':
        canvas = canvas[:, :, ::-1]
    elif order != 'rgb':
        raise ValueError('channel_order must be rgb or bgr')
    dtype = config['input_dtype']
    if dtype == 'uint8':
        if config['input_scale'] != 1.0 or config['input_offset'] != 0.0:
            raise ValueError('uint8 input requires identity scaling')
        tensor = np.ascontiguousarray(canvas)
    elif dtype == 'float32':
        tensor = np.ascontiguousarray(canvas, dtype=np.float32)
        tensor = tensor * config['input_scale'] + config['input_offset']
    else:
        raise ValueError('input_dtype must be uint8 or float32')
    return tensor[None, ...], Letterbox(resized_w / source_w, resized_h / source_h, left, top)


def persons_from_rows(rows, mapping):
    """Rows: [xmin,ymin,xmax,ymax,det_score, 17*(x,y,score)], model-input pixels.

    Decoder alone owns model selection/NMS. Do not apply any joint score threshold,
    clipping, interpolation, zero-as-missing rule, or additional person filtering here.
    """
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 56:
        raise ValueError('decoder must return an (N,56) array, including (0,56) for no persons')
    persons = []
    for row in rows:
        bbox = row[:4].copy()
        joints = row[5:].reshape(17, 3).copy()
        if not np.isfinite(row[:5]).all() or bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise ValueError('invalid bbox/detection score')
        valid = np.isfinite(joints).all(axis=1)
        missing = np.isnan(joints).all(axis=1)
        if not (valid | missing).all():
            raise ValueError('partial NaN or Inf in a COCO17 joint')
        bbox[[0, 2]] = (bbox[[0, 2]] - mapping.left) / mapping.scale_x
        bbox[[1, 3]] = (bbox[[1, 3]] - mapping.top) / mapping.scale_y
        joints[:, 0] = (joints[:, 0] - mapping.left) / mapping.scale_x
        joints[:, 1] = (joints[:, 1] - mapping.top) / mapping.scale_y
        if not np.isfinite(bbox).all() or not np.isfinite(joints[valid]).all():
            raise ValueError('coordinate restoration overflow')
        persons.append(PersonPose(
            bbox_xyxy=bbox.tolist(), detection_score=float(row[4]),
            keypoint_x=joints[:, 0].tolist(), keypoint_y=joints[:, 1].tolist(),
            keypoint_score=joints[:, 2].tolist()))
    return persons
