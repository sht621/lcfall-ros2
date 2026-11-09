"""Strict PointCloud2 XYZ/tag I/O, including row padding and wire endianness."""
from copy import deepcopy

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


OUTPUT_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='tag', offset=12, datatype=PointField.UINT8, count=1),
]
OUTPUT_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('tag', 'u1')])


def read_xyz_tag(cloud):
    fields = {}
    for field in cloud.fields:
        if field.name in fields:
            raise ValueError('duplicate PointCloud2 field name')
        fields[field.name] = field
    formats, offsets = [], []
    endian = '>' if cloud.is_bigendian else '<'
    for name, datatype, fmt, size in (
        ('x', PointField.FLOAT32, endian + 'f4', 4),
        ('y', PointField.FLOAT32, endian + 'f4', 4),
        ('z', PointField.FLOAT32, endian + 'f4', 4),
        ('tag', PointField.UINT8, 'u1', 1),
    ):
        field = fields.get(name)
        if field is None or field.datatype != datatype or field.count != 1:
            raise ValueError(f'input {name} missing or wrong type/count; requires Pi verification')
        if field.offset < 0 or field.offset + size > cloud.point_step:
            raise ValueError('PointCloud2 field exceeds point_step')
        formats.append(fmt)
        offsets.append(field.offset)
    if any(a < b + sb and b < a + sa
           for i, (a, sa) in enumerate(zip(offsets, (4, 4, 4, 1)))
           for b, sb in list(zip(offsets, (4, 4, 4, 1)))[i + 1:]):
        raise ValueError('overlapping XYZ/tag fields')
    if cloud.height < 1 or cloud.row_step < cloud.width * cloud.point_step:
        raise ValueError('invalid cloud dimensions/row_step')
    if len(cloud.data) != cloud.row_step * cloud.height:
        raise ValueError('PointCloud2 data length mismatch')
    if cloud.width == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint8)
    dtype = np.dtype(dict(names=['x', 'y', 'z', 'tag'], formats=formats,
                         offsets=offsets, itemsize=cloud.point_step))
    raw = np.ndarray((cloud.height, cloud.width), dtype=dtype, buffer=cloud.data,
                     strides=(cloud.row_step, cloud.point_step)).reshape(-1)
    xyz = np.column_stack([raw[name] for name in ('x', 'y', 'z')]).astype(np.float32)
    return xyz, raw['tag'].copy()


def make_cloud(header, target_frame, xyz, tags):
    wire = np.empty(len(xyz), dtype=OUTPUT_DTYPE)
    for index, name in enumerate(('x', 'y', 'z')):
        wire[name] = xyz[:, index]
    wire['tag'] = tags
    result = PointCloud2()
    result.header = deepcopy(header)
    result.header.frame_id = target_frame
    result.height, result.width = 1, len(wire)
    result.fields = deepcopy(OUTPUT_FIELDS)
    result.is_bigendian = False
    result.point_step = OUTPUT_DTYPE.itemsize
    result.row_step = result.width * result.point_step
    result.is_dense = True
    result.data = wire.tobytes()
    return result


def transform_xyz(xyz, transform):
    t, q = transform.translation, transform.rotation
    values = np.array([q.x, q.y, q.z, q.w, t.x, t.y, t.z], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite TF')
    norm = np.linalg.norm(values[:4])
    if abs(norm - 1.0) > 1e-3:
        raise ValueError('TF quaternion is not normalized')
    x, y, z, w = values[:4] / norm
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    return xyz @ rotation.T + values[4:]
