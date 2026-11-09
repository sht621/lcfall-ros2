"""Isolated HailoRT vstream binding. ABI and actual HEF require Pi verification.

API reference (source only, not a runtime compatibility assertion):
https://github.com/hailo-ai/hailort/blob/master/hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py
"""
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
import math

import yaml
from .pose_decoder import RawYolov8PoseDecoder
from .pose_processing import prepare


class HailoBackend:
    def __init__(self, hef_path, manifest_path):
        if not Path(hef_path).is_file() or not Path(manifest_path).is_file():
            raise ValueError('HEF and model manifest are required; requires Pi verification')
        with open(manifest_path, encoding='utf-8') as stream:
            self.manifest = yaml.safe_load(stream)
        cfg = self.manifest['preprocess']
        if self.manifest['verified_for_hef'] is not True:
            raise ValueError('Set verified_for_hef only after matching the manifest to the actual HEF')
        if (cfg['input_dtype'] not in ('uint8', 'float32')
                or cfg['channel_order'] not in ('rgb', 'bgr')
                or not isinstance(cfg['pad_value'], int) or not 0 <= cfg['pad_value'] <= 255
                or not math.isfinite(cfg['input_scale']) or not math.isfinite(cfg['input_offset'])
                or not isinstance(cfg['input_quantized'], bool)):
            raise ValueError('invalid explicit preprocessing configuration')
        self.stack = ExitStack()
        try:
            import hailo_platform as hp
            hef = hp.HEF(hef_path)
            inputs, outputs = hef.get_input_vstream_infos(), hef.get_output_vstream_infos()
            if len(inputs) != 1 or len(inputs[0].shape) != 3:
                raise ValueError('requires exactly one HWC input')
            self.input_name = inputs[0].name
            self.input_shape = tuple(inputs[0].shape)
            expected = tuple(self.manifest['input_shape'])
            if self.input_shape != expected or expected[2] != 3:
                raise ValueError('manifest/HEF HWC input shape mismatch')
            shapes = {info.name: tuple(info.shape) for info in outputs}
            decode_cfg = self.manifest['decoder']
            if decode_cfg['type'] == 'yolov8_pose_raw':
                self.decoder = RawYolov8PoseDecoder(decode_cfg, shapes, self.input_shape)
            elif decode_cfg['type'] == 'custom':
                # Explicit HEF-specific dependency, not a communication abstraction.
                module = import_module(decode_cfg['module'])
                self.decoder = module.create_decoder(decode_cfg, shapes, self.input_shape)
            else:
                raise ValueError('unsupported HEF decoder type')
            device = self.stack.enter_context(hp.VDevice())
            groups = device.configure(hef, hp.ConfigureParams.create_from_hef(
                hef, interface=hp.HailoStreamInterface.PCIe))
            if len(groups) != 1:
                raise ValueError('requires exactly one Hailo network group')
            group = groups[0]
            fmt = hp.FormatType.UINT8 if cfg['input_dtype'] == 'uint8' else hp.FormatType.FLOAT32
            input_params = hp.InputVStreamParams.make(
                group, quantized=cfg['input_quantized'], format_type=fmt)
            output_params = hp.OutputVStreamParams.make(
                group, quantized=False, format_type=hp.FormatType.FLOAT32)
            self.pipeline = self.stack.enter_context(hp.InferVStreams(group, input_params, output_params))
            self.stack.enter_context(group.activate(group.create_params()))
        except Exception:
            self.stack.close()
            raise

    def infer(self, rgb):
        tensor, mapping = prepare(rgb, self.input_shape, self.manifest['preprocess'])
        outputs = self.pipeline.infer({self.input_name: tensor})
        return self.decoder.decode(outputs), mapping

    def close(self):
        self.stack.close()
