# Runtime model inputs (not bundled)

Provide the actual Hailo-8 YOLOv8m-Pose HEF as `yolov8m_pose.hef` and a matching
`pose_model.yaml`, based on `src/fall_detection_edge/config/pose_model.example.yaml`.
The example deliberately refuses to run. requires Pi verification: input shape,
scaling, quantization, output names/layout, DFL/keypoint decoding, missing values,
and original model person selection/NMS conditions.

For an incompatible output layout, provide a HEF-specific Python module satisfying
the decoder contract described in the example. It can live in this directory
(`/models` is added to PYTHONPATH in Compose). No compiled model is downloaded or
substituted automatically.
