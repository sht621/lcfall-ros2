# AGENTS.md — Edge pipeline implementation

## Scope and authoritative design
- This branch implements the Raspberry Pi 5 + Hailo-8 ROS 2 edge pipeline, not the legacy GPU-based fall detection application.
- **The sole authoritative specification is `docs/edge_pipeline_design_v2.6.md`. Read it before editing code.** Maintain every explicitly agreed message schema, topic name, default, processing rule, and scope boundary. This document is approved; do not rewrite or silently reinterpret it.
- Use the `oc-demo` branch's implementations as references for camera/LiDAR driver configuration, timestamp synchronization, frame transforms, ROI handling, and launch configuration. Rewrite/restructure freely in this feature branch. Never modify `oc-demo`, `tpr-demo`, or `main`.
- The target runtime is **Raspberry Pi 5 (Linux ARM64), ROS 2 Jazzy, Hailo-8 using Python/HailoRT, RealSense D455 (RGB only), and Livox MID-360**. Use a new single Docker image/container if viable as prescribed. Do not substitute an x86/CUDA/PyTorch GPU runtime or change the approved architecture.

## CRITICAL: implementation-only phase — DO NOT TEST
- **Until all planned implementation code, interfaces, configuration, launch files, and deployment artifacts are finished, do not run any tests, build commands, simulations, sample publishers, mock/synthetic data pipelines, benchmarks, hardware probes, execution/launch commands, or deployment/installation commands.**
- This prohibition includes `pytest`, `unittest`, `colcon build`, `docker build`, `docker compose up`, `ros2 run`, `ros2 launch`, `python -m compileall`, execution-based lint checks, and any command that launches a node or checks runtime behavior.
- Work on the MacBook **by reading and editing source files only**. No Hailo-8 or ROS 2 sensor runtime is expected there.
- Reviewing text, source code, package manifests, Dockerfiles, or configuration without executing them is permitted.
- 完了時は日本語で簡潔に、変更内容・未解決事項・ビルド／テスト／実機検証が未実施であることを報告して停止する。
- 実装完了後も自動で検証を開始しない。ユーザーがPi段階を開始し、明示的に許可するまで待つ。
- 後のPi検証による証拠なしに、ビルド可能・動作確認済み・10Hz達成・ハードウェア互換とは主張しない。
k
## Architecture
- New packages: `fall_detection_msgs` (ament_cmake) and `fall_detection_edge` (ament_python), with approved `.msg` and `SendSynchronizedFrame.srv` interface definitions.
- Sensor input → `sync_node` → parallel `pose_estimation_node` and `lidar_preprocess_node` → `frame_publisher_node` → `/synchronized_frame`.
- For temporary EdgeRM service-only transport, add `topic_to_service_bridge_node`, which subscribes to `/synchronized_frame` and calls `SendSynchronizedFrame` asynchronously. The internal nodes remain conventional ROS 2 nodes; no generalized communication adapter framework.
- When EdgeRM topic communication becomes available, remove/disable only the bridge; the GPU server subscribes to `/synchronized_frame`.
- Do not send raw RGB images to the GPU server, add inference/tracking/fall decisions to the Pi, or replace the agreed ROS 2 message contract with the legacy `PreprocessedFrame`.
- Bridge initial parameters: `max_in_flight_requests=2`, `pending_buffer_size=1` (latest wins), `request_timeout_sec=0.5`, no retries. Ensure timely accounting and clean-up of pending requests without blocking frame processing.
- Internal synchronizer: raw RGB+LiDAR `ApproximateTimeSynchronizer`, queue=10, slop=0.0167 s; propagate original stamps and pair info. Collector performs exact timestamp correlation (no re-synchronization), completed-order publish, up to 10 pending items per input and 1 s expiry. Empty person list and empty ROI point cloud are valid.
- Reference the design for all remaining field, coordinate/TF, ROI, QoS, and sensor details, especially any deliberately deferred verification.

## How to work
1. Read `docs/edge_pipeline_design_v2.6.md` and inspect the pertinent `oc-demo` files.
2. Implement incrementally, keeping boundaries modular and adapting legacy source only where correct. Remove obsolete application files from this feature branch if they add confusion; retain needed driver references/configuration.
3. Keep a short log of completed implementation stages, decisions, and Pi-only validation tasks in `docs/implementation_progress.md`.
4. If exact HailoRT ABI / model HEF I/O / EdgeRM configuration cannot be determined without target hardware or artifacts, isolate the dependency, label it **"requires Pi verification"**, and continue implementing what is known. Do not fabricate API compatibility or test outcomes.
5. Before declaring implementation complete, perform a **source-only design-to-code review** (no command execution, build, runtime checks, or test execution), report all remaining missing items, and stop at handoff.
