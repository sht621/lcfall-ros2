#!/usr/bin/env python3
"""Independent camera/LiDAR ROI tuning GUI."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy.typing import NDArray
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
import yaml

from lcfall_ros2.utils.lidar_preprocessing import (
    apply_lidar_rotation,
    create_rotation_matrix,
    pointcloud2_to_numpy,
)


ROI_KEYS = (
    "roi_x_min", "roi_x_max",
    "roi_y_min", "roi_y_max",
    "roi_z_min", "roi_z_max",
)
AXES = (
    ("x", "前後（X）", -10.0, 20.0),
    ("y", "左右（Y）", -15.0, 15.0),
    ("z", "高さ（Z）", -5.0, 10.0),
)
AXIS_COLORS = {
    "x": "#ff5a5f",
    "y": "#33c36b",
    "z": "#4595e6",
}
SLIDER_SCALE = 100
MAX_DISPLAY_POINTS = 20000
VIEW_ELEVATION = 25.0
VIEW_AZIMUTH = -135.0


def load_roi_config(path: Path) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    if not path.is_file():
        raise ValueError(f"ROI設定ファイルがありません: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        document = yaml.safe_load(text)
        parameters = document["/**"]["ros__parameters"]
    except Exception as exc:
        raise ValueError(f"ROI設定YAMLを読み込めません: {exc}") from exc

    values: Dict[str, float] = {}
    for key in ROI_KEYS:
        if key not in parameters:
            raise ValueError(f"ROI値が不足しています: {key}")
        value = float(parameters[key])
        if not math.isfinite(value):
            raise ValueError(f"ROI値が不正です: {key}")
        values[key] = value

    for axis in ("x", "y", "z"):
        if values[f"roi_{axis}_min"] >= values[f"roi_{axis}_max"]:
            raise ValueError(f"ROIのmin/maxが不正です: {axis.upper()}")

    transform = {
        "enabled": bool(parameters.get("apply_coordinate_transform", True)),
        "roll": float(parameters.get("lidar_roll", 1.1)),
        "pitch": float(parameters.get("lidar_pitch", 27.8)),
        "yaw": float(parameters.get("lidar_yaw", 0.0)),
    }
    return text, values, transform


def save_roi_config(path: Path, original_text: str, values: Dict[str, float]) -> None:
    updated = original_text
    for key in ROI_KEYS:
        pattern = re.compile(
            rf"^(\s*{re.escape(key)}\s*:\s*)([^#\n]+?)(\s*(?:#.*)?)$",
            re.MULTILINE,
        )
        updated, count = pattern.subn(
            lambda match, value=values[key]: (
                f"{match.group(1)}"
                f"{_format_like_original(value, match.group(2))}"
                f"{match.group(3)}"
            ),
            updated,
        )
        if count != 1:
            raise ValueError(f"ROI設定行を一意に更新できません: {key}")

    parsed = yaml.safe_load(updated)
    parameters = parsed["/**"]["ros__parameters"]
    for key in ROI_KEYS:
        if not math.isclose(float(parameters[key]), values[key], abs_tol=1e-9):
            raise ValueError(f"保存後のROI値を検証できません: {key}")

    stat = path.stat()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, stat.st_mode & 0o7777)
        if os.geteuid() == 0:
            os.chown(temporary_path, stat.st_uid, stat.st_gid)
        yaml.safe_load(temporary_path.read_text(encoding="utf-8"))
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _format_like_original(value: float, original: str) -> str:
    token = original.strip()
    decimal_match = re.fullmatch(r"[-+]?\d+\.(\d+)", token)
    if decimal_match:
        decimals = len(decimal_match.group(1))
        if not math.isclose(round(value, decimals), value, abs_tol=1e-9):
            decimals = max(decimals, 2)
        return f"{value:.{decimals}f}"
    return f"{value:.6g}"


class RangeSlider(QWidget):
    """Horizontal slider with lower and upper handles."""

    rangeChanged = pyqtSignal(int, int)
    _HANDLE_RADIUS = 8
    _TRACK_MARGIN = 12

    def __init__(
        self,
        minimum: int,
        maximum: int,
        lower_value: int,
        upper_value: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self._lower_value = lower_value
        self._upper_value = upper_value
        self._active_handle: Optional[str] = None
        self.setMinimumSize(260, 30)
        self.setMouseTracking(True)
        self.setValues(lower_value, upper_value, emit=False)

    @property
    def lowerValue(self) -> int:
        return self._lower_value

    @property
    def upperValue(self) -> int:
        return self._upper_value

    def setValues(self, lower_value: int, upper_value: int, emit: bool = True) -> None:
        lower_value = max(self._minimum, min(lower_value, self._maximum - 1))
        upper_value = max(self._minimum + 1, min(upper_value, self._maximum))
        if lower_value >= upper_value:
            if self._active_handle == "lower":
                lower_value = upper_value - 1
            else:
                upper_value = lower_value + 1
        changed = (
            lower_value != self._lower_value or upper_value != self._upper_value
        )
        self._lower_value = lower_value
        self._upper_value = upper_value
        self.update()
        if changed and emit:
            self.rangeChanged.emit(lower_value, upper_value)

    def _value_to_x(self, value: int) -> float:
        track_width = max(1, self.width() - 2 * self._TRACK_MARGIN)
        ratio = (value - self._minimum) / (self._maximum - self._minimum)
        return self._TRACK_MARGIN + ratio * track_width

    def _x_to_value(self, x_position: float) -> int:
        track_width = max(1, self.width() - 2 * self._TRACK_MARGIN)
        ratio = (x_position - self._TRACK_MARGIN) / track_width
        ratio = max(0.0, min(1.0, ratio))
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center_y = self.height() / 2
        left = self._value_to_x(self._minimum)
        right = self._value_to_x(self._maximum)
        lower_x = self._value_to_x(self._lower_value)
        upper_x = self._value_to_x(self._upper_value)

        painter.setPen(QPen(QColor("#777777"), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(int(left), int(center_y), int(right), int(center_y))
        painter.setPen(QPen(QColor("#24a9e0"), 5, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(int(lower_x), int(center_y), int(upper_x), int(center_y))

        painter.setPen(QPen(QColor("#137ba7"), 2))
        painter.setBrush(QColor("#f5f5f5"))
        for x_position in (lower_x, upper_x):
            painter.drawEllipse(
                int(x_position - self._HANDLE_RADIUS),
                int(center_y - self._HANDLE_RADIUS),
                self._HANDLE_RADIUS * 2,
                self._HANDLE_RADIUS * 2,
            )

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.button() != Qt.LeftButton:
            return
        lower_distance = abs(event.x() - self._value_to_x(self._lower_value))
        upper_distance = abs(event.x() - self._value_to_x(self._upper_value))
        self._active_handle = "lower" if lower_distance <= upper_distance else "upper"
        self._move_active_handle(event.x())
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._active_handle is not None:
            self._move_active_handle(event.x())
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.button() == Qt.LeftButton:
            self._active_handle = None
            event.accept()

    def _move_active_handle(self, x_position: int) -> None:
        value = self._x_to_value(x_position)
        if self._active_handle == "lower":
            self.setValues(min(value, self._upper_value - 1), self._upper_value)
        elif self._active_handle == "upper":
            self.setValues(self._lower_value, max(value, self._lower_value + 1))


class SensorNode(Node):
    def __init__(self) -> None:
        super().__init__("roi_tuner")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.latest_image: Optional[NDArray[np.uint8]] = None
        self.latest_points: Optional[NDArray[np.float32]] = None
        self.image_error: Optional[str] = None
        self.lidar_error: Optional[str] = None
        self.image_messages = 0
        self.lidar_messages = 0
        self.create_subscription(Image, "/camera/image_raw", self._image_callback, qos)
        self.create_subscription(PointCloud2, "/livox/lidar", self._lidar_callback, qos)

    def _image_callback(self, msg: Image) -> None:
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding not in ("rgb8", "bgr8"):
                raise ValueError(f"未対応encoding: {msg.encoding}")
            image = data.reshape((msg.height, msg.width, 3))
            if msg.encoding == "bgr8":
                image = image[:, :, ::-1]
            self.latest_image = np.ascontiguousarray(image)
            self.image_messages += 1
            self.image_error = None
        except Exception as exc:
            self.image_error = str(exc)

    def _lidar_callback(self, msg: PointCloud2) -> None:
        try:
            self.latest_points = pointcloud2_to_numpy(msg)
            self.lidar_messages += 1
            self.lidar_error = None
        except Exception as exc:
            self.lidar_error = str(exc)


class RoiTunerWindow(QMainWindow):
    def __init__(self, config_path: Path, sensor: SensorNode) -> None:
        super().__init__()
        self._config_path = config_path
        self._sensor = sensor
        self._config_text, self._values, transform = load_roi_config(config_path)
        self._rotation_matrix = None
        if transform["enabled"]:
            self._rotation_matrix = create_rotation_matrix(
                transform["roll"], transform["pitch"], transform["yaw"]
            )
        self._last_image_messages = -1
        self._last_lidar_messages = -1
        self._range_sliders: Dict[str, RangeSlider] = {}
        self._spinboxes: Dict[str, QDoubleSpinBox] = {}
        self._test_exit_code = 0
        self._auto_test = os.environ.get("ROI_TUNER_AUTOTEST", "").strip().lower()
        self._auto_test_deadline = time.monotonic() + 40.0
        self._auto_test_done = False

        self.setWindowTitle("ROI調整")
        self.setMinimumSize(1400, 800)
        self.resize(1900, 1050)
        self._build_ui()
        self._initial_limits = self._make_initial_limits()
        self._reset_view()

        self._ros_timer = QTimer(self)
        self._ros_timer.timeout.connect(self._spin_ros)
        self._ros_timer.start(10)

        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._refresh_views)
        self._render_timer.start(150)

        if self._auto_test:
            self._auto_timer = QTimer(self)
            self._auto_timer.timeout.connect(self._run_auto_test)
            self._auto_timer.start(250)

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        self._camera_label = QLabel("カメラ映像を待機中")
        self._camera_label.setAlignment(Qt.AlignCenter)
        self._camera_label.setMinimumSize(640, 420)
        self._camera_label.setStyleSheet("background: #121212; color: #dddddd;")
        splitter.addWidget(self._camera_label)

        lidar_widget = QWidget()
        lidar_layout = QVBoxLayout(lidar_widget)
        self._figure = Figure(facecolor="#121212", layout="constrained")
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")
        lidar_layout.addWidget(self._canvas, stretch=1)
        reset_button = QPushButton("視点をリセット")
        reset_button.clicked.connect(self._reset_view)
        lidar_layout.addWidget(reset_button, alignment=Qt.AlignRight)
        splitter.addWidget(lidar_widget)
        splitter.setSizes([760, 840])
        root.addWidget(splitter, stretch=1)

        controls = QGridLayout()
        for row, (axis, label, lower, upper) in enumerate(AXES):
            axis_label = QLabel(label)
            axis_label.setStyleSheet(
                f"color: {AXIS_COLORS[axis]}; font-weight: bold;"
            )
            controls.addWidget(axis_label, row, 0)
            controls.addWidget(QLabel("min"), row, 1)
            controls.addWidget(QLabel("max"), row, 4)
            for column, bound in ((2, "min"), (5, "max")):
                key = f"roi_{axis}_{bound}"
                spinbox = QDoubleSpinBox()
                spinbox.setRange(lower, upper)
                spinbox.setDecimals(2)
                spinbox.setSingleStep(0.01)
                spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
                spinbox.setValue(self._values[key])
                self._spinboxes[key] = spinbox
                spinbox.valueChanged.connect(
                    lambda value, item=key: self._spinbox_changed(item, value)
                )
                controls.addWidget(spinbox, row, column)
            range_slider = RangeSlider(
                int(lower * SLIDER_SCALE),
                int(upper * SLIDER_SCALE),
                int(round(self._values[f"roi_{axis}_min"] * SLIDER_SCALE)),
                int(round(self._values[f"roi_{axis}_max"] * SLIDER_SCALE)),
            )
            self._range_sliders[axis] = range_slider
            range_slider.rangeChanged.connect(
                lambda minimum, maximum, item=axis: self._range_slider_changed(
                    item, minimum, maximum
                )
            )
            controls.addWidget(range_slider, row, 3)
        controls.setColumnStretch(3, 1)
        root.addLayout(controls)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        discard_button = QPushButton("保存せず終了")
        discard_button.clicked.connect(self._discard_and_close)
        save_button = QPushButton("保存して終了")
        save_button.clicked.connect(self._save_and_close)
        buttons.addWidget(discard_button)
        buttons.addWidget(save_button)
        buttons.addStretch(1)
        root.addLayout(buttons)
        self.setCentralWidget(central)

    def _make_initial_limits(self):
        limits = []
        for axis in ("x", "y", "z"):
            minimum = self._values[f"roi_{axis}_min"]
            maximum = self._values[f"roi_{axis}_max"]
            padding = max((maximum - minimum) * 0.25, 0.25)
            limits.append((minimum - padding, maximum + padding))
        return limits

    def _range_slider_changed(
        self, axis: str, raw_minimum: int, raw_maximum: int
    ) -> None:
        minimum_key = f"roi_{axis}_min"
        maximum_key = f"roi_{axis}_max"
        self._values[minimum_key] = round(raw_minimum / SLIDER_SCALE, 2)
        self._values[maximum_key] = round(raw_maximum / SLIDER_SCALE, 2)
        for key in (minimum_key, maximum_key):
            spinbox = self._spinboxes[key]
            spinbox.blockSignals(True)
            spinbox.setValue(self._values[key])
            spinbox.blockSignals(False)
        self._draw_pointcloud()

    def _spinbox_changed(self, key: str, value: float) -> None:
        self._set_value(key, value)

    def _set_value(self, key: str, value: float) -> None:
        axis, bound = key.split("_")[1:]
        other_bound = "max" if bound == "min" else "min"
        other_key = f"roi_{axis}_{other_bound}"
        if bound == "min" and value >= self._values[other_key]:
            value = self._values[other_key] - 0.01
        elif bound == "max" and value <= self._values[other_key]:
            value = self._values[other_key] + 0.01
        self._values[key] = round(value, 2)
        self._sync_axis_widgets(axis)
        self._draw_pointcloud()

    def _sync_axis_widgets(self, axis: str) -> None:
        minimum_key = f"roi_{axis}_min"
        maximum_key = f"roi_{axis}_max"
        range_slider = self._range_sliders[axis]
        range_slider.blockSignals(True)
        range_slider.setValues(
            int(round(self._values[minimum_key] * SLIDER_SCALE)),
            int(round(self._values[maximum_key] * SLIDER_SCALE)),
            emit=False,
        )
        range_slider.blockSignals(False)
        for key in (minimum_key, maximum_key):
            spinbox = self._spinboxes[key]
            spinbox.blockSignals(True)
            spinbox.setValue(self._values[key])
            spinbox.blockSignals(False)

    def _spin_ros(self) -> None:
        rclpy.spin_once(self._sensor, timeout_sec=0.0)

    def _refresh_views(self) -> None:
        if self._sensor.image_messages != self._last_image_messages:
            self._last_image_messages = self._sensor.image_messages
            self._draw_camera()
        if self._sensor.lidar_messages != self._last_lidar_messages:
            self._last_lidar_messages = self._sensor.lidar_messages
            self._draw_pointcloud()

    def _draw_camera(self) -> None:
        image = self._sensor.latest_image
        if image is None:
            if self._sensor.image_error:
                self._camera_label.setText(f"カメラ受信エラー: {self._sensor.image_error}")
            return
        height, width, _ = image.shape
        qimage = QImage(image.data, width, height, image.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self._camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._camera_label.setPixmap(pixmap)

    def _transformed_points(self) -> NDArray[np.float32]:
        points = self._sensor.latest_points
        if points is None or points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        if self._rotation_matrix is not None:
            points = apply_lidar_rotation(points, rotation_matrix=self._rotation_matrix)
        if points.shape[0] > MAX_DISPLAY_POINTS:
            step = int(math.ceil(points.shape[0] / MAX_DISPLAY_POINTS))
            points = points[::step]
        return points

    def _draw_pointcloud(self) -> None:
        axes = self._axes
        elevation, azimuth = axes.elev, axes.azim
        xlim, ylim, zlim = axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d()
        axes.clear()
        axes.set_facecolor("#1c1c1c")
        points = self._transformed_points()
        minimum = np.array([
            self._values["roi_x_min"], self._values["roi_y_min"], self._values["roi_z_min"]
        ], dtype=np.float32)
        maximum = np.array([
            self._values["roi_x_max"], self._values["roi_y_max"], self._values["roi_z_max"]
        ], dtype=np.float32)

        if points.shape[0] > 0:
            inside = np.all((points >= minimum) & (points <= maximum), axis=1)
            outside_points = points[~inside]
            inside_points = points[inside]
            if outside_points.shape[0] > 0:
                axes.scatter(
                    *outside_points.T,
                    s=3,
                    c="#a0a0a0",
                    alpha=0.35,
                    depthshade=False,
                )
            if inside_points.shape[0] > 0:
                axes.scatter(
                    *inside_points.T,
                    s=7,
                    c="#35d0ba",
                    alpha=0.95,
                    depthshade=False,
                )

        self._draw_roi_box(minimum, maximum)
        axes.set_xlabel("X", color=AXIS_COLORS["x"], fontweight="bold")
        axes.set_ylabel("Y", color=AXIS_COLORS["y"], fontweight="bold")
        axes.set_zlabel("Z", color=AXIS_COLORS["z"], fontweight="bold")
        axes.tick_params(axis="x", colors="#f2f2f2")
        axes.tick_params(axis="y", colors="#f2f2f2")
        axes.tick_params(axis="z", colors="#f2f2f2")
        axes.set_xlim3d(xlim)
        axes.set_ylim3d(ylim)
        axes.set_zlim3d(zlim)
        axes.set_box_aspect((1.0, 1.0, 0.8))
        axes.view_init(elev=elevation, azim=azimuth)
        axes.grid(True, alpha=0.25)
        self._canvas.draw_idle()

    def _draw_roi_box(self, minimum: NDArray[np.float32], maximum: NDArray[np.float32]) -> None:
        x0, y0, z0 = minimum
        x1, y1, z1 = maximum
        vertices = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        faces = [
            vertices[[0, 1, 2, 3]], vertices[[4, 5, 6, 7]],
            vertices[[0, 1, 5, 4]], vertices[[2, 3, 7, 6]],
            vertices[[1, 2, 6, 5]], vertices[[0, 3, 7, 4]],
        ]
        axes_faces = Poly3DCollection(
            faces,
            facecolors=[
                AXIS_COLORS["z"], AXIS_COLORS["z"],
                AXIS_COLORS["y"], AXIS_COLORS["y"],
                AXIS_COLORS["x"], AXIS_COLORS["x"],
            ],
            edgecolors="none",
            alpha=0.16,
        )
        self._axes.add_collection3d(axes_faces)
        edges = (
            (0, 1, "x"), (2, 3, "x"), (4, 5, "x"), (6, 7, "x"),
            (1, 2, "y"), (3, 0, "y"), (5, 6, "y"), (7, 4, "y"),
            (0, 4, "z"), (1, 5, "z"), (2, 6, "z"), (3, 7, "z"),
        )
        for first, second, axis in edges:
            segment = vertices[[first, second]]
            self._axes.plot(
                *segment.T, color=AXIS_COLORS[axis], linewidth=2.0
            )

    def _reset_view(self) -> None:
        self._axes.set_xlim3d(self._initial_limits[0])
        self._axes.set_ylim3d(self._initial_limits[1])
        self._axes.set_zlim3d(self._initial_limits[2])
        self._axes.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
        self._draw_pointcloud()

    def _discard_and_close(self) -> None:
        self.close()

    def _save_and_close(self) -> None:
        try:
            save_roi_config(self._config_path, self._config_text, self._values)
        except Exception as exc:
            if self._auto_test:
                print(f"[roi-tuner-test] save_error={exc}", flush=True)
                self._test_exit_code = 1
                self.close()
            else:
                QMessageBox.critical(self, "保存エラー", str(exc))
            return
        self.close()

    def _run_auto_test(self) -> None:
        if self._auto_test_done:
            return
        if self._sensor.image_messages > 0 and self._sensor.lidar_messages > 0:
            self._auto_test_done = True
            original = self._values["roi_x_min"]
            x_slider = self._range_sliders["x"]
            x_slider.setValues(x_slider.lowerValue + 1, x_slider.upperValue)
            slider_sync = math.isclose(
                self._spinboxes["roi_x_min"].value(), original + 0.01, abs_tol=1e-9
            )
            self._spinboxes["roi_x_min"].setValue(original)
            spin_sync = x_slider.lowerValue == int(round(original * SLIDER_SCALE))
            self._axes.view_init(elev=0.0, azim=0.0)
            self._axes.set_xlim3d((0.0, 1.0))
            self._reset_view()
            view_reset = (
                math.isclose(self._axes.elev, VIEW_ELEVATION, abs_tol=1e-9)
                and math.isclose(self._axes.azim, VIEW_AZIMUTH, abs_tol=1e-9)
                and np.allclose(self._axes.get_xlim3d(), self._initial_limits[0])
            )
            self._draw_camera()
            self._draw_pointcloud()
            screenshot = os.environ.get("ROI_TUNER_SCREENSHOT_PATH", "")
            if screenshot:
                self.grab().save(screenshot)
            print(
                "[roi-tuner-test] camera=yes lidar=yes roi_box=yes "
                f"slider_sync={'yes' if slider_sync else 'no'} "
                f"spin_sync={'yes' if spin_sync else 'no'} "
                f"view_reset={'yes' if view_reset else 'no'}",
                flush=True,
            )
            if not slider_sync or not spin_sync or not view_reset:
                self._test_exit_code = 1
            if self._auto_test == "save-change":
                self._spinboxes["roi_x_min"].setValue(original + 0.01)
                self._save_and_close()
            elif self._auto_test == "save":
                self._save_and_close()
            elif self._auto_test == "close":
                self.close()
            else:
                self._discard_and_close()
        elif time.monotonic() >= self._auto_test_deadline:
            self._auto_test_done = True
            print("[roi-tuner-test] sensor_timeout=yes", flush=True)
            self._test_exit_code = 1
            self.close()

    @property
    def test_exit_code(self) -> int:
        return self._test_exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()

    try:
        load_roi_config(arguments.config)
    except Exception as exc:
        print(f"[roi-tuner] {exc}", flush=True)
        return 1

    rclpy.init()
    sensor = SensorNode()
    application = QApplication([])
    window = RoiTunerWindow(arguments.config, sensor)
    window.showMaximized()
    qt_result = application.exec_()
    result = window.test_exit_code or qt_result
    sensor.destroy_node()
    rclpy.shutdown()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
