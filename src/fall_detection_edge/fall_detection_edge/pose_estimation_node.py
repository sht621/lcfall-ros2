"""Selected ROS images -> direct Python/HailoRT -> all COCO17 person results."""
from copy import deepcopy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from fall_detection_msgs.msg import PoseFrame
from .common import Console, internal_qos, parameter, positive, run, stamp_ns
from .hailo_backend import HailoBackend
from .pose_processing import persons_from_rows


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
        self.console = Console(self)
        self.width = int(positive(parameter(self, 'image_width', 1280), 'image_width'))
        self.height = int(positive(parameter(self, 'image_height', 720), 'image_height'))
        hef_path = parameter(self, 'hef_path', '/models/yolov8m_pose.hef')
        manifest_path = parameter(self, 'model_manifest', '/models/pose_model.yaml')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(PoseFrame, '/pose_frame', internal_qos())
        try:
            self.backend = HailoBackend(hef_path, manifest_path)
        except Exception as exc:
            self.get_logger().error(f'Hailo initialization failed (requires Pi verification): {exc}')
            raise
        self.subscription = self.create_subscription(Image, '/sync/image', self.receive, internal_qos())
        self.get_logger().info('HailoRT connected; runtime/model compatibility remains subject to Pi validation')

    def receive(self, image):
        try:
            stamp_ns(image.header.stamp)
            if image.width != self.width or image.height != self.height:
                raise ValueError('RGB size differs from fixed server/camera configuration')
            rgb = self.bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
            rows, mapping = self.backend.infer(rgb)
            persons = persons_from_rows(rows, mapping)
            self.publisher.publish(PoseFrame(header=deepcopy(image.header), persons=persons))
            self.console.show('ok', f'PoseFrame published: {len(persons)} persons (zero is valid)')
        except Exception as exc:
            self.console.show('failed', f'Pose inference failed; no output: {exc}', True)

    def destroy_node(self):
        try:
            self.backend.close()
        finally:
            super().destroy_node()


def main(args=None):
    run(PoseEstimationNode, args)
