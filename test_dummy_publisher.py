#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from lcfall_msgs.msg import PreprocessedFrame
import numpy as np

class DummyPublisher(Node):
    def __init__(self):
        super().__init__('dummy_publisher')
        self.publisher_ = self.create_publisher(PreprocessedFrame, '/preprocessed/frame', 10)
        # Publish at 10Hz (Livox MID-360 の実際の周波数に合わせる)
        self.timer = self.create_timer(1.0 / 10.0, self.timer_callback)
        self.count = 0
        self.get_logger().info('Dummy Publisher started. Publishing 10 frames/sec to /preprocessed/frame')

    def timer_callback(self):
        msg = PreprocessedFrame()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'dummy_frame'
        
        # 1人分の2D skeleton座標 (17 * 3 = 51)
        # normalizeされた乱数として送信 (-1.0 ~ 1.0などでもOK)
        dummy_skeleton = np.random.rand(51).astype(np.float32).tolist()
        msg.skeleton_2d = dummy_skeleton
        
        # 前景生点群256点 (256 * 3 = 768)
        # [-1.0, 1.0]の範囲の乱数点群
        dummy_pc = (np.random.rand(768) * 2 - 1).astype(np.float32).tolist()
        msg.pointcloud_frame = dummy_pc
        
        self.publisher_.publish(msg)
        self.count += 1
        
        # 48フレーム(1推論分)たまったら間引いてログを出す
        if self.count % 48 == 0:
            self.get_logger().info(f'Published {self.count} frames (triggered inference logic)')

def main(args=None):
    rclpy.init(args=args)
    node = DummyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
