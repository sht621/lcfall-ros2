#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np

class RawSensorPublisher(Node):
    def __init__(self):
        super().__init__('raw_sensor_publisher')
        self.img_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/livox/lidar', 10)
        
        # Camera: 30Hz, LiDAR: 10Hz
        self.create_timer(1.0 / 30.0, self.camera_callback)
        self.create_timer(1.0 / 10.0, self.lidar_callback)
        
        self.count = 0
        self.get_logger().info('Started publishing DUMMY RAW SENSOR DATA (Camera: 30Hz, LiDAR: 10Hz)')

    def camera_callback(self):
        now = self.get_clock().now()
        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = 'camera_link'

        # Dummy Image (真っ黒な画像)
        img_msg = Image()
        img_msg.header = header
        img_msg.height = 720
        img_msg.width = 1280
        img_msg.encoding = 'bgr8'
        img_msg.step = 1280 * 3
        img_msg.data = bytearray(1280 * 720 * 3)
        self.img_pub.publish(img_msg)
        
        if self.count % 30 == 0:
            self.get_logger().info(f'Published {self.count // 3} camera frames')

    def lidar_callback(self):
        now = self.get_clock().now()
        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = 'livox_frame'

        # Dummy PointCloud2
        pc_msg = PointCloud2()
        pc_msg.header = header
        pc_msg.height = 1
        pc_msg.width = 100 # 100 points
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.fields = fields
        pc_msg.is_bigendian = False
        pc_msg.point_step = 16
        pc_msg.row_step = 16 * 100
        pc_msg.is_dense = True
        
        points = np.random.rand(100, 4).astype(np.float32)
        pc_msg.data = points.tobytes()
        self.lidar_pub.publish(pc_msg)
        
        self.count += 1
        if self.count % 10 == 0:
            self.get_logger().info(f'Published {self.count} LiDAR frames (Camera is running at 30Hz)')

def main(args=None):
    rclpy.init(args=args)
    node = RawSensorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
