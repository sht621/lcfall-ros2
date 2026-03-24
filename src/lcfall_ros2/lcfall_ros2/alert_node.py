#!/usr/bin/env python3
"""alert_node: 転倒検知結果に基づくアラート通知.

/fall_detection/result を subscribe し、
prediction == 1 (falling) のとき notify_fall() を呼び出す。
publish はしない。
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from lcfall_msgs.msg import FallDetectionResult


class AlertNode(Node):
    """転倒アラート通知ノード.

    初版ではログ出力のみ。将来的にブザー制御、外部通知 API 呼び出し
    などに差し替えやすい構造にしてある。
    """

    def __init__(self) -> None:
        super().__init__("alert_node")

        self._sub = self.create_subscription(
            FallDetectionResult,
            "/fall_detection/result",
            self._result_callback,
            10,
        )

        self.get_logger().info("AlertNode initialized.")

    # ==================================================================
    # コールバック
    # ==================================================================

    def _result_callback(self, msg: FallDetectionResult) -> None:
        """転倒検知結果の処理.

        prediction == 1 → notify_fall()
        prediction == 0 → 何もしない
        """
        if msg.prediction == 1:
            self.notify_fall(msg)

    # ==================================================================
    # 通知メソッド (差し替え可能)
    # ==================================================================

    def notify_fall(self, msg: FallDetectionResult) -> None:
        """転倒検知時の通知処理.

        初版ではログ出力のみ行う。
        将来的にはこのメソッドを拡張して、以下のような処理に対応する:
          - ブザー / LED 制御
          - 外部通知 API (Slack, LINE, メール等)
          - データベースへのイベント記録
          - 音声警告

        Args:
            msg: 転倒検知結果メッセージ。
        """
        self.get_logger().warn(
            f"🚨 FALL DETECTED! "
            f"fall_prob={msg.confidence:.3f}, "
            f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AlertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
