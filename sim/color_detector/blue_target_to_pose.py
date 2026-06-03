#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped
import tf2_ros
import tf2_geometry_msgs


class BlueTargetToPose(Node):
    def __init__(self):
        super().__init__('blue_target_to_pose')

        # Parameters
        self.declare_parameter('input_topic', '/blue_target_primary')
        self.declare_parameter('output_topic', '/target_pose')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('position_epsilon', 0.01)  # 1 cm threshold

        # Calibration offset in target_frame (base_link)
        # These default to 0; you can set them from launch/CLI.
        self.declare_parameter('calib_dx', 0.0)
        self.declare_parameter('calib_dy', 0.0)
        self.declare_parameter('calib_dz', 0.0)

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.position_epsilon = self.get_parameter('position_epsilon').get_parameter_value().double_value

        self.calib_dx = self.get_parameter('calib_dx').get_parameter_value().double_value
        self.calib_dy = self.get_parameter('calib_dy').get_parameter_value().double_value
        self.calib_dz = self.get_parameter('calib_dz').get_parameter_value().double_value

        self.get_logger().info(f"Subscribing to blue target: {input_topic}")
        self.get_logger().info(f"Publishing PoseStamped to: {output_topic}")
        self.get_logger().info(f"Target frame for pose: {self.target_frame}")
        self.get_logger().info(f"Position epsilon: {self.position_epsilon} m")
        self.get_logger().info(
            f"Calibration offset in {self.target_frame}: "
            f"dx={self.calib_dx:.3f}, dy={self.calib_dy:.3f}, dz={self.calib_dz:.3f}"
        )

        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Sub & Pub
        self.sub = self.create_subscription(
            PointStamped,
            input_topic,
            self.target_callback,
            10
        )

        self.pub = self.create_publisher(
            PoseStamped,
            output_topic,
            10
        )

        # Store last published target in target_frame
        self.last_x = None
        self.last_y = None
        self.last_z = None

    def target_callback(self, msg: PointStamped):
        source_frame = msg.header.frame_id or self.target_frame
        pt = msg

        # Only transform if needed
        if source_frame != self.target_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,   # to
                    source_frame,        # from
                    msg.header.stamp     # at detection time
                )
                pt = tf2_geometry_msgs.do_transform_point(msg, transform)
            except Exception as e:
                self.get_logger().warn(
                    f"TF transform {source_frame} -> {self.target_frame} failed: {e}"
                )
                return

        # Raw point in target_frame
        x = pt.point.x
        y = pt.point.y
        z = pt.point.z

        # Apply calibration offset (in target_frame coords)
        x_corr = x + self.calib_dx
        y_corr = y + self.calib_dy
        z_corr = z + self.calib_dz

        # Change detection: only publish if moved more than epsilon
        if self.last_x is not None:
            dx = x_corr - self.last_x
            dy = y_corr - self.last_y
            dz = z_corr - self.last_z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist < self.position_epsilon:
                return

        self.last_x = x_corr
        self.last_y = y_corr
        self.last_z = z_corr

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.target_frame

        pose.pose.position.x = x_corr
        pose.pose.position.y = y_corr
        pose.pose.position.z = z_corr

        # Simple identity orientation; beam node builds its own ray dir
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.pub.publish(pose)

        self.get_logger().info(
            f"NEW target pose -> /target_pose: "
            f"[{x_corr:.3f}, {y_corr:.3f}, {z_corr:.3f}] in {self.target_frame}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = BlueTargetToPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
