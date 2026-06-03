#!/usr/bin/env python3
"""
Swincar Line Follower - Ground Truth Version with Target Stop
- Uses ground truth pose (not odometry) for accurate control
- Stops for targets, waits for beam completion
- ALWAYS publishes fresh /swincar_stopped confirmation
- Smooth acceleration to avoid jerky starts
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from std_msgs.msg import Bool


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class SwincarLineFollower(Node):
    def __init__(self):
        super().__init__("swincar_line_follower")

        # Parameters
        self.declare_parameter("cmd_vel_topic", "/swincar/cmd_vel")
        self.declare_parameter("pose_topic", "/model/swincar_ur3/pose")
        self.declare_parameter("target_topic", "/blue_target_primary")
        self.declare_parameter("beam_done_topic", "/beam_task_done")
        self.declare_parameter("beam_failed_topic", "/beam_task_failed")
        self.declare_parameter("swincar_stopped_topic", "/swincar_stopped")
        self.declare_parameter("x_goal", -99.0)
        self.declare_parameter("y_target", 0.0)
        self.declare_parameter("goal_tolerance", 1.5)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("base_speed", 1.0)
        self.declare_parameter("max_angular", 1.5)
        self.declare_parameter("k_heading", 2.0)
        self.declare_parameter("lookahead", 3.0)
        # Smooth acceleration parameters
        self.declare_parameter("accel_rate", 0.1)  # m/s per second
        self.declare_parameter("decel_rate", 0.5)  # m/s per second (faster decel)

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        pose_topic = self.get_parameter("pose_topic").value
        target_topic = self.get_parameter("target_topic").value
        beam_done_topic = self.get_parameter("beam_done_topic").value
        beam_failed_topic = self.get_parameter("beam_failed_topic").value
        swincar_stopped_topic = self.get_parameter("swincar_stopped_topic").value
        self.x_goal = self.get_parameter("x_goal").value
        self.y_target = self.get_parameter("y_target").value
        self.goal_tolerance = self.get_parameter("goal_tolerance").value
        self.control_rate = self.get_parameter("control_rate").value
        self.base_speed = abs(self.get_parameter("base_speed").value)
        self.max_angular = self.get_parameter("max_angular").value
        self.k_heading = self.get_parameter("k_heading").value
        self.lookahead = self.get_parameter("lookahead").value
        self.accel_rate = self.get_parameter("accel_rate").value
        self.decel_rate = self.get_parameter("decel_rate").value

        # State
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.pose_ok = False
        self.goal_reached = False
        
        # Target/beam state
        self.stopped_for_target = False
        self.current_target = None
        
        # Smooth speed control
        self.current_speed = 0.0  # Current actual speed (ramped)
        self.dt = 1.0 / self.control_rate

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.stopped_pub = self.create_publisher(Bool, swincar_stopped_topic, 10)
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self.pose_cb, 10
        )
        self.target_sub = self.create_subscription(
            PointStamped, target_topic, self.target_cb, 10
        )
        self.beam_done_sub = self.create_subscription(
            Bool, beam_done_topic, self.beam_done_cb, 10
        )
        self.beam_failed_sub = self.create_subscription(
            Bool, beam_failed_topic, self.beam_failed_cb, 10
        )
        
        self.timer = self.create_timer(self.dt, self.control)
        self.log_timer = self.create_timer(2.0, self.log_status)

        self.get_logger().info(f"=== Swincar Line Follower ===")
        self.get_logger().info(f"Pose: {pose_topic}")
        self.get_logger().info(f"Target: {target_topic}")
        self.get_logger().info(f"Beam done: {beam_done_topic}")
        self.get_logger().info(f"Beam failed: {beam_failed_topic}")
        self.get_logger().info(f"Stopped topic: {swincar_stopped_topic}")
        self.get_logger().info(f"Goal: x={self.x_goal}, y={self.y_target}")
        self.get_logger().info(f"Accel: {self.accel_rate} m/s², Decel: {self.decel_rate} m/s²")

    def pose_cb(self, msg: PoseStamped):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.yaw = yaw_from_quat(msg.pose.orientation)
        if not self.pose_ok:
            self.pose_ok = True
            self.get_logger().info(
                f"Pose OK: x={self.x:.2f}, y={self.y:.2f}, yaw={math.degrees(self.yaw):.1f}°"
            )

    def target_cb(self, msg: PointStamped):
        """Called when detector publishes a target - STOP and confirm!
        
        CRITICAL: Always publish /swincar_stopped = True, even if already stopped.
        The detector needs a FRESH confirmation for each stop request.
        """
        self.current_target = msg.point
        self.stopped_for_target = True
        
        self.get_logger().info(
            f"🎯 TARGET at ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}) - STOPPING"
        )
        
        # Immediately stop
        self.current_speed = 0.0
        self.stop()
        
        # ALWAYS publish fresh confirmation
        self._publish_stopped(True)

    def _publish_stopped(self, stopped: bool):
        """Publish stopped status"""
        msg = Bool()
        msg.data = stopped
        self.stopped_pub.publish(msg)
        self.get_logger().info(f"📡 Published /swincar_stopped = {stopped}")

    def beam_done_cb(self, msg: Bool):
        """Called when UR3 beam operation succeeded"""
        if msg.data and self.stopped_for_target:
            self.get_logger().info("✅ BEAM DONE - RESUMING")
            self._resume()

    def beam_failed_cb(self, msg: Bool):
        """Called when UR3 beam operation failed"""
        if msg.data and self.stopped_for_target:
            self.get_logger().warn("⚠️ BEAM FAILED - RESUMING")
            self._resume()

    def _resume(self):
        """Resume driving after beam completion"""
        self.stopped_for_target = False
        self.current_target = None
        # Don't reset current_speed - let it ramp up smoothly
        self._publish_stopped(False)

    def control(self):
        if not self.pose_ok:
            return
        
        # If stopped for target, stay stopped
        if self.stopped_for_target:
            self.current_speed = 0.0
            self.stop()
            return

        # Check goal reached
        dist_to_goal = abs(self.x - self.x_goal)
        if dist_to_goal < self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info(f"🏁 GOAL REACHED at x={self.x:.2f}, y={self.y:.2f}")
                self.goal_reached = True
            self.current_speed = 0.0
            self.stop()
            return

        # Pure pursuit control (driving in reverse since mesh faces +X)
        if self.x_goal < self.x:
            target_x = self.x - self.lookahead
        else:
            target_x = self.x + self.lookahead
        target_y = self.y_target

        dx = target_x - self.x
        dy = target_y - self.y
        angle_to_target = math.atan2(dy, dx)
        
        # Desired yaw so rear points at target
        desired_yaw = normalize_angle(angle_to_target - math.pi)
        yaw_error = normalize_angle(desired_yaw - self.yaw)
        
        # Inverted steering for reverse driving
        angular_z = -self.k_heading * yaw_error
        angular_z = max(-self.max_angular, min(self.max_angular, angular_z))

        # Speed control based on heading error
        if abs(yaw_error) > math.radians(30):
            target_speed = self.base_speed * 0.4
        elif abs(yaw_error) > math.radians(15):
            target_speed = self.base_speed * 0.7
        else:
            target_speed = self.base_speed

        # Smooth acceleration/deceleration
        if self.current_speed < target_speed:
            # Accelerating
            self.current_speed += self.accel_rate * self.dt
            self.current_speed = min(self.current_speed, target_speed)
        elif self.current_speed > target_speed:
            # Decelerating
            self.current_speed -= self.decel_rate * self.dt
            self.current_speed = max(self.current_speed, target_speed)

        # Negative because we're reversing
        linear_x = -self.current_speed

        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_pub.publish(cmd)

    def stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def log_status(self):
        if not self.pose_ok:
            return
        
        if self.stopped_for_target:
            status = "⏸️  STOPPED (waiting for beam)"
        elif self.goal_reached:
            status = "🏁 GOAL REACHED"
        else:
            status = f"🚗 DRIVING ({self.current_speed:.2f} m/s)"
        
        dist = abs(self.x - self.x_goal)
        self.get_logger().info(
            f"[{status}] x={self.x:+7.2f} y={self.y:+6.3f} | dist={dist:.1f}m"
        )

    def destroy_node(self):
        self.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SwincarLineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()