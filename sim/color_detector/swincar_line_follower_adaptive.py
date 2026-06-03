#!/usr/bin/env python3
"""
Swincar Line Follower - ADAPTIVE SPEED (Fixed with Pure Pursuit)

Combines stable pure pursuit control from working version
with adaptive speed control from detector.

- Uses GT pose with proper yaw-based steering (pure pursuit)
- Smooth acceleration/deceleration ramping
- Adaptive speed commanded by color detector via /target_slowdown_speed
- Stays responsive to detector's speed commands
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from std_msgs.msg import Bool, Float32


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


class SwincarLineFollowerAdaptive(Node):
    def __init__(self):
        super().__init__("swincar_line_follower_adaptive")

        # Parameters
        self.declare_parameter("cmd_vel_topic", "/swincar/cmd_vel")
        self.declare_parameter("pose_topic", "/model/swincar_ur3/pose")
        self.declare_parameter("target_topic", "/blue_target_primary")
        self.declare_parameter("beam_done_topic", "/beam_task_done")
        self.declare_parameter("beam_failed_topic", "/beam_task_failed")
        self.declare_parameter("target_slowdown_topic", "/target_slowdown_speed")
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
        # Adaptive speed parameters
        self.declare_parameter("slowdown_timeout", 10.0)

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        pose_topic = self.get_parameter("pose_topic").value
        target_topic = self.get_parameter("target_topic").value
        beam_done_topic = self.get_parameter("beam_done_topic").value
        beam_failed_topic = self.get_parameter("beam_failed_topic").value
        target_slowdown_topic = self.get_parameter("target_slowdown_topic").value
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
        self.slowdown_timeout = self.get_parameter("slowdown_timeout").value

        # State
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.pose_ok = False
        self.goal_reached = False
        
        # Target/beam state
        self.stopped_for_target = False
        self.current_target = None
        
        # Adaptive speed state
        self.detector_slowdown_active = False
        self.detector_speed = self.base_speed
        self.detector_slowdown_time = None
        
        # Smooth speed control
        self.current_speed = 0.0  # Current actual speed (ramped)
        self.dt = 1.0 / self.control_rate

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        
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
        self.slowdown_sub = self.create_subscription(
            Float32, target_slowdown_topic, self.slowdown_speed_cb, 10
        )
        
        self.timer = self.create_timer(self.dt, self.control)
        self.log_timer = self.create_timer(2.0, self.log_status)

        self.get_logger().info("=" * 60)
        self.get_logger().info("Swincar Adaptive Speed Line Follower (Pure Pursuit + Detector)")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Pose: {pose_topic}")
        self.get_logger().info(f"Target: {target_topic}")
        self.get_logger().info(f"Detector slowdown: {target_slowdown_topic}")
        self.get_logger().info(f"Beam done: {beam_done_topic}")
        self.get_logger().info(f"Beam failed: {beam_failed_topic}")
        self.get_logger().info(f"Goal: x={self.x_goal}, y={self.y_target}")
        self.get_logger().info(f"Base speed: {self.base_speed:.2f} m/s")
        self.get_logger().info(f"Accel: {self.accel_rate} m/s², Decel: {self.decel_rate} m/s²")
        self.get_logger().info("=" * 60)

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
        """Called when detector publishes a target - just log it, don't stop.
        
        Speed control is handled by slowdown_speed_cb.
        Robot keeps moving; detector commands speeds via /target_slowdown_speed.
        """
        self.current_target = msg.point
        
        self.get_logger().info(
            f"🎯 TARGET at ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f})"
        )

    def slowdown_speed_cb(self, msg: Float32):
        """Detector commands a specific speed for approach/targeting.
        
        This overrides base_speed and allows fine-grained control.
        """
        self.detector_speed = max(0.0, float(msg.data))
        self.detector_slowdown_active = True
        self.detector_slowdown_time = self.get_clock().now()
        
        self.get_logger().info(
            f"[DETECTOR] Speed command: {self.detector_speed:.3f} m/s"
        )

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

    # ── Control loop ──────────────────────────────────────────────────────────
    def control(self):
        if not self.pose_ok:
            return

        # Check if detector slowdown timeout expired
        if self.detector_slowdown_active:
            elapsed = (self.get_clock().now() - self.detector_slowdown_time).nanoseconds * 1e-9
            if elapsed > self.slowdown_timeout:
                self.get_logger().warn("[TIMEOUT] Detector slowdown expired - resuming base speed")
                self.detector_slowdown_active = False
                self.detector_speed = self.base_speed

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

        # Determine target speed: use detector speed if active, otherwise base speed
        if self.detector_slowdown_active:
            speed_target = self.detector_speed
        else:
            # Speed control based on heading error (normal operation)
            if abs(yaw_error) > math.radians(30):
                speed_target = self.base_speed * 0.4
            elif abs(yaw_error) > math.radians(15):
                speed_target = self.base_speed * 0.7
            else:
                speed_target = self.base_speed

        # Smooth acceleration/deceleration
        if self.current_speed < speed_target:
            # Accelerating
            self.current_speed += self.accel_rate * self.dt
            self.current_speed = min(self.current_speed, speed_target)
        elif self.current_speed > speed_target:
            # Decelerating
            self.current_speed -= self.decel_rate * self.dt
            self.current_speed = max(self.current_speed, speed_target)

        # Negative because we're reversing
        linear_x = -self.current_speed

        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_pub.publish(cmd)

    def stop(self):
        """Publish zero velocity"""
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
            if self.detector_slowdown_active:
                status = f"🎯 DETECTOR ({self.detector_speed:.3f} m/s)"
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
    node = SwincarLineFollowerAdaptive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()