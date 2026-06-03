#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class SwincarLineFollower(Node):
    """Simple line follower: go from start to x_goal, keep y near y_ref."""

    def __init__(self):
        super().__init__("swincar_line_follower")

        # Parameters
        self.declare_parameter("cmd_vel_topic", "/swincar/cmd_vel")
        self.declare_parameter("odom_topic", "/swincar/odom")
        self.declare_parameter("x_goal", -99.0)
        self.declare_parameter("y_ref", 0.0)
        self.declare_parameter("tolerance", 0.30)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("speed", 0.5)
        self.declare_parameter("max_angular", 2.0)
        self.declare_parameter("lookahead", 2.0)  # Distance ahead to aim for
        self.declare_parameter("lateral_gain", 2.5)  # How aggressively to correct drift
        self.declare_parameter("yaw_gain", 3.0)  # Heading correction gain
        self.declare_parameter("startup_speed_factor", 0.3)  # Slow start until aligned
        self.declare_parameter("startup_y_threshold", 0.5)  # When to go full speed
        
        # For rotated mesh: set this to -1.0 if your "forward" is actually backward
        self.declare_parameter("direction_sign", -1.0)

        # Get params
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        self.x_goal = float(self.get_parameter("x_goal").value)
        self.y_ref = float(self.get_parameter("y_ref").value)
        self.tolerance = float(self.get_parameter("tolerance").value)
        rate_hz = float(self.get_parameter("control_rate").value)
        self.speed = abs(float(self.get_parameter("speed").value))
        self.max_w = float(self.get_parameter("max_angular").value)
        self.lookahead = float(self.get_parameter("lookahead").value)
        self.k_lateral = float(self.get_parameter("lateral_gain").value)
        self.k_yaw = float(self.get_parameter("yaw_gain").value)
        self.startup_factor = float(self.get_parameter("startup_speed_factor").value)
        self.startup_y_thresh = float(self.get_parameter("startup_y_threshold").value)
        self.dir_sign = float(self.get_parameter("direction_sign").value)

        # State
        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.goal_reached = False

        # Publishers/Subscribers
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.timer = self.create_timer(1.0 / rate_hz, self.control_loop)

        self.get_logger().info(f"Goal: x={self.x_goal:.2f}, y={self.y_ref:.2f}")
        self.get_logger().info(f"Speed: {self.speed:.2f} m/s, lookahead: {self.lookahead:.2f} m")
        self.get_logger().info(f"Direction sign: {self.dir_sign:+.0f}, lateral_gain: {self.k_lateral:.2f}")

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.have_odom = True

    def control_loop(self):
        if not self.have_odom:
            return

        # Check if goal reached (only check x since we want to stay on y_ref)
        dist_to_goal = abs(self.x_goal - self.x)
        lateral_error = abs(self.y - self.y_ref)
        
        if dist_to_goal < self.tolerance and lateral_error < self.tolerance:
            if not self.goal_reached:
                self.get_logger().info(f"Goal reached! x={self.x:.2f}, y={self.y:.2f}")
                self.goal_reached = True
            self.cmd_pub.publish(Twist())  # Stop
            return

        # Determine which direction along x-axis to go
        toward_goal_dir = 1.0 if (self.x_goal - self.x) > 0 else -1.0
        
        # Pure pursuit: aim for a point ahead on the line
        lookahead_x = self.x + toward_goal_dir * self.lookahead
        lookahead_y = self.y_ref  # Stay on the line
        
        # Calculate desired heading to lookahead point
        dx = lookahead_x - self.x
        dy = lookahead_y - self.y
        desired_yaw = math.atan2(dy, dx)
        
        # Yaw error
        yaw_error = wrap_pi(desired_yaw - self.yaw)
        
        # Lateral error (perpendicular distance to line y=y_ref)
        lateral_err = self.y - self.y_ref
        
        # Combined steering: correct both heading and lateral drift
        # NEGATIVE sign: if y > y_ref (positive error), steer negative to get back
        angular = self.k_yaw * yaw_error - self.k_lateral * lateral_err
        angular = clamp(angular, -self.max_w, self.max_w)

        # Reduce speed when far from centerline or badly misaligned (startup protection)
        speed_mult = 1.0
        if abs(lateral_err) > self.startup_y_thresh or abs(yaw_error) > math.radians(30):
            speed_mult = self.startup_factor
        
        # Constant forward speed (in the mesh's "forward" direction)
        linear = self.dir_sign * self.speed * speed_mult

        # Publish command
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)
        
        # Debug info every 2 seconds
        if not hasattr(self, '_last_log'):
            self._last_log = self.get_clock().now()
        if (self.get_clock().now() - self._last_log).nanoseconds * 1e-9 > 2.0:
            self.get_logger().info(
                f"x={self.x:.2f}, y={self.y:.2f}, y_err={lateral_err:.3f}, "
                f"yaw_err={math.degrees(yaw_error):.1f}°"
            )
            self._last_log = self.get_clock().now()

    def destroy_node(self):
        self.cmd_pub.publish(Twist())  # Stop on shutdown
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()