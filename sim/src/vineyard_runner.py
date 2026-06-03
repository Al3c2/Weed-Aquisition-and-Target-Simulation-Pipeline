#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PointStamped


class VineyardRunner(Node):
    def __init__(self):
        super().__init__('vineyard_runner')

        # ===================== Parameters =====================
        self.declare_parameter('cmd_vel_topic', '/swincar/cmd_vel')
        self.declare_parameter('blue_topic', '/blue_target_primary')

        # Base motion behaviour
        self.declare_parameter('forward_speed', 0.6)               # m/s when driving
        self.declare_parameter('stop_duration', 4.0)               # s to stand still & let UR3 point
        self.declare_parameter('min_time_between_stops', 6.0)      # s, don’t stop again immediately
        self.declare_parameter('min_target_distance_change', 0.05) # m, ignore tiny jitter

        # "Gate" in camera frame for valid targets (in robot/base frame)
        self.declare_parameter('target_x_min', 0.5)  # don’t stop for things too close / behind
        self.declare_parameter('target_x_max', 3.0)  # only stop for things up to 3 m ahead
        self.declare_parameter('target_y_max_abs', 0.6)  # must be roughly in front (|y| <= this)

        # (Optional future extension) – switch to "wait for UR3 done" instead of fixed time
        self.declare_parameter('use_beam_done', False)
        self.declare_parameter('beam_done_topic', '/beam_pointing_done')

        # ===================== Get params =====================
        self.cmd_vel_topic = (
            self.get_parameter('cmd_vel_topic')
            .get_parameter_value()
            .string_value
        )
        blue_topic = (
            self.get_parameter('blue_topic')
            .get_parameter_value()
            .string_value
        )

        self.forward_speed = (
            self.get_parameter('forward_speed')
            .get_parameter_value()
            .double_value
        )
        self.stop_duration = (
            self.get_parameter('stop_duration')
            .get_parameter_value()
            .double_value
        )
        self.min_time_between_stops = (
            self.get_parameter('min_time_between_stops')
            .get_parameter_value()
            .double_value
        )
        self.min_target_distance_change = (
            self.get_parameter('min_target_distance_change')
            .get_parameter_value()
            .double_value
        )

        self.target_x_min = (
            self.get_parameter('target_x_min')
            .get_parameter_value()
            .double_value
        )
        self.target_x_max = (
            self.get_parameter('target_x_max')
            .get_parameter_value()
            .double_value
        )
        self.target_y_max_abs = (
            self.get_parameter('target_y_max_abs')
            .get_parameter_value()
            .double_value
        )

        self.use_beam_done = (
            self.get_parameter('use_beam_done')
            .get_parameter_value()
            .bool_value
        )
        self.beam_done_topic = (
            self.get_parameter('beam_done_topic')
            .get_parameter_value()
            .string_value
        )

        self.get_logger().info(
            f"VineyardRunner: driving on {self.cmd_vel_topic} at {self.forward_speed:.2f} m/s, "
            f"stop {self.stop_duration:.1f}s when target seen on {blue_topic}"
        )

        # ===================== ROS I/O =====================
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.blue_sub = self.create_subscription(
            PointStamped,
            blue_topic,
            self.blue_callback,
            10
        )

        # If later you add a Bool /beam_pointing_done, you can uncomment this:
        # if self.use_beam_done:
        #     from std_msgs.msg import Bool
        #     self.beam_done_sub = self.create_subscription(
        #         Bool,
        #         self.beam_done_topic,
        #         self.beam_done_callback,
        #         10
        #     )

        # ===================== State machine =====================
        self.state = "DRIVING"   # or "STOPPED"
        self.last_stop_time = -1.0    # sim time, seconds
        self.stop_start_time = None   # sim time, seconds

        # For simple "new target" detection
        self.last_target_x = None
        self.last_target_y = None
        self.last_target_z = None

        # Timer to publish cmd_vel
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

    # ---------------------- Helpers ----------------------
    def now(self) -> float:
        """Current time in seconds, using ROS (sim) time."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------- Main control loop ----------------------
    def timer_callback(self):
        """Periodic control of cmd_vel based on state."""
        cmd = Twist()

        if self.state == "DRIVING":
            cmd.linear.x = self.forward_speed
            cmd.angular.z = 0.0

        elif self.state == "STOPPED":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

            # If not using UR3-done handshake, use time-based resume
            if not self.use_beam_done and self.stop_start_time is not None:
                now = self.now()
                if now - self.stop_start_time >= self.stop_duration:
                    self.get_logger().info("Stop duration elapsed; resuming DRIVING.")
                    self.state = "DRIVING"
                    self.stop_start_time = None
                    self.last_stop_time = now

        self.cmd_pub.publish(cmd)

    # ---------------------- Blue target callback ----------------------
    def blue_callback(self, msg: PointStamped):
        """Called whenever a blue target is detected."""
        # Only react if we are currently driving
        if self.state != "DRIVING":
            return

        x = msg.point.x
        y = msg.point.y
        z = msg.point.z

        # 1) Gate: only stop for targets in front of the robot within a reasonable window.
        if not (self.target_x_min <= x <= self.target_x_max):
            return
        if abs(y) > self.target_y_max_abs:
            return

        # 2) Is this actually a new target, or just jitter of the same one?
        if self.last_target_x is not None:
            dx = x - self.last_target_x
            dy = y - self.last_target_y
            dz = z - self.last_target_z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < self.min_target_distance_change:
                # Same target / noise; ignore
                return

        # At this point, we consider it a new vine post / blue ball.
        self.last_target_x = x
        self.last_target_y = y
        self.last_target_z = z

        now = self.now()

        # 3) Respect minimum time between stops to avoid brake spam.
        if self.last_stop_time > 0.0 and (now - self.last_stop_time) < self.min_time_between_stops:
            return

        # 4) Trigger a stop.
        self.get_logger().info(
            f"🙋 Blue target detected at [{x:.3f}, {y:.3f}, {z:.3f}] "
            f"(gate x∈[{self.target_x_min},{self.target_x_max}], |y| ≤ {self.target_y_max_abs}) "
            f"-> STOP for {self.stop_duration:.1f}s"
        )
        self.state = "STOPPED"
        self.stop_start_time = now
        # From this point on, /blue_target_to_pose will keep publishing /target_pose
        # and your UR3 planner will point at the target while we are stopped.

    # ---------------------- Optional UR3 handshake ----------------------
    # def beam_done_callback(self, msg: Bool):
    #     """If you wire a 'UR3 done' Bool here, you can resume as soon as the beam finishes."""
    #     if self.state == "STOPPED" and msg.data:
    #         now = self.now()
    #         self.get_logger().info("UR3 beam done signal received; resuming DRIVING.")
    #         self.state = "DRIVING"
    #         self.stop_start_time = None
    #         self.last_stop_time = now


def main(args=None):
    rclpy.init(args=args)
    node = VineyardRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
