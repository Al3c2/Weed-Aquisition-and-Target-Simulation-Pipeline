#!/usr/bin/env python3
"""
Real-Time Continuous Beam Pointing System
==========================================

This node does EVERYTHING in real-time:
1. Color detection & tracking
2. Line following with speed modulation
3. DIRECT UR3 control for beam pointing (no separate planner!)

Key Features:
- Publishes joint commands DIRECTLY to /joint_trajectory_controller/joint_trajectory
- Real-time beam tracking while vehicle moves
- No MoveIt planning delays
- Continuous operation - never stops
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Bool, Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cv_bridge import CvBridge
import cv2

import tf2_ros
import tf2_geometry_msgs


class SystemState(Enum):
    """States of the real-time continuous system"""
    NORMAL_NAVIGATION = "normal_navigation"
    TRACKING_TARGET = "tracking_target"      # Actively tracking and pointing


@dataclass
class TrackedTarget:
    """A target being tracked in world frame"""
    id: int
    world_x: float
    world_y: float
    world_z: float
    first_seen_time: float
    last_seen_time: float
    times_seen: int = 1
    being_tracked: bool = False
    confidence: float = 1.0
    position_locked: bool = False
    
    def distance_to(self, x: float, y: float, z: float) -> float:
        return math.sqrt(
            (self.world_x - x)**2 + 
            (self.world_y - y)**2 + 
            (self.world_z - z)**2
        )
    
    def update(self, x: float, y: float, z: float, time: float, alpha: float = 0.3):
        """Update position with exponential moving average"""
        self.last_seen_time = time
        self.times_seen += 1
        self.confidence = min(1.0, self.confidence + 0.1)
        
        if not self.position_locked:
            if self.times_seen <= 3:
                self.world_x = alpha * x + (1 - alpha) * self.world_x
                self.world_y = alpha * y + (1 - alpha) * self.world_y
                self.world_z = alpha * z + (1 - alpha) * self.world_z
            else:
                self.position_locked = True


def yaw_from_quat(q):
    """Extract yaw angle from quaternion"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_pi(angle):
    """Wrap angle to [-pi, pi]"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


class RealTimeContinuousBeamSystem(Node):
    """
    Real-time system that:
    - Follows line continuously
    - Detects and tracks targets
    - Controls UR3 joints DIRECTLY for beam pointing while moving
    """
    
    def __init__(self):
        super().__init__('realtime_continuous_beam_system')
        
        # ==================== Parameters ====================
        
        # Camera topics
        self.declare_parameter('rgb_topic', '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/image')
        self.declare_parameter('depth_topic', '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/depth_image')
        self.declare_parameter('camera_info_topic', '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/camera_info')
        
        # Robot state
        self.declare_parameter('use_ground_truth', True)
        self.declare_parameter('ground_truth_topic', '/model/swincar_ur3/pose')
        self.declare_parameter('odom_topic', '/swincar/odom')
        self.declare_parameter('joint_states_topic', '/joint_states')
        
        # Base link offset
        self.declare_parameter('base_link_offset_x', 0.0)
        self.declare_parameter('base_link_offset_y', 0.2)
        self.declare_parameter('base_link_offset_z', 0.35)
        
        # Output topics
        self.declare_parameter('cmd_vel_topic', '/swincar/cmd_vel')
        self.declare_parameter('joint_trajectory_topic', '/joint_trajectory_controller/joint_trajectory')
        
        # Detection parameters
        self.declare_parameter('camera_hfov', 0.6)  # 34.4° horizontal FOV (RealSense D435)
        self.declare_parameter('association_radius', 0.08)
        self.declare_parameter('min_observations', 2)
        self.declare_parameter('track_timeout', 2.0)
        self.declare_parameter('min_blob_area', 200.0)
        
        # Trigger zone
        self.declare_parameter('trigger_min_y', 0.6)
        self.declare_parameter('trigger_max_y', 1.5)
        self.declare_parameter('trigger_min_x', -0.3)
        self.declare_parameter('trigger_max_x', 0.3)
        
        # Navigation
        self.declare_parameter('x_start', -1.0)
        self.declare_parameter('x_goal', -99.0)
        self.declare_parameter('y_target', 0.0)
        self.declare_parameter('goal_tolerance', 0.5)
        
        # Speed parameters
        self.declare_parameter('normal_speed', 0.8)
        self.declare_parameter('tracking_speed', 0.3)  # Speed when actively tracking
        self.declare_parameter('max_angular_speed', 1.5)
        
        # Pure pursuit
        self.declare_parameter('lookahead_distance', 3.0)
        self.declare_parameter('lateral_gain', 2.0)
        self.declare_parameter('heading_gain', 1.5)
        self.declare_parameter('direction_sign', -1.0)
        
        # UR3 control parameters
        self.declare_parameter('joint_control_rate', 20.0)  # Hz for joint updates
        self.declare_parameter('pointing_p_gain', 2.0)       # Proportional gain for pointing
        self.declare_parameter('max_joint_velocity', 1.0)    # rad/s
        
        self._load_parameters()
        
        # ==================== State Variables ====================
        
        self.system_state = SystemState.NORMAL_NAVIGATION
        self.current_target: Optional[TrackedTarget] = None
        self.current_target_id: Optional[int] = None
        
        # Speed control
        self.current_speed = self.normal_speed
        self.target_speed = self.normal_speed
        self.speed_transition_rate = 2.0
        
        # Camera state
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth = None
        self.bridge = CvBridge()
        
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.robot_yaw = 0.0
        self.odom_received = False
        self.goal_reached = False
        
        # UR3 joint state
        self.current_joint_positions = [0.0] * 6  # shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joints_received = False
        
        # Tracking
        self.tracked_targets: Dict[int, TrackedTarget] = {}
        self.next_target_id = 0
        self.completed_targets = []  # Targets we've already pointed at
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ==================== Subscriptions ====================
        
        self.rgb_sub = self.create_subscription(
            Image, self.get_parameter('rgb_topic').value, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self.depth_callback, 10)
        self.cam_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self.cam_info_callback, 10)
        
        # Robot pose
        if self.use_ground_truth:
            gt_topic = self.get_parameter('ground_truth_topic').value
            self.ground_truth_frame_id = 'swincar_ur3'  # Gazebo publishes in this frame
            self.pose_sub = self.create_subscription(
                PoseStamped, gt_topic, self.ground_truth_callback, 10)
            self.get_logger().info(f"✓ Using GROUND TRUTH from {gt_topic}")
        else:
            self.odom_sub = self.create_subscription(
                Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)
        
        # Joint states
        self.joint_sub = self.create_subscription(
            JointState, self.get_parameter('joint_states_topic').value,
            self.joint_state_callback, 10)
        
        # ==================== Publishers ====================
        
        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.joint_traj_pub = self.create_publisher(
            JointTrajectory, self.get_parameter('joint_trajectory_topic').value, 10)
        
        # ==================== Timers ====================
        
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz
        self.joint_control_timer = self.create_timer(
            1.0 / self.joint_control_rate, self.joint_control_loop)  # Real-time joint control
        self.log_timer = self.create_timer(1.0, self.log_status)
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("🚀 Real-Time Continuous Beam System Started")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"Normal speed: {self.normal_speed:.2f} m/s")
        self.get_logger().info(f"Tracking speed: {self.tracking_speed:.2f} m/s")
        self.get_logger().info(f"Joint control rate: {self.joint_control_rate:.0f} Hz")
        self.get_logger().info("=" * 70)
    
    def _load_parameters(self):
        """Load all parameters"""
        self.camera_hfov = self.get_parameter('camera_hfov').value
        self.association_radius = self.get_parameter('association_radius').value
        self.min_observations = self.get_parameter('min_observations').value
        self.track_timeout = self.get_parameter('track_timeout').value
        self.min_blob_area = self.get_parameter('min_blob_area').value
        
        self.base_link_offset_x = self.get_parameter('base_link_offset_x').value
        self.base_link_offset_y = self.get_parameter('base_link_offset_y').value
        self.base_link_offset_z = self.get_parameter('base_link_offset_z').value
        
        self.trigger_min_x = self.get_parameter('trigger_min_x').value
        self.trigger_max_x = self.get_parameter('trigger_max_x').value
        self.trigger_min_y = self.get_parameter('trigger_min_y').value
        self.trigger_max_y = self.get_parameter('trigger_max_y').value
        
        self.use_ground_truth = self.get_parameter('use_ground_truth').value
        
        self.x_start = self.get_parameter('x_start').value
        self.x_goal = self.get_parameter('x_goal').value
        self.y_target = self.get_parameter('y_target').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        
        self.normal_speed = abs(self.get_parameter('normal_speed').value)
        self.tracking_speed = abs(self.get_parameter('tracking_speed').value)
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        
        self.lookahead = self.get_parameter('lookahead_distance').value
        self.k_lateral = self.get_parameter('lateral_gain').value
        self.k_heading = self.get_parameter('heading_gain').value
        self.dir_sign = self.get_parameter('direction_sign').value
        
        self.joint_control_rate = self.get_parameter('joint_control_rate').value
        self.pointing_p_gain = self.get_parameter('pointing_p_gain').value
        self.max_joint_velocity = self.get_parameter('max_joint_velocity').value
    
    # ==================== Callbacks ====================
    
    def ground_truth_callback(self, msg: PoseStamped):
        """Update robot position from ground truth (Gazebo)"""
        # Check frame_id matches (Gazebo publishes in 'empty' frame)
        if msg.header.frame_id != self.ground_truth_frame_id:
            if not self.odom_received:
                self.get_logger().warn(
                    f"Ground truth frame_id mismatch: got '{msg.header.frame_id}', "
                    f"expected '{self.ground_truth_frame_id}'"
                )
            return
        
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.robot_z = msg.pose.position.z
        self.robot_yaw = yaw_from_quat(msg.pose.orientation)
        
        if not self.odom_received:
            self.odom_received = True
            self.get_logger().info(
                f"✓ First pose: x={self.robot_x:.2f}, y={self.robot_y:.2f}, z={self.robot_z:.2f} "
                f"(frame: {msg.header.frame_id})"
            )
    
    def odom_callback(self, msg: Odometry):
        """Update robot position from odometry"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_z = msg.pose.pose.position.z
        self.robot_yaw = yaw_from_quat(msg.pose.pose.orientation)
        
        if not self.odom_received:
            self.odom_received = True
    
    def joint_state_callback(self, msg: JointState):
        """Update current joint positions"""
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joint_positions[i] = msg.position[idx]
        
        if not self.joints_received:
            self.joints_received = True
            self.get_logger().info("Joint states received")
    
    def cam_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics"""
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            
            # Verify and correct if needed
            # For RealSense D435: width=848, height=480, hfov=0.6 rad (34.4°)
            # Correct fx should be: width / (2 * tan(hfov/2)) = 848 / (2 * tan(0.3)) ≈ 1378
            expected_fx = 848.0 / (2.0 * math.tan(0.6 / 2.0))  # hfov = 0.6 rad
            
            if abs(self.fx - expected_fx) > 100:  # Significant difference
                self.get_logger().warn(
                    f"Camera fx mismatch! Got {self.fx:.1f}, expected ~{expected_fx:.1f}. "
                    f"Using corrected values."
                )
                self.fx = expected_fx
                self.fy = expected_fx  # Assume square pixels
                self.cx = 848.0 / 2.0  # Image center
                self.cy = 480.0 / 2.0
            
            self.get_logger().info(f"✓ Camera: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
    
    def depth_callback(self, msg: Image):
        """Store latest depth image"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f"Depth error: {e}")
    
    def rgb_callback(self, msg: Image):
        """Main detection and tracking callback"""
        if self.fx is None or self.latest_depth is None:
            return
        
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Detect blue blobs
        blobs = self.find_blue_blobs(bgr_image)
        
        # Convert to 3D world coordinates
        detections = []
        for blob in blobs:
            depth_val = self.get_averaged_depth(blob['cx_px'], blob['cy_px'])
            if depth_val is None or depth_val <= 0:
                continue
            
            world_coords = self._pixel_to_world(blob['cx_px'], blob['cy_px'], depth_val)
            if world_coords is not None:
                detections.append({
                    'x': world_coords[0],
                    'y': world_coords[1],
                    'z': world_coords[2],
                })
        
        # Update tracking
        self._update_tracks(detections, current_time)
        
        # Select best target to track
        self._select_tracking_target(current_time)
    
    # ==================== Real-Time Joint Control ====================
    
    def joint_control_loop(self):
        """Real-time joint control loop - runs at high frequency"""
        if not self.joints_received or not self.odom_received:
            return
        
        if self.system_state == SystemState.TRACKING_TARGET and self.current_target is not None:
            # Compute desired joint angles to point at target
            desired_joints = self._compute_pointing_joints(self.current_target)
            
            if desired_joints is not None:
                # Publish joint trajectory
                self._publish_joint_trajectory(desired_joints, duration_sec=0.1)
        else:
            # Return to home position when not tracking
            home_position = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]  # Adjust as needed
            self._publish_joint_trajectory(home_position, duration_sec=0.5)
    
    def _compute_pointing_joints(self, target: TrackedTarget) -> Optional[List[float]]:
        """
        Compute joint angles to point at target using simple geometric inverse kinematics
        This is a simplified version - for real use, you'd want full IK
        """
        # Get target in robot base_link frame
        robot_coords = self._get_robot_frame_coords(target)
        if robot_coords is None:
            return None
        
        tx, ty, tz = robot_coords
        
        # Simple 2-DOF pointing (shoulder_pan and shoulder_lift)
        # This assumes the arm can point like a "laser pointer"
        
        # Pan angle (rotation around Z axis)
        pan_angle = math.atan2(ty, tx)
        
        # Tilt angle (elevation)
        horizontal_dist = math.sqrt(tx**2 + ty**2)
        tilt_angle = math.atan2(tz, horizontal_dist)
        
        # Create joint configuration
        # Adjust these based on your UR3 setup
        desired_joints = [
            pan_angle,           # shoulder_pan_joint
            -1.57 - tilt_angle,  # shoulder_lift_joint (adjust offset)
            1.57,                # elbow_joint (keep extended)
            -1.57,               # wrist_1_joint
            -1.57,               # wrist_2_joint
            0.0                  # wrist_3_joint
        ]
        
        return desired_joints
    
    def _publish_joint_trajectory(self, joint_positions: List[float], duration_sec: float = 0.1):
        """Publish joint trajectory command"""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        
        # Compute velocities based on position error
        velocities = []
        for i in range(len(joint_positions)):
            error = joint_positions[i] - self.current_joint_positions[i]
            velocity = self.pointing_p_gain * error
            velocity = clamp(velocity, -self.max_joint_velocity, self.max_joint_velocity)
            velocities.append(velocity)
        
        point.velocities = velocities
        point.time_from_start = Duration(seconds=duration_sec).to_msg()
        
        traj.points = [point]
        self.joint_traj_pub.publish(traj)
    
    # ==================== Navigation Control ====================
    
    def control_loop(self):
        """Main navigation control with speed modulation"""
        if not self.odom_received:
            return
        
        # Check goal
        distance_to_goal = abs(self.robot_x - self.x_goal)
        if distance_to_goal < self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info("🏁 GOAL REACHED!")
                self.goal_reached = True
            self.publish_cmd_vel(0.0, 0.0)
            return
        
        # Set target speed based on state
        if self.system_state == SystemState.TRACKING_TARGET:
            self.target_speed = self.tracking_speed
        else:
            self.target_speed = self.normal_speed
        
        # Smooth speed transition
        if abs(self.current_speed - self.target_speed) > 0.01:
            speed_diff = self.target_speed - self.current_speed
            max_change = self.speed_transition_rate * 0.05
            change = clamp(speed_diff, -max_change, max_change)
            self.current_speed += change
        else:
            self.current_speed = self.target_speed
        
        # Pure pursuit
        direction = 1.0 if self.x_goal > self.robot_x else -1.0
        lookahead_x = self.robot_x + direction * self.lookahead
        lookahead_y = self.y_target
        
        dx = lookahead_x - self.robot_x
        dy = lookahead_y - self.robot_y
        desired_yaw = math.atan2(dy, dx)
        
        heading_error = wrap_pi(desired_yaw - self.robot_yaw)
        lateral_error = self.robot_y - self.y_target
        
        angular_velocity = (
            self.k_heading * heading_error - 
            self.k_lateral * lateral_error
        )
        angular_velocity = clamp(angular_velocity, -self.max_angular_speed, self.max_angular_speed)
        
        linear_velocity = self.dir_sign * self.current_speed
        
        self.publish_cmd_vel(linear_velocity, angular_velocity)
    
    def publish_cmd_vel(self, linear: float, angular: float):
        """Publish velocity command"""
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)
    
    # ==================== Target Tracking ====================
    
    def _select_tracking_target(self, current_time: float):
        """Select best target to track in real-time"""
        best_target = None
        best_target_id = None
        best_score = -float('inf')
        
        for tid, track in self.tracked_targets.items():
            if track.times_seen < self.min_observations:
                continue
            
            # Skip if already completed
            if tid in self.completed_targets:
                continue
            
            # Check if in trigger zone
            robot_coords = self._get_robot_frame_coords(track)
            if robot_coords is None:
                continue
            
            rx, ry, rz = robot_coords
            
            in_zone = (self.trigger_min_x <= rx <= self.trigger_max_x and
                      self.trigger_min_y <= ry <= self.trigger_max_y)
            
            if not in_zone:
                continue
            
            # Score based on forward distance (prioritize closest)
            score = -ry  # Negative because we want smallest y
            
            if score > best_score:
                best_score = score
                best_target = track
                best_target_id = tid
        
        # Update state
        if best_target is not None:
            if self.current_target_id != best_target_id:
                self.get_logger().info(f"🎯 Now tracking target #{best_target_id}")
                self.current_target = best_target
                self.current_target_id = best_target_id
                self.system_state = SystemState.TRACKING_TARGET
        else:
            # No valid target, return to normal navigation
            if self.system_state == SystemState.TRACKING_TARGET:
                self.get_logger().info("📍 Target lost, resuming navigation")
                if self.current_target_id is not None:
                    self.completed_targets.append(self.current_target_id)
                self.current_target = None
                self.current_target_id = None
                self.system_state = SystemState.NORMAL_NAVIGATION
    
    def _pixel_to_world(self, px: float, py: float, depth: float) -> Optional[tuple]:
        """Convert pixel + depth to world coordinates"""
        x_cam = (px - self.cx) * depth / self.fx
        y_cam = (py - self.cy) * depth / self.fy
        z_cam = depth
        
        base_link_world_x = self.robot_x + self.base_link_offset_x
        base_link_world_y = self.robot_y + self.base_link_offset_y
        base_link_world_z = self.robot_z + self.base_link_offset_z
        
        world_x = base_link_world_x + y_cam
        world_y = base_link_world_y + x_cam
        world_z = base_link_world_z + z_cam
        
        return (world_x, world_y, world_z)
    
    def _update_tracks(self, detections: List[dict], current_time: float):
        """Update tracked targets"""
        used_detections = set()
        
        for tid, track in list(self.tracked_targets.items()):
            best_detection = None
            best_distance = float('inf')
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                
                dist = track.distance_to(det['x'], det['y'], det['z'])
                if dist < self.association_radius and dist < best_distance:
                    best_distance = dist
                    best_detection = (i, det)
            
            if best_detection is not None:
                i, det = best_detection
                track.update(det['x'], det['y'], det['z'], current_time)
                used_detections.add(i)
        
        # Create new tracks
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_track = TrackedTarget(
                    id=self.next_target_id,
                    world_x=det['x'],
                    world_y=det['y'],
                    world_z=det['z'],
                    first_seen_time=current_time,
                    last_seen_time=current_time
                )
                self.tracked_targets[self.next_target_id] = new_track
                self.next_target_id += 1
        
        # Remove stale tracks
        stale_ids = [
            tid for tid, track in self.tracked_targets.items()
            if current_time - track.last_seen_time > self.track_timeout
        ]
        for tid in stale_ids:
            del self.tracked_targets[tid]
    
    def _get_robot_frame_coords(self, track: TrackedTarget) -> Optional[tuple]:
        """Convert world to robot frame"""
        base_link_world_x = self.robot_x + self.base_link_offset_x
        base_link_world_y = self.robot_y + self.base_link_offset_y
        base_link_world_z = self.robot_z + self.base_link_offset_z
        
        rx = track.world_x - base_link_world_x
        ry = track.world_y - base_link_world_y
        rz = track.world_z - base_link_world_z
        return (rx, ry, rz)
    
    # ==================== Detection ====================
    
    def find_blue_blobs(self, bgr_image) -> List[dict]:
        """Detect blue blobs"""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_blob_area:
                continue
            
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            
            blobs.append({'cx_px': cx, 'cy_px': cy, 'area': area})
        
        return blobs
    
    def get_averaged_depth(self, cx: float, cy: float, radius: int = 5) -> Optional[float]:
        """Get depth at pixel"""
        if self.latest_depth is None:
            return None
        
        h, w = self.latest_depth.shape
        y_min = max(0, int(cy - radius))
        y_max = min(h, int(cy + radius + 1))
        x_min = max(0, int(cx - radius))
        x_max = min(w, int(cx + radius + 1))
        
        roi = self.latest_depth[y_min:y_max, x_min:x_max]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        
        if len(valid) < 3:
            return None
        
        return float(np.median(valid))
    
    # ==================== Logging ====================
    
    def log_status(self):
        """Status logging"""
        if not self.odom_received or self.goal_reached:
            return
        
        state_name = self.system_state.value
        speed_pct = (self.current_speed / self.normal_speed) * 100
        
        tracking_info = ""
        if self.current_target is not None:
            tracking_info = f"Target #{self.current_target_id}"
        
        self.get_logger().info(
            f"State: {state_name:20s} | "
            f"Speed: {self.current_speed:.2f} ({speed_pct:.0f}%) | "
            f"Pos: x={self.robot_x:6.2f} y={self.robot_y:6.2f} | "
            f"{tracking_info}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RealTimeContinuousBeamSystem()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()