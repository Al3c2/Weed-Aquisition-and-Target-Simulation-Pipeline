#!/usr/bin/env python3

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose, PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool

from cv_bridge import CvBridge
import cv2
import numpy as np

import tf2_ros
import tf2_geometry_msgs

from scipy.spatial.transform import Rotation


@dataclass
class TrackedTarget:
    id: int
    world_x: float
    world_y: float
    world_z: float
    first_seen_time: float
    last_seen_time: float
    times_seen: int = 1
    published: bool = False
    position_locked: bool = False
    
    pos_history_x: List[float] = field(default_factory=list)
    pos_history_y: List[float] = field(default_factory=list)
    pos_history_z: List[float] = field(default_factory=list)
    quality_history: List[float] = field(default_factory=list)
    
    reacquired_after_stop: bool = False
    reacquisition_count: int = 0
    
    beam_attempts: int = 0
    max_beam_attempts: int = 3
    last_beam_attempt_time: float = 0.0
    beam_succeeded: bool = False
    
    needs_immediate_retry: bool = False
    skip_reacquisition: bool = False  # skip re-acquisition on retry
    
    def __post_init__(self):
        if not self.pos_history_x:
            self.pos_history_x = [self.world_x]
            self.pos_history_y = [self.world_y]
            self.pos_history_z = [self.world_z]
            self.quality_history = [1.0]
    
    def distance_3d(self, x: float, y: float, z: float) -> float:
        return math.sqrt(
            (self.world_x - x)**2 + 
            (self.world_y - y)**2 + 
            (self.world_z - z)**2
        )
    
    def clear_history_for_reacquisition(self):
        self.pos_history_x = []
        self.pos_history_y = []
        self.pos_history_z = []
        self.quality_history = []
        self.position_locked = False
        self.reacquisition_count = 0
    
    def update(self, x: float, y: float, z: float, time: float, quality: float = 1.0):
        self.last_seen_time = time
        self.times_seen += 1
        
        if not self.position_locked:
            self.pos_history_x.append(x)
            self.pos_history_y.append(y)
            self.pos_history_z.append(z)
            self.quality_history.append(quality)
            
            max_history = 30
            if len(self.pos_history_x) > max_history:
                self.pos_history_x = self.pos_history_x[-max_history:]
                self.pos_history_y = self.pos_history_y[-max_history:]
                self.pos_history_z = self.pos_history_z[-max_history:]
                self.quality_history = self.quality_history[-max_history:]
            
            if len(self.pos_history_x) >= 3:
                self.world_x = self._weighted_trimmed_mean(self.pos_history_x, self.quality_history)
                self.world_y = self._weighted_trimmed_mean(self.pos_history_y, self.quality_history)
                self.world_z = self._weighted_trimmed_mean(self.pos_history_z, self.quality_history)
            else:
                self.world_x = np.mean(self.pos_history_x)
                self.world_y = np.mean(self.pos_history_y)
                self.world_z = np.mean(self.pos_history_z)
    
    def update_reacquisition(self, x: float, y: float, z: float, time: float, quality: float = 1.0):
        self.last_seen_time = time
        self.reacquisition_count += 1
        
        self.pos_history_x.append(x)
        self.pos_history_y.append(y)
        self.pos_history_z.append(z)
        self.quality_history.append(quality)
        
        if len(self.pos_history_x) >= 2:
            self.world_x = self._weighted_trimmed_mean(self.pos_history_x, self.quality_history)
            self.world_y = self._weighted_trimmed_mean(self.pos_history_y, self.quality_history)
            self.world_z = self._weighted_trimmed_mean(self.pos_history_z, self.quality_history)
        else:
            self.world_x = x
            self.world_y = y
            self.world_z = z
    
    def _weighted_trimmed_mean(self, values: List[float], weights: List[float], 
                                trim_percent: float = 0.1) -> float:
        if len(values) < 3:
            return float(np.mean(values))
        
        arr = np.array(values)
        w = np.array(weights)
        
        order = np.argsort(arr)
        arr = arr[order]
        w = w[order]
        
        n = len(arr)
        trim_count = max(0, int(n * trim_percent))
        if n - 2 * trim_count < 2:
            trim_count = 0
        
        if trim_count > 0:
            arr = arr[trim_count:-trim_count]
            w = w[trim_count:-trim_count]
        
        if np.sum(w) > 0:
            return float(np.average(arr, weights=w))
        return float(np.mean(arr))
    
    def can_retry_beam(self, current_time: float, min_retry_interval: float = 2.0) -> bool:
        if self.beam_succeeded:
            return False
        if self.beam_attempts >= self.max_beam_attempts:
            return False
        if self.needs_immediate_retry:
            return True
        if current_time - self.last_beam_attempt_time < min_retry_interval:
            return False
        return True


class BallDetectorV419(Node):
    def __init__(self):
        super().__init__('ball_detector_v419')

        self.declare_parameter('rgb_topic', '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/image')
        self.declare_parameter('depth_topic', '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/depth_image')
        self.declare_parameter('camera_info_topic', '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/camera_info')
        
        self.declare_parameter('use_ground_truth', True)
        self.declare_parameter('ground_truth_topic', '/model/swincar_ur3/pose')
        self.declare_parameter('ground_truth_frame_id', 'empty')
        
        self.declare_parameter('base_link_offset_x', 0.0)
        self.declare_parameter('base_link_offset_y', 0.2)
        self.declare_parameter('base_link_offset_z', 0.35)
        
        self.declare_parameter('odom_topic', '/swincar/odom')
        
        self.declare_parameter('camera_frame', 'camera_optical_link')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('world_frame', 'world')
        
        self.declare_parameter('target_topic', '/target_pose')
        self.declare_parameter('target_point_topic', '/blue_target_primary')
        self.declare_parameter('all_targets_topic', '/all_tracked_targets')
        self.declare_parameter('beam_done_topic', '/beam_task_done')
        self.declare_parameter('beam_failed_topic', '/beam_task_failed')
        self.declare_parameter('swincar_stopped_topic', '/swincar_stopped')
        
        self.declare_parameter('camera_hfov', 0.6)
        self.declare_parameter('ball_radius', 0.005)
        self.declare_parameter('center_method', 'enclosing_circle')
        self.declare_parameter('min_ball_area', 3.0)
        self.declare_parameter('max_ball_area', 5000.0)
        self.declare_parameter('min_contour_points', 4)
        
        self.declare_parameter('reacquisition_frames', 5)
        self.declare_parameter('reacquisition_timeout', 2.0)
        
        self.declare_parameter('association_radius', 0.08)
        self.declare_parameter('published_position_radius', 0.05)
        self.declare_parameter('min_observations', 3)
        self.declare_parameter('track_timeout', 5.0)
        
        self.declare_parameter('trigger_min_y', 0.6)
        self.declare_parameter('trigger_max_y', 1.2)
        self.declare_parameter('trigger_min_x', -0.26)
        self.declare_parameter('trigger_max_x', 0.25)
        self.declare_parameter('trigger_min_z', -0.5)
        self.declare_parameter('trigger_max_z', 0.5)
        
        self.declare_parameter('depth_sample_radius', 3)
        self.declare_parameter('use_gaussian_depth', True)
        
        self.declare_parameter('swincar_stop_timeout', 10.0)
        self.declare_parameter('debug_detection', True)

        self._load_parameters()
        
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth = None
        self.bridge = CvBridge()
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.robot_qx = 0.0
        self.robot_qy = 0.0
        self.robot_qz = 0.0
        self.robot_qw = 1.0
        self.last_pose_time = None
        self.pose_source = "none"
        
        self.state = "IDLE"
        self.pending_target = None
        self.pending_track_id = None
        self.state_start_time = None
        
        self.swincar_stopped = False
        self.waiting_for_fresh_stop = False
        
        self.published_positions = []
        
        self.tracked_targets: Dict[int, TrackedTarget] = {}
        self.next_target_id = 0
        
        self.frame_count = 0
        self.debug_count = 0
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self._setup_subscriptions()
        self._setup_publishers()
        
        self.create_timer(0.1, self.state_machine_callback)
        
        self._print_config()

    def _load_parameters(self):
        self.camera_frame = self.get_parameter('camera_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        
        self.camera_hfov = self.get_parameter('camera_hfov').value
        
        self.base_link_offset_x = self.get_parameter('base_link_offset_x').value
        self.base_link_offset_y = self.get_parameter('base_link_offset_y').value
        self.base_link_offset_z = self.get_parameter('base_link_offset_z').value
        
        self.ball_radius = self.get_parameter('ball_radius').value
        self.center_method = self.get_parameter('center_method').value
        
        self.min_ball_area = self.get_parameter('min_ball_area').value
        self.max_ball_area = self.get_parameter('max_ball_area').value
        self.min_contour_points = self.get_parameter('min_contour_points').value
        
        self.reacquisition_frames = self.get_parameter('reacquisition_frames').value
        self.reacquisition_timeout = self.get_parameter('reacquisition_timeout').value
        
        self.association_radius = self.get_parameter('association_radius').value
        self.published_position_radius = self.get_parameter('published_position_radius').value
        self.min_observations = self.get_parameter('min_observations').value
        self.track_timeout = self.get_parameter('track_timeout').value
        
        self.trigger_min_y = self.get_parameter('trigger_min_y').value
        self.trigger_max_y = self.get_parameter('trigger_max_y').value
        self.trigger_min_x = self.get_parameter('trigger_min_x').value
        self.trigger_max_x = self.get_parameter('trigger_max_x').value
        self.trigger_min_z = self.get_parameter('trigger_min_z').value
        self.trigger_max_z = self.get_parameter('trigger_max_z').value
        
        self.depth_sample_radius = self.get_parameter('depth_sample_radius').value
        self.use_gaussian_depth = self.get_parameter('use_gaussian_depth').value
        
        self.swincar_stop_timeout = self.get_parameter('swincar_stop_timeout').value
        self.debug_detection = self.get_parameter('debug_detection').value

    def _setup_subscriptions(self):
        self.rgb_sub = self.create_subscription(
            Image, self.get_parameter('rgb_topic').value, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, self.get_parameter('depth_topic').value, self.depth_callback, 10)
        self.cam_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self.cam_info_callback, 10)
        
        if self.get_parameter('use_ground_truth').value:
            self.ground_truth_frame_id = self.get_parameter('ground_truth_frame_id').value
            self.gt_sub = self.create_subscription(
                PoseStamped, self.get_parameter('ground_truth_topic').value,
                self.ground_truth_callback, 10)
        else:
            self.odom_sub = self.create_subscription(
                Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)
        
        self.beam_done_sub = self.create_subscription(
            Bool, self.get_parameter('beam_done_topic').value, self.beam_done_callback, 10)
        self.beam_failed_sub = self.create_subscription(
            Bool, self.get_parameter('beam_failed_topic').value, self.beam_failed_callback, 10)
        self.swincar_stopped_sub = self.create_subscription(
            Bool, self.get_parameter('swincar_stopped_topic').value, self.swincar_stopped_callback, 10)

    def _setup_publishers(self):
        self.target_pub = self.create_publisher(
            PoseStamped, self.get_parameter('target_topic').value, 10)
        self.target_point_pub = self.create_publisher(
            PointStamped, self.get_parameter('target_point_topic').value, 10)
        self.all_targets_pub = self.create_publisher(
            PoseArray, self.get_parameter('all_targets_topic').value, 10)
        self.debug_pub = self.create_publisher(Image, '/ball_detector/debug_image', 10)

    def _print_config(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("Ball Detector v4.19 - Retry-hang fix for stop-start flow")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Ball radius: {self.ball_radius*1000:.1f}mm")
        self.get_logger().info(f"Camera HFOV: {self.camera_hfov} rad ({math.degrees(self.camera_hfov):.1f}°)")
        self.get_logger().info(f"Min ball area: {self.min_ball_area}")
        self.get_logger().info(f"Min contour points: {self.min_contour_points}")
        self.get_logger().info("FIX: Short-circuit WAITING_FOR_STOP on retries (car already stopped)")
        self.get_logger().info("=" * 60)

    # ==================== Circle Fitting ====================
    
    def _get_center_enclosing_circle(self, contour: np.ndarray) -> Tuple[float, float, float, float]:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        contour_area = cv2.contourArea(contour)
        circle_area = math.pi * radius * radius
        quality = min(1.0, contour_area / circle_area) if circle_area > 0 else 0.5
        return cx, cy, radius, quality
    
    def _get_center_ellipse(self, contour: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        if len(contour) < 5:
            return None
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (width, height), angle = ellipse
            avg_radius = (width + height) / 4.0
            aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0.5
            return cx, cy, avg_radius, aspect_ratio
        except:
            return None
    
    def get_ball_center(self, contour: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
        if self.center_method == 'ellipse':
            result = self._get_center_ellipse(contour)
            if result is not None:
                return result
        return self._get_center_enclosing_circle(contour)

    # ==================== Ray-Based 3D Projection ====================
    
    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.0
        ray = np.array([x, y, z])
        return ray / np.linalg.norm(ray)
    
    def surface_to_center(self, u: float, v: float, depth: float) -> Tuple[float, float, float]:
        ray = self.pixel_to_ray(u, v)
        t_surface = depth / ray[2]
        surface_point = ray * t_surface
        center_point = surface_point + ray * self.ball_radius
        return center_point[0], center_point[1], center_point[2]

    # ==================== Depth Sampling ====================
    
    def get_gaussian_weighted_depth(self, cx: float, cy: float) -> Optional[Tuple[float, float]]:
        if self.latest_depth is None:
            return None
        
        h, w = self.latest_depth.shape
        r = self.depth_sample_radius
        
        y_min = max(0, int(cy - r))
        y_max = min(h, int(cy + r + 1))
        x_min = max(0, int(cx - r))
        x_max = min(w, int(cx + r + 1))
        
        if y_max <= y_min or x_max <= x_min:
            return None
        
        roi = self.latest_depth[y_min:y_max, x_min:x_max].copy()
        roi_h, roi_w = roi.shape
        y_coords, x_coords = np.ogrid[:roi_h, :roi_w]
        
        center_x = cx - x_min
        center_y = cy - y_min
        
        sigma = r / 2.0
        weights = np.exp(-((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * sigma**2))
        
        valid_mask = np.isfinite(roi) & (roi > 0.05) & (roi < 5.0)
        
        if np.sum(valid_mask) < 3:
            ix, iy = int(cx), int(cy)
            if 0 <= ix < w and 0 <= iy < h:
                d = self.latest_depth[iy, ix]
                if np.isfinite(d) and d > 0:
                    return float(d), 0.5
            return None
        
        valid_depths = roi[valid_mask]
        valid_weights = weights[valid_mask]
        
        weighted_depth = np.average(valid_depths, weights=valid_weights)
        depth_std = np.std(valid_depths)
        confidence = max(0.3, 1.0 - min(1.0, depth_std / 0.01))
        
        return float(weighted_depth), confidence

    def get_median_depth(self, cx: float, cy: float) -> Optional[float]:
        if self.latest_depth is None:
            return None
        
        h, w = self.latest_depth.shape
        r = self.depth_sample_radius
        
        roi = self.latest_depth[
            max(0, int(cy-r)):min(h, int(cy+r+1)),
            max(0, int(cx-r)):min(w, int(cx+r+1))
        ]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        
        if len(valid) >= 3:
            return float(np.median(valid))
        
        ix, iy = int(cx), int(cy)
        if 0 <= ix < w and 0 <= iy < h:
            d = self.latest_depth[iy, ix]
            if np.isfinite(d) and d > 0:
                return float(d)
        return None

    # ==================== State Machine ====================
    
    def swincar_stopped_callback(self, msg: Bool):
        old_value = self.swincar_stopped
        self.swincar_stopped = msg.data
        
        if msg.data and not old_value:
            # Fresh False→True edge: line-follower has just brought car to rest.
            if self.waiting_for_fresh_stop:
                self.waiting_for_fresh_stop = False
                
                if self.state == "WAITING_FOR_STOP" and self.pending_target is not None:
                    if self.pending_target.skip_reacquisition:
                        self.get_logger().info("📍 STOPPED - Skipping re-acquisition (retry)")
                        self.pending_target.skip_reacquisition = False
                        self._send_target_to_ur3()
                        self.state = "WAITING_FOR_BEAM"
                        self.state_start_time = self.get_clock().now()
                    else:
                        self.get_logger().info("📍 STOPPED - Starting re-acquisition")
                        self.state = "REACQUIRING"
                        self.state_start_time = self.get_clock().now()
                        self.pending_target.clear_history_for_reacquisition()
                        self.get_logger().info(f"🔄 Re-acquiring target #{self.pending_track_id}")
                    
        elif not msg.data and old_value:
            self.get_logger().info("🚗 MOVING")
    
    def state_machine_callback(self):
        now = self.get_clock().now()
        
        if self.state == "WAITING_FOR_STOP":
            if self.state_start_time is not None:
                elapsed = (now - self.state_start_time).nanoseconds / 1e9
                if elapsed > self.swincar_stop_timeout:
                    self.get_logger().warn("⚠️ Stop timeout - sending anyway")
                    self.waiting_for_fresh_stop = False
                    self._send_target_to_ur3()
                    self.state = "WAITING_FOR_BEAM"
                    self.state_start_time = now
        
        elif self.state == "REACQUIRING":
            if self.pending_target is not None:
                if self.pending_target.reacquisition_count >= self.reacquisition_frames:
                    old_pos = f"[{self.pending_target.world_x:.4f}, {self.pending_target.world_y:.4f}, {self.pending_target.world_z:.4f}]"
                    self.get_logger().info(f"✅ Re-acquisition complete ({self.pending_target.reacquisition_count} frames)")
                    self.get_logger().info(f"📍 Final position: {old_pos}")
                    self._send_target_to_ur3()
                    self.state = "WAITING_FOR_BEAM"
                    self.state_start_time = now
                else:
                    if self.state_start_time is not None:
                        elapsed = (now - self.state_start_time).nanoseconds / 1e9
                        if elapsed > self.reacquisition_timeout:
                            self.get_logger().warn(f"⚠️ Re-acquisition timeout ({self.pending_target.reacquisition_count} frames)")
                            if self.pending_target.reacquisition_count > 0:
                                self._send_target_to_ur3()
                                self.state = "WAITING_FOR_BEAM"
                                self.state_start_time = now
                            else:
                                self.get_logger().warn(f"❌ Re-acquisition failed for #{self.pending_track_id} - target lost")
                                if self.pending_track_id in self.tracked_targets:
                                    track = self.tracked_targets[self.pending_track_id]
                                    if track.beam_attempts >= track.max_beam_attempts:
                                        track.beam_succeeded = True
                                        self.get_logger().error(f"❌ GAVE UP #{self.pending_track_id}")
                                    else:
                                        track.published = False
                                        track.needs_immediate_retry = True
                                        track.skip_reacquisition = True  # Skip re-acq on retry
                                        self.get_logger().info(f"🔁 Will retry #{self.pending_track_id} immediately (skip re-acq)")
                                self._reset_to_idle()

    def _send_target_to_ur3(self):
        if self.pending_target is None:
            return
        
        track = self.pending_target
        stamp = self.get_clock().now().to_msg()
        
        robot_coords = self._get_robot_frame_coords(track)
        if robot_coords is None:
            return
        
        rx, ry, rz = robot_coords
        
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.robot_frame
        pose.pose.position.x = rx
        pose.pose.position.y = ry
        pose.pose.position.z = rz
        pose.pose.orientation.w = 1.0
        self.target_pub.publish(pose)
        
        self.get_logger().info(f"📤 SENT #{track.id}: robot=[{rx:.3f}, {ry:.3f}, {rz:.3f}]")

    def beam_done_callback(self, msg: Bool):
        if msg.data and self.state == "WAITING_FOR_BEAM":
            track_id = self.pending_track_id
            if track_id is not None and track_id in self.tracked_targets:
                track = self.tracked_targets[track_id]
                track.beam_succeeded = True
                track.needs_immediate_retry = False
                track.skip_reacquisition = False
                self._remember_published_position(track.world_x, track.world_y, track.world_z)
                self.get_logger().info(f"✅ SUCCESS #{track_id}")
            self._reset_to_idle()
    
    def beam_failed_callback(self, msg: Bool):
        """
        v4.19 FIX: when we will retry, KEEP needs_immediate_retry=True so that
        maybe_publish_targets takes the short-circuit path (car already
        stopped → skip WAITING_FOR_STOP → straight to REACQUIRING).
        Previously this callback cleared the flag, forcing every
        beam-execution retry through the 10s swincar_stop_timeout.
        """
        if msg.data and self.state == "WAITING_FOR_BEAM":
            track_id = self.pending_track_id
            if track_id is not None and track_id in self.tracked_targets:
                track = self.tracked_targets[track_id]
                # Execution failed → we want a fresh re-acq (in case the arm
                # bumped something), never send stale cached pose directly.
                track.skip_reacquisition = False
                
                if track.beam_attempts >= track.max_beam_attempts:
                    track.beam_succeeded = True
                    track.needs_immediate_retry = False
                    self._remember_published_position(track.world_x, track.world_y, track.world_z)
                    self.get_logger().error(f"❌ GAVE UP #{track_id} after {track.beam_attempts} attempts")
                else:
                    # Car is still stopped; flag so next maybe_publish_targets
                    # tick short-circuits WAITING_FOR_STOP.
                    track.needs_immediate_retry = True
                    self.get_logger().warn(
                        f"⚠️ FAILED #{track_id} (attempt {track.beam_attempts}/{track.max_beam_attempts}) - will retry"
                    )
            self._reset_to_idle()
    
    def _reset_to_idle(self):
        self.state = "IDLE"
        self.pending_target = None
        self.pending_track_id = None
        self.state_start_time = None
        self.waiting_for_fresh_stop = False

    # ==================== Robot Pose ====================
    
    def ground_truth_callback(self, msg: PoseStamped):
        if msg.header.frame_id != self.ground_truth_frame_id:
            return
        
        prev_x = self.robot_x
        
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.robot_z = msg.pose.position.z
        
        self.robot_qx = msg.pose.orientation.x
        self.robot_qy = msg.pose.orientation.y
        self.robot_qz = msg.pose.orientation.z
        self.robot_qw = msg.pose.orientation.w
        
        if abs(self.robot_x - prev_x) > 1.0 or self.pose_source == "none":
            rot = Rotation.from_quat([self.robot_qx, self.robot_qy, self.robot_qz, self.robot_qw])
            roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
            self.get_logger().info(f"GT: X={self.robot_x:.2f} roll={roll:.1f}° pitch={pitch:.1f}°")
        
        self.last_pose_time = self.get_clock().now()
        self.pose_source = "ground_truth"
    
    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_z = msg.pose.pose.position.z
        
        self.robot_qx = msg.pose.pose.orientation.x
        self.robot_qy = msg.pose.pose.orientation.y
        self.robot_qz = msg.pose.pose.orientation.z
        self.robot_qw = msg.pose.pose.orientation.w
        
        self.last_pose_time = self.get_clock().now()
        self.pose_source = "odom"

    def _robot_to_world(self, rx: float, ry: float, rz: float) -> tuple:
        offset = np.array([self.base_link_offset_x, self.base_link_offset_y, self.base_link_offset_z])
        point_robot = np.array([rx, ry, rz])
        rot = Rotation.from_quat([self.robot_qx, self.robot_qy, self.robot_qz, self.robot_qw])
        point_body = offset + point_robot
        point_world = np.array([self.robot_x, self.robot_y, self.robot_z]) + rot.apply(point_body)
        return tuple(point_world)
    
    def _world_to_robot(self, wx: float, wy: float, wz: float) -> tuple:
        rot = Rotation.from_quat([self.robot_qx, self.robot_qy, self.robot_qz, self.robot_qw])
        rot_inv = rot.inv()
        world_point = np.array([wx, wy, wz])
        robot_pos = np.array([self.robot_x, self.robot_y, self.robot_z])
        offset = np.array([self.base_link_offset_x, self.base_link_offset_y, self.base_link_offset_z])
        point_body = rot_inv.apply(world_point - robot_pos)
        point_robot = point_body - offset
        return tuple(point_robot)

    # ==================== Camera ====================
    
    def cam_info_callback(self, msg: CameraInfo):
        if self.fx is not None:
            return
        
        w, h = float(msg.width), float(msg.height)
        self.fx = (w / 2.0) / math.tan(self.camera_hfov / 2.0)
        self.fy = self.fx
        self.cx = w / 2.0
        self.cy = h / 2.0
        
        self.get_logger().info(f"Camera: {int(w)}x{int(h)}, fx={self.fx:.0f}")

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except:
            pass

    # ==================== Main Detection ====================
    
    def rgb_callback(self, msg: Image):
        if self.latest_depth is None or self.fx is None:
            return
            
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except:
            return
        
        stamp = msg.header.stamp
        current_time = stamp.sec + stamp.nanosec * 1e-9
        
        self.frame_count += 1
        
        detections, debug_img = self.find_balls(rgb)
        
        if self.debug_detection and self.frame_count % 10 == 0:
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
            except:
                pass
        
        if not detections:
            return
        
        detections_world = []
        for det in detections:
            world_pos = self.detection_to_world(det, stamp)
            if world_pos is not None:
                detections_world.append(world_pos)
        
        if detections_world:
            if self.state == "REACQUIRING":
                self._update_reacquisition(detections_world, current_time)
            else:
                self.update_tracking(detections_world, current_time)
                self.maybe_publish_targets(stamp, current_time)
            
            self.publish_all_tracked(stamp)
        
        if self.frame_count % 30 == 0:
            self.maintenance(current_time)

    def _update_reacquisition(self, detections_world: List[Tuple], current_time: float):
        if self.pending_target is None or self.pending_track_id is None:
            return
        
        best_det = None
        best_dist = float('inf')
        
        for det in detections_world:
            wx, wy, wz, quality = det
            dist = self.pending_target.distance_3d(wx, wy, wz)
            if dist < best_dist and dist < self.association_radius:
                best_dist = dist
                best_det = det
        
        if best_det is not None:
            wx, wy, wz, quality = best_det
            self.pending_target.update_reacquisition(wx, wy, wz, current_time, quality)
            
            if self.debug_count % 10 == 0:
                self.get_logger().info(
                    f"🔄 Re-acq #{self.pending_track_id}: frame {self.pending_target.reacquisition_count}/{self.reacquisition_frames}"
                )

    def find_balls(self, bgr_image) -> Tuple[List[dict], np.ndarray]:
        debug_img = bgr_image.copy()
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h_img, w_img = bgr_image.shape[:2]
        
        lower_blue = np.array([90, 80, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.debug_count += 1
        if self.debug_count % 30 == 0:
            self.get_logger().info(f"[DEBUG] Blue pixels: {np.sum(blue_mask > 0)}, Contours: {len(contours)}, State: {self.state}")
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if not (self.min_ball_area <= area <= self.max_ball_area):
                continue
            
            if len(contour) < self.min_contour_points:
                continue
            
            contour_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            
            cx, cy, radius_px, quality = self.get_ball_center(contour, contour_mask)
            
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 1)
            cv2.circle(debug_img, (int(cx), int(cy)), int(radius_px), (255, 0, 255), 2)
            cv2.circle(debug_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            
            state_color = (0, 255, 255) if self.state == "REACQUIRING" else (0, 255, 0)
            cv2.putText(debug_img, self.state[:4], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            
            detections.append({
                'cx': cx,
                'cy': cy,
                'radius_px': radius_px,
                'area': area,
                'quality': quality
            })
        
        return detections, debug_img

    def detection_to_world(self, det: dict, stamp) -> Optional[Tuple[float, float, float, float]]:
        cx, cy = det['cx'], det['cy']
        quality = det['quality']
        
        if self.use_gaussian_depth:
            depth_result = self.get_gaussian_weighted_depth(cx, cy)
            if depth_result is None:
                return None
            depth, depth_confidence = depth_result
            quality = quality * depth_confidence
        else:
            depth = self.get_median_depth(cx, cy)
            if depth is None or depth <= 0:
                return None
        
        Xc, Yc, Zc = self.surface_to_center(cx, cy, depth)
        
        pt_cam = PointStamped()
        pt_cam.header.stamp = stamp
        pt_cam.header.frame_id = self.camera_frame
        pt_cam.point.x, pt_cam.point.y, pt_cam.point.z = float(Xc), float(Yc), float(Zc)
        
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame, self.camera_frame, stamp
            )
            pt_robot = tf2_geometry_msgs.do_transform_point(pt_cam, transform)
            
            rx, ry, rz = pt_robot.point.x, pt_robot.point.y, pt_robot.point.z
            wx, wy, wz = self._robot_to_world(rx, ry, rz)
            
            return (wx, wy, wz, quality)
        except:
            return None

    # ==================== Tracking ====================
    
    def update_tracking(self, detections: List[Tuple], current_time: float):
        for det_tuple in detections:
            wx, wy, wz, quality = det_tuple
            
            closest_track = None
            closest_dist = float('inf')
            
            for tid, track in self.tracked_targets.items():
                dist = track.distance_3d(wx, wy, wz)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_track = track
            
            if closest_track is not None and closest_dist < self.association_radius:
                closest_track.update(wx, wy, wz, current_time, quality)
                continue
            
            too_close = any(t.distance_3d(wx, wy, wz) < self.association_radius 
                          for t in self.tracked_targets.values())
            if too_close:
                continue
            
            new_id = self.next_target_id
            self.next_target_id += 1
            
            self.tracked_targets[new_id] = TrackedTarget(
                id=new_id, world_x=wx, world_y=wy, world_z=wz,
                first_seen_time=current_time, last_seen_time=current_time
            )
            
            self.get_logger().info(f"NEW #{new_id}: [{wx:.3f}, {wy:.3f}, {wz:.3f}]")

    def maintenance(self, current_time: float):
        if not self.tracked_targets or self.state not in ["IDLE"]:
            return
        
        to_remove = [tid for tid, t in self.tracked_targets.items()
                    if (t.beam_succeeded and current_time - t.last_seen_time > 2.0)
                    or current_time - t.last_seen_time > self.track_timeout]
        
        for tid in to_remove:
            self.tracked_targets.pop(tid)

    # ==================== Publishing ====================
    
    def maybe_publish_targets(self, stamp, current_time: float):
        """
        v4.19 FIX: on retries where the car is already stopped, skip
        WAITING_FOR_STOP entirely and jump straight to REACQUIRING (normal
        retry) or WAITING_FOR_BEAM (skip_reacquisition retry). Eliminates
        the ~10s swincar_stop_timeout wait that previously fired on every
        retry because /swincar_stopped never had a False→True edge.
        """
        if self.state != "IDLE":
            return
        
        best_track = None
        best_track_id = None
        best_robot_y = float('inf')
        
        for tid, track in self.tracked_targets.items():
            if track.times_seen < self.min_observations or track.beam_succeeded:
                continue
            
            if self._is_position_published(track.world_x, track.world_y, track.world_z):
                track.beam_succeeded = True
                continue
            
            if not track.published or track.can_retry_beam(current_time):
                robot_coords = self._get_robot_frame_coords(track)
                if robot_coords is None:
                    continue
                
                rx, ry, rz = robot_coords
                
                in_trigger = (self.trigger_min_x <= rx <= self.trigger_max_x and
                             self.trigger_min_y <= ry <= self.trigger_max_y and
                             self.trigger_min_z <= rz <= self.trigger_max_z)
                
                if track.needs_immediate_retry and not in_trigger:
                    self.get_logger().warn(f"⚠️ #{tid} needs retry but OUT of zone")
                    track.beam_succeeded = True
                    track.needs_immediate_retry = False
                    track.skip_reacquisition = False
                    self.get_logger().error(f"❌ SKIPPING #{tid} - out of trigger zone")
                    continue
                
                if in_trigger:
                    if ry < best_robot_y:
                        best_robot_y = ry
                        best_track = track
                        best_track_id = tid
        
        if best_track is None:
            return
        
        # Commit this as the current pending target
        best_track.beam_attempts += 1
        best_track.last_beam_attempt_time = current_time
        best_track.published = True
        
        # Snapshot the retry flag; skip_reacquisition stays on the track
        # so the fresh-stop callback can still consume it on the normal path.
        is_retry = best_track.needs_immediate_retry
        best_track.needs_immediate_retry = False
        
        self.pending_target = best_track
        self.pending_track_id = best_track_id
        
        retry_note = " (RETRY)" if is_retry else ""
        self.get_logger().info(
            f"🎯 #{best_track_id}: [{best_track.world_x:.3f}, {best_track.world_y:.3f}, {best_track.world_z:.3f}] (seen {best_track.times_seen}x){retry_note}"
        )
        
        # ── v4.19 short-circuit ────────────────────────────────────────────
        # If this is a retry and the car is already stopped, do NOT re-enter
        # WAITING_FOR_STOP — there won't be a False→True edge to unblock us.
        if is_retry and self.swincar_stopped:
            if best_track.skip_reacquisition:
                best_track.skip_reacquisition = False
                self.get_logger().info("⚡ RETRY (car stopped) - sending directly to UR3")
                self._send_target_to_ur3()
                self.state = "WAITING_FOR_BEAM"
                self.state_start_time = self.get_clock().now()
            else:
                self.get_logger().info("⚡ RETRY (car stopped) - re-acquiring now")
                best_track.clear_history_for_reacquisition()
                self.state = "REACQUIRING"
                self.state_start_time = self.get_clock().now()
            return
        
        # ── Normal flow: publish stop signal, wait for /swincar_stopped edge ──
        self._publish_stop_signal(best_track, stamp)
        self.state = "WAITING_FOR_STOP"
        self.state_start_time = self.get_clock().now()
        self.waiting_for_fresh_stop = True

    def _publish_stop_signal(self, track: TrackedTarget, stamp):
        robot_coords = self._get_robot_frame_coords(track)
        if robot_coords is None:
            return
        
        rx, ry, rz = robot_coords
        
        point = PointStamped()
        point.header.stamp = stamp
        point.header.frame_id = self.robot_frame
        point.point.x, point.point.y, point.point.z = rx, ry, rz
        self.target_point_pub.publish(point)
        
        self.get_logger().info(f"📡 STOP [{rx:.3f}, {ry:.3f}, {rz:.3f}]")

    def _get_robot_frame_coords(self, track: TrackedTarget) -> Optional[tuple]:
        return self._world_to_robot(track.world_x, track.world_y, track.world_z)
    
    def _is_position_published(self, x: float, y: float, z: float) -> bool:
        return any(math.sqrt((x-px)**2 + (y-py)**2 + (z-pz)**2) < self.published_position_radius
                  for px, py, pz in self.published_positions)
    
    def _remember_published_position(self, x: float, y: float, z: float):
        self.published_positions.append((x, y, z))
        if len(self.published_positions) > 200:
            self.published_positions.pop(0)

    def publish_all_tracked(self, stamp):
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.world_frame
        
        for track in self.tracked_targets.values():
            pose = Pose()
            pose.position.x = track.world_x
            pose.position.y = track.world_y
            pose.position.z = track.world_z
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        
        self.all_targets_pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    node = BallDetectorV419()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()