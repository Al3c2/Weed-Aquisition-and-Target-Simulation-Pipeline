#!/usr/bin/env python3
"""
Predictive multi-target detector for a moving-base UR3 + line-follower platform.

Pipeline:
  1. Detect blue balls in the RGB image (HSV threshold + contour analysis).
  2. Convert each detection to a world-frame position using the depth image,
     camera intrinsics, the camera->base_link TF, and the robot's ground-truth
     pose (with latency compensation via a pose-history buffer).
  3. Track detections across frames with nearest-neighbour association and an
     EMA position filter, locking a target's position once it is well observed.
  4. Maintain a priority queue of reachable targets and dispatch them to the
     arm one at a time, modulating the platform's driving speed so the arm has
     time to plan and execute before the target leaves its workspace.

Topics in:  RGB, depth, camera_info, ground-truth pose, beam done/failed,
            arm trajectory duration.
Topics out: arm target pose (base_link), target world point, target robot point,
            slowdown speed, full tracked-target array.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose, PointStamped, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32

from cv_bridge import CvBridge
import cv2
import numpy as np

import tf2_ros
import tf2_geometry_msgs

from scipy.spatial.transform import Rotation


@dataclass
class TrackedTarget:
    """A single tracked ball.

    Position is filtered with an EMA until `position_locked` is set, after
    which it is held fixed so a dispatched target does not drift mid-aim.
    """
    id: int
    world_x: float
    world_y: float
    world_z: float
    first_seen_time: float
    last_seen_time: float
    times_seen: int = 1
    published: bool = False
    position_locked: bool = False
    beam_attempts: int = 0
    max_beam_attempts: int = 2
    beam_succeeded: bool = False

    def distance_to(self, x: float, y: float, z: float) -> float:
        """Euclidean distance from this target to a world-frame point."""
        return math.sqrt(
            (self.world_x-x)**2 + (self.world_y-y)**2 + (self.world_z-z)**2)

    def update(self, x: float, y: float, z: float, time: float, alpha: float = 0.3):
        """Blend a new observation into the stored position via EMA.

        Locks the position after enough consistent observations so the target
        stops moving once we're confident in it.
        """
        self.last_seen_time = time
        self.times_seen += 1
        if not self.position_locked:
            self.world_x = alpha*x + (1-alpha)*self.world_x
            self.world_y = alpha*y + (1-alpha)*self.world_y
            self.world_z = alpha*z + (1-alpha)*self.world_z
            if self.times_seen >= 5:
                self.position_locked = True


class PredictiveMultiTargetDetector(Node):
    def __init__(self):
        super().__init__('predictive_multi_target_detector')

        # ── Topics and frames ────────────────────────────────────────────────
        self.declare_parameter('rgb_topic', '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/image')
        self.declare_parameter('depth_topic', '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/depth_image')
        self.declare_parameter('camera_info_topic', '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/camera_info')
        self.declare_parameter('use_ground_truth', True)
        self.declare_parameter('ground_truth_topic', '/model/swincar_ur3/pose')
        self.declare_parameter('ground_truth_frame_id', 'empty')
        self.declare_parameter('base_link_offset_x', 0.0)
        self.declare_parameter('base_link_offset_y', 0.2)
        self.declare_parameter('base_link_offset_z', 0.35)
        self.declare_parameter('camera_frame', 'camera_optical_link')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('target_topic', '/target_pose')
        self.declare_parameter('target_point_topic', '/blue_target_primary')
        self.declare_parameter('target_speed_topic', '/target_slowdown_speed')
        self.declare_parameter('all_targets_topic', '/all_tracked_targets')
        self.declare_parameter('beam_done_topic', '/beam_task_done')
        self.declare_parameter('beam_failed_topic', '/beam_task_failed')

        # ── Vision / detection tuning ────────────────────────────────────────
        self.declare_parameter('camera_hfov', 0.6)
        self.declare_parameter('ball_radius', 0.005)
        self.declare_parameter('min_ball_area', 3.0)
        self.declare_parameter('max_ball_area', 5000.0)
        self.declare_parameter('min_contour_points', 4)
        self.declare_parameter('association_radius', 0.06)
        self.declare_parameter('max_detection_range', 3.5)
        self.declare_parameter('min_observations', 1)

        # Minimum tracked targets before dispatch is allowed. Kept at 1 because
        # near the end of a run only one or two targets remain visible; a higher
        # value would permanently block dispatch for the final targets.
        self.declare_parameter('min_targets_before_publish', 1)
        self.declare_parameter('track_timeout', 120.0)

        # ── Arm workspace limits (base_link frame) ───────────────────────────
        self.declare_parameter('arm_reach_x_min', -0.3)
        self.declare_parameter('arm_reach_x_max',  0.3)
        self.declare_parameter('arm_reach_y_min',  0.6)
        self.declare_parameter('arm_reach_y_max',  1.4)
        self.declare_parameter('arm_reach_z_min', -0.35)
        self.declare_parameter('arm_reach_z_max',  0.2)
        self.declare_parameter('vision_min_distance', 0.5)
        self.declare_parameter('vision_max_distance', 3.5)
        self.declare_parameter('depth_sample_radius', 3)
        self.declare_parameter('use_gaussian_depth', True)

        # ── Motion / timing model ────────────────────────────────────────────
        self.declare_parameter('arm_movement_time', 2.0)
        self.declare_parameter('min_speed', 0.03)
        self.declare_parameter('max_speed', 0.35)

        # Default arm aim point in base_link X. A larger magnitude gives a wider
        # sweep window: at -0.29 the window is ~280mm, which at min_speed
        # (~0.03 m/s) corresponds to ~9s of sweep time to catch the hit.
        self.declare_parameter('default_x_exec', -0.29)
        self.declare_parameter('planning_time_est', 3)

        # Expected duration of the arm hold/sweep phase. Observed hold phases
        # run 7-9s; 20s leaves a comfortable margin so the TARGETING timeout
        # never fires before beam_done/beam_failed arrives.
        self.declare_parameter('arm_hold_time_est', 20.0)

        self._load_parameters()

        # ── Runtime state ────────────────────────────────────────────────────
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_depth = None
        self.bridge = CvBridge()
        self.robot_x = self.robot_y = self.robot_z = 0.0
        self.robot_vx = self.robot_vy = 0.0
        self.robot_qx = self.robot_qy = self.robot_qz = 0.0
        self.robot_qw = 1.0
        self.last_pose_time = None
        self.pose_history: deque = deque(maxlen=100)
        self.camera_to_robot_transform = None
        self.state = "DRIVING"
        self.last_queue_log_time = 0.0
        self.last_completed_x = None
        self.recovery_until = 0.0
        self.recovery_time = 3.0
        self.current_target_id = None
        self.current_x_exec = -0.2
        self.targeting_start_time = None
        self.direct_aim_mode = False  # when set, stop the robot and aim at the current ball position
        self.target_queue: List[int] = []
        self.published_positions = []
        self.published_position_radius = 0.04
        self.tracked_targets: Dict[int, TrackedTarget] = {}
        self.next_target_id = 0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._setup_subs_pubs()
        self.create_timer(0.5, self.maintenance_callback)
        self.get_logger().info(
            f"Detector ready | y=[{self.arm_reach_y_min:.2f},{self.arm_reach_y_max:.2f}]m "
            f"default_x_exec={self.default_x_exec:.2f}m | Gate2=-0.8m | "
            f"arm_time={self.arm_movement_time:.1f}s max_spd={self.max_speed:.2f}m/s")

    def _load_parameters(self):
        """Read all declared parameters into instance attributes."""
        self.camera_frame = self.get_parameter('camera_frame').value
        self.robot_frame  = self.get_parameter('robot_frame').value
        self.world_frame  = self.get_parameter('world_frame').value
        self.camera_hfov  = self.get_parameter('camera_hfov').value
        self.ball_radius  = self.get_parameter('ball_radius').value
        self.base_link_offset_x = self.get_parameter('base_link_offset_x').value
        self.base_link_offset_y = self.get_parameter('base_link_offset_y').value
        self.base_link_offset_z = self.get_parameter('base_link_offset_z').value
        self.association_radius         = self.get_parameter('association_radius').value
        self.max_detection_range        = self.get_parameter('max_detection_range').value
        self.min_observations           = self.get_parameter('min_observations').value
        self.min_targets_before_publish = self.get_parameter('min_targets_before_publish').value
        self.track_timeout              = self.get_parameter('track_timeout').value
        self.arm_reach_x_min = self.get_parameter('arm_reach_x_min').value
        self.arm_reach_x_max = self.get_parameter('arm_reach_x_max').value
        self.arm_reach_y_min = self.get_parameter('arm_reach_y_min').value
        self.arm_reach_y_max = self.get_parameter('arm_reach_y_max').value
        self.arm_reach_z_min = self.get_parameter('arm_reach_z_min').value
        self.arm_reach_z_max = self.get_parameter('arm_reach_z_max').value
        self.vision_min_distance = self.get_parameter('vision_min_distance').value
        self.vision_max_distance = self.get_parameter('vision_max_distance').value
        self.depth_sample_radius = self.get_parameter('depth_sample_radius').value
        self.use_gaussian_depth  = self.get_parameter('use_gaussian_depth').value
        self.min_ball_area       = self.get_parameter('min_ball_area').value
        self.max_ball_area       = self.get_parameter('max_ball_area').value
        self.min_contour_points  = self.get_parameter('min_contour_points').value
        self.arm_movement_time   = self.get_parameter('arm_movement_time').value
        self.min_speed           = self.get_parameter('min_speed').value
        self.max_speed           = self.get_parameter('max_speed').value
        self.default_x_exec      = self.get_parameter('default_x_exec').value
        self.planning_time_est   = self.get_parameter('planning_time_est').value
        self.arm_hold_time_est   = self.get_parameter('arm_hold_time_est').value

    def _setup_subs_pubs(self):
        """Create all subscriptions and publishers."""
        self.rgb_sub      = self.create_subscription(Image,      self.get_parameter('rgb_topic').value,         self.rgb_callback,      10)
        self.depth_sub    = self.create_subscription(Image,      self.get_parameter('depth_topic').value,       self.depth_callback,    10)
        self.cam_info_sub = self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value, self.cam_info_callback, 10)

        if self.get_parameter('use_ground_truth').value:
            self.ground_truth_frame_id = self.get_parameter('ground_truth_frame_id').value
            self.gt_sub = self.create_subscription(
                PoseStamped, self.get_parameter('ground_truth_topic').value,
                self.ground_truth_callback, 10)

        self.target_pub        = self.create_publisher(PoseStamped,  self.get_parameter('target_topic').value,       10)
        self.target_point_pub  = self.create_publisher(PointStamped, self.get_parameter('target_point_topic').value, 10)
        self.target_speed_pub  = self.create_publisher(Float32,      self.get_parameter('target_speed_topic').value, 10)
        self.all_targets_pub   = self.create_publisher(PoseArray,    self.get_parameter('all_targets_topic').value,  10)
        # Ball world position consumed by the arm hold-phase sweep.
        self.target_world_pub  = self.create_publisher(PointStamped, '/target_world_pos', 10)

        self.beam_done_sub  = self.create_subscription(
            Bool, self.get_parameter('beam_done_topic').value,   self.beam_done_callback,   10)
        self.beam_failed_sub = self.create_subscription(
            Bool, self.get_parameter('beam_failed_topic').value, self.beam_failed_callback, 10)
        self.traj_duration_sub = self.create_subscription(
            Float32, '/arm_trajectory_duration', self.traj_duration_callback, 10)

    # ── Trajectory duration feedback ──────────────────────────────────────────
    def traj_duration_callback(self, msg: Float32):
        """Smooth the measured arm trajectory duration into our timing estimate."""
        v = float(msg.data)
        if not (0.1 < v < 15.0):
            return
        self.arm_movement_time = 0.4*v + 0.6*self.arm_movement_time

    # ── Geometry helpers ──────────────────────────────────────────────────────
    def _world_to_robot_at(self, wx, wy, wz, rx_, ry_, rz_) -> tuple:
        """Transform a world point into base_link, given an arbitrary robot pose."""
        rot    = Rotation.from_quat([self.robot_qx, self.robot_qy, self.robot_qz, self.robot_qw])
        offset = np.array([self.base_link_offset_x, self.base_link_offset_y, self.base_link_offset_z])
        return tuple(rot.inv().apply(np.array([wx, wy, wz]) - np.array([rx_, ry_, rz_])) - offset)

    def _world_to_robot(self, wx, wy, wz) -> tuple:
        """Transform a world point into base_link using the current robot pose."""
        return self._world_to_robot_at(wx, wy, wz, self.robot_x, self.robot_y, self.robot_z)

    def _robot_to_world(self, rx, ry, rz) -> tuple:
        """Transform a base_link point into the world frame using the current pose."""
        rot    = Rotation.from_quat([self.robot_qx, self.robot_qy, self.robot_qz, self.robot_qw])
        offset = np.array([self.base_link_offset_x, self.base_link_offset_y, self.base_link_offset_z])
        return tuple(np.array([self.robot_x, self.robot_y, self.robot_z])
                     + rot.apply(offset + np.array([rx, ry, rz])))

    # ── Pose history (camera/GT latency compensation) ─────────────────────────
    def _get_pose_at(self, t_s: float) -> Optional[dict]:
        """Return the robot pose at time `t_s`, linearly interpolating between
        stored samples. Clamps to the earliest/latest sample at the edges."""
        if not self.pose_history:
            return None
        before = after = None
        for p in self.pose_history:
            if p['t'] <= t_s:
                before = p
            elif after is None:
                after = p
                break
        if before is None: return self.pose_history[0]
        if after  is None: return before
        dt = after['t'] - before['t']
        if dt < 1e-6: return before
        a = (t_s - before['t']) / dt
        return {k: before[k] + a*(after[k] - before[k]) for k in before}

    def _robot_to_world_at_pose(self, rx, ry, rz, pose: dict) -> tuple:
        """Transform a base_link point into the world frame using a given pose dict."""
        rot    = Rotation.from_quat([pose['qx'], pose['qy'], pose['qz'], pose['qw']])
        offset = np.array([self.base_link_offset_x, self.base_link_offset_y, self.base_link_offset_z])
        return tuple(np.array([pose['x'], pose['y'], pose['z']])
                     + rot.apply(offset + np.array([rx, ry, rz])))

    # ── Queue management ──────────────────────────────────────────────────────
    def will_be_in_pointing_range(self, track) -> tuple:
        """Check whether a target's lateral offset is within arm reach, and
        return (ready, |ry|, rx, estimated_time_to_arrival)."""
        curr_rx, curr_ry, curr_rz = self._world_to_robot_at(
            track.world_x, track.world_y, track.world_z,
            self.robot_x, self.robot_y, self.robot_z)
        abs_ry = abs(curr_ry)
        ready  = self.arm_reach_y_min <= abs_ry <= self.arm_reach_y_max
        spd    = math.sqrt(self.robot_vx**2 + self.robot_vy**2)
        d3     = math.sqrt(curr_rx**2 + curr_ry**2 + curr_rz**2)
        tta    = d3/spd if spd > 0.01 else d3*10.0
        return (ready, abs_ry, curr_rx, tta)

    def update_target_queue(self):
        """Rebuild and re-sort the dispatch queue: drop done/passed targets,
        append newly eligible ones, then sort by imminence."""
        if self.last_pose_time is None:
            return
        prev_len = len(self.target_queue)

        # Remove done/stale entries and targets the robot has already driven past.
        for tid in list(self.target_queue):
            if tid not in self.tracked_targets:
                self.target_queue.remove(tid)
                continue
            track = self.tracked_targets[tid]
            if track.beam_succeeded:
                self.target_queue.remove(tid)
                continue
            if self._is_position_published(track.world_x, track.world_y, track.world_z):
                track.beam_succeeded = True
                self.target_queue.remove(tid)
                continue
            # Purge targets the robot has already passed (curr_rx > 0.05m).
            # Leaving them in the queue would block the sort from reaching newer
            # reachable targets further down the track. Never purge the active target.
            if tid != self.current_target_id:
                curr_rx, _, _ = self._world_to_robot_at(
                    track.world_x, track.world_y, track.world_z,
                    self.robot_x, self.robot_y, self.robot_z)
                if curr_rx > 0.05:
                    track.beam_succeeded = True
                    self._remember_published_position(
                        track.world_x, track.world_y, track.world_z)
                    self.target_queue.remove(tid)

        # Append newly eligible targets.
        for tid, track in self.tracked_targets.items():
            if tid in self.target_queue: continue
            if track.times_seen < self.min_observations: continue
            if track.beam_succeeded: continue
            if self._is_position_published(track.world_x, track.world_y, track.world_z):
                track.beam_succeeded = True
                continue
            ok, _, _, _ = self.will_be_in_pointing_range(track)
            if not ok: continue
            rc = self._world_to_robot(track.world_x, track.world_y, track.world_z)
            if rc is None: continue
            if math.sqrt(rc[0]**2 + rc[1]**2) >= self.vision_max_distance: continue
            self.target_queue.append(tid)
            self.get_logger().info(f"Queued #{tid} wx={track.world_x:.3f}")

        # Sort by imminence: upcoming targets (rx <= 0.2) ordered closest-first,
        # already-passed targets pushed to the back.
        def _key(tid):
            if tid not in self.tracked_targets: return 999.0
            t = self.tracked_targets[tid]
            rx, _, _ = self._world_to_robot_at(
                t.world_x, t.world_y, t.world_z,
                self.robot_x, self.robot_y, self.robot_z)
            return 500.0 + rx if rx > 0.2 else -rx
        self.target_queue.sort(key=_key)

        # Throttled queue logging (at most once every 5s, only on change).
        if len(self.target_queue) != prev_len:
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.target_queue and (now - self.last_queue_log_time) > 5.0:
                self.last_queue_log_time = now
                ids = [f"#{t}(wx={self.tracked_targets[t].world_x:.2f})"
                       for t in self.target_queue[:6] if t in self.tracked_targets]
                self.get_logger().info(f"Queue({len(self.target_queue)}): {' '.join(ids)}")

    # ── Speed calculation ─────────────────────────────────────────────────────
    def calculate_speed_for_target(self, track, x_exec: float = None) -> float:
        """Compute the driving speed that lets the robot reach the firing point
        within the available planning + execution budget. Crawls if already past."""
        curr_rx, _, _ = self._world_to_robot_at(
            track.world_x, track.world_y, track.world_z,
            self.robot_x, self.robot_y, self.robot_z)
        if curr_rx >= 0.0:
            return self.min_speed
        if x_exec is None:
            x_exec = self.default_x_exec
        travel_needed = max(0.0, abs(curr_rx) - abs(x_exec))
        total_budget  = self.arm_movement_time + self.planning_time_est
        required = travel_needed / total_budget if total_budget > 0.1 else self.max_speed
        return float(np.clip(required, self.min_speed, self.max_speed))

    # ── Beam callbacks ────────────────────────────────────────────────────────
    def beam_done_callback(self, msg: Bool):
        """Handle a successful beam: mark the target done, reset to idle, slow
        the platform, and immediately attempt to dispatch the next target."""
        if msg.data and self.state == "TARGETING":
            if self.current_target_id in self.tracked_targets:
                track = self.tracked_targets[self.current_target_id]
                track.beam_succeeded = True
                self._remember_published_position(track.world_x, track.world_y, track.world_z)
                self.last_completed_x = self.robot_x
                self.get_logger().info(
                    f"SUCCESS #{self.current_target_id} | "
                    f"world=[{track.world_x:.3f},{track.world_y:.3f},{track.world_z:.3f}] "
                    f"robot_x={self.robot_x:.3f}")
            self._reset_to_idle()

            # Always drop to min_speed after a hit. Following the Phase 2 sweep
            # the platform may be at max_speed; the line follower needs time to
            # decelerate before the next arm trajectory begins, for both close
            # and far next targets.
            self._publish_speed(self.min_speed)

            stamp = self.get_clock().now().to_msg()
            self.update_target_queue()
            self.maybe_publish_next_target(stamp)

    def beam_failed_callback(self, msg: Bool):
        """Handle a failed beam: retry if attempts remain and the target is
        still reachable, otherwise mark it done and move on."""
        if msg.data and self.state == "TARGETING":
            self.recovery_until = self.get_clock().now().nanoseconds * 1e-9 + self.recovery_time
            if self.current_target_id in self.tracked_targets:
                track = self.tracked_targets[self.current_target_id]
                self.last_completed_x = self.robot_x

                curr_rx, _, _ = self._world_to_robot_at(
                    track.world_x, track.world_y, track.world_z,
                    self.robot_x, self.robot_y, self.robot_z)
                # Only abandon the retry if the ball is physically behind the
                # robot (curr_rx > 0); otherwise it is still reachable via
                # direct-aim mode (stop robot, aim at current ball position).
                too_close = curr_rx > 0.0

                if too_close:
                    track.beam_succeeded = True
                    self._remember_published_position(track.world_x, track.world_y, track.world_z)
                    self.get_logger().warn(
                        f"SKIP RETRY #{self.current_target_id}: "
                        f"curr_rx={curr_rx:.3f}m — no sweep distance left")
                elif track.beam_attempts < track.max_beam_attempts:
                    self.get_logger().warn(f"RETRY #{self.current_target_id}")
                else:
                    track.beam_succeeded = True
                    self._remember_published_position(track.world_x, track.world_y, track.world_z)
                    self.get_logger().error(f"GAVE UP #{self.current_target_id}")
            self._reset_to_idle()

    def _reset_to_idle(self):
        """Return to the DRIVING state and clear per-target dispatch fields."""
        self.state = "DRIVING"
        self.current_target_id  = None
        self.targeting_start_time = None
        self.direct_aim_mode = False

    # ── Robot pose ────────────────────────────────────────────────────────────
    def ground_truth_callback(self, msg: PoseStamped):
        """Update robot pose and velocity from ground truth, store a stamped
        sample for latency compensation, and refresh speed control."""
        if msg.header.frame_id != self.ground_truth_frame_id:
            return
        prev_x, prev_y = self.robot_x, self.robot_y
        prev_time = self.last_pose_time
        self.robot_x  = msg.pose.position.x
        self.robot_y  = msg.pose.position.y
        self.robot_z  = msg.pose.position.z
        self.robot_qx = msg.pose.orientation.x
        self.robot_qy = msg.pose.orientation.y
        self.robot_qz = msg.pose.orientation.z
        self.robot_qw = msg.pose.orientation.w
        current_time = self.get_clock().now()
        if prev_time is not None:
            dt = (current_time - prev_time).nanoseconds * 1e-9
            if dt > 0.001:
                self.robot_vx = (self.robot_x - prev_x) / dt
                self.robot_vy = (self.robot_y - prev_y) / dt
        # Store stamped pose for camera/GT latency compensation.
        msg_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pose_history.append({
            't': msg_t,
            'x': self.robot_x, 'y': self.robot_y, 'z': self.robot_z,
            'qx': self.robot_qx, 'qy': self.robot_qy,
            'qz': self.robot_qz, 'qw': self.robot_qw})
        if prev_time is None:
            self.get_logger().info(f'First GT: x={self.robot_x:.3f} y={self.robot_y:.3f}')
        self.last_pose_time = current_time
        self._update_camera_transform()
        self._update_driving_speed()
        now_s = current_time.nanoseconds * 1e-9
        if not hasattr(self, "_lgl") or (now_s - self._lgl) > 3.0:
            self._lgl = now_s
            spd = math.sqrt(self.robot_vx**2 + self.robot_vy**2)
            self.get_logger().info(
                f"pos=[{self.robot_x:.3f},{self.robot_y:.3f}] spd={spd:.3f}m/s "
                f"state={self.state} q={len(self.target_queue)}")

    def _update_camera_transform(self):
        """Cache the latest camera->base_link transform; ignore lookup failures."""
        try:
            self.camera_to_robot_transform = self.tf_buffer.lookup_transform(
                self.robot_frame, self.camera_frame, rclpy.time.Time())
        except Exception:
            pass

    def _update_driving_speed(self):
        """Drive the platform speed based on the current state.

        TARGETING uses three phases:
          Phase 1 (planning + executing): near-stop so the collision scene and
            joint trajectory stay valid.
          Phase 2 (hold/sweep): moderate speed so the beam sweeps through the
            window without overshooting it.
          direct_aim_mode: near-stop throughout, since any motion shifts the
            beam off the already-aimed ball position.

        Non-TARGETING states handle Gate 1 (drive fast until past the last
        completed target), a proximity brake near the next target, and a
        proportional approach toward the firing point.
        """
        if self.last_pose_time is None or not self.target_queue:
            return

        # ── 1. TARGETING ─────────────────────────────────────────────────────
        if self.state == "TARGETING":
            speed = self.min_speed
            if (self.current_target_id is not None
                    and self.current_target_id in self.tracked_targets):

                elapsed = 0.0
                if self.targeting_start_time is not None:
                    elapsed = (self.get_clock().now().nanoseconds * 1e-9
                               - self.targeting_start_time)
                arm_cycle = self.planning_time_est + self.arm_movement_time

                if elapsed < self.planning_time_est:
                    # Phase 1a — arm planning: stay still so the collision scene is valid.
                    speed = 0.01 if self.direct_aim_mode else self.min_speed
                elif elapsed < arm_cycle:
                    # Phase 1b — arm executing: crawl while it follows the trajectory.
                    speed = 0.01 if self.direct_aim_mode else self.min_speed
                else:
                    # Phase 2 — arm done, hold/sweep running. Moderate speed:
                    # fast enough to traverse the ~280mm window in reasonable
                    # time, slow enough not to blow past it.
                    if self.direct_aim_mode:
                        speed = 0.01
                    else:
                        speed = float(np.clip(self.min_speed * 3,
                                              self.min_speed, self.max_speed * 0.3))

            self._publish_speed(speed)
            return

        # ── 2. Gate 1: not yet past the last completed target — drive fast ────
        if self.last_completed_x is not None and self.robot_x > self.last_completed_x:
            self._publish_speed(self.max_speed)
            return

        # ── 3. Proximity brake within 0.8m of the next target ─────────────────
        if self.target_queue:
            tid0 = self.target_queue[0]
            if tid0 in self.tracked_targets:
                t0 = self.tracked_targets[tid0]
                rx0, _, _ = self._world_to_robot_at(
                    t0.world_x, t0.world_y, t0.world_z,
                    self.robot_x, self.robot_y, self.robot_z)
                if rx0 > -0.8:
                    self._publish_speed(self.min_speed)
                    return

        # ── 4. Proportional approach toward the firing point ──────────────────
        best_rx = None
        for tid in self.target_queue[:4]:
            if tid not in self.tracked_targets: continue
            rx, _, _ = self._world_to_robot_at(
                self.tracked_targets[tid].world_x,
                self.tracked_targets[tid].world_y,
                self.tracked_targets[tid].world_z,
                self.robot_x, self.robot_y, self.robot_z)
            if not (-self.vision_max_distance < rx < 0.0): continue
            if best_rx is None or rx > best_rx: best_rx = rx
        if best_rx is None:
            return

        firing_offset = abs(self.default_x_exec)
        distance = max(0.0, abs(best_rx) - firing_offset)
        now = self.get_clock().now().nanoseconds * 1e-9
        recovery_remaining = max(0.0, self.recovery_until - now)
        t_budget = self.arm_movement_time + self.planning_time_est + recovery_remaining
        required = distance / t_budget if t_budget > 0.1 else self.max_speed
        self._publish_speed(float(np.clip(required, self.min_speed, self.max_speed)))

    # ── Camera ────────────────────────────────────────────────────────────────
    def cam_info_callback(self, msg: CameraInfo):
        """Compute pinhole intrinsics once from the configured horizontal FOV."""
        if self.fx is not None: return
        w, h = float(msg.width), float(msg.height)
        self.fx = (w / 2.0) / math.tan(self.camera_hfov / 2.0)
        self.fy = self.fx
        self.cx, self.cy = w / 2.0, h / 2.0

    def depth_callback(self, msg: Image):
        """Cache the latest depth image (float32, metres)."""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception:
            pass

    # ── Detection ─────────────────────────────────────────────────────────────
    def rgb_callback(self, msg: Image):
        """Per-frame pipeline: detect balls, project to world, track, requeue,
        dispatch, and publish the full tracked set."""
        if self.latest_depth is None or self.fx is None: return
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return
        stamp        = msg.header.stamp
        current_time = self.get_clock().now().nanoseconds * 1e-9
        dets = self.find_balls(rgb)
        if not dets: return
        dw = [w for d in dets if (w := self.detection_to_world(d, stamp)) is not None]
        if not dw: return
        self.update_tracking(dw, current_time)
        self.update_target_queue()
        self.maybe_publish_next_target(stamp)
        self.publish_all_tracked(stamp)

    # ── Tracking ──────────────────────────────────────────────────────────────
    def update_tracking(self, detections_world, current_time):
        """Associate detections with existing tracks (nearest neighbour within
        association_radius) and spawn new tracks for unmatched, in-range,
        plausibly-placed detections. EMA alpha scales down with robot speed."""
        used_tracks, used_dets = set(), set()
        spd       = math.sqrt(self.robot_vx**2 + self.robot_vy**2)
        ema_alpha = 0.55 - float(np.clip(spd / max(self.max_speed, 1e-6), 0., 1.)) * 0.45

        for i, (wx, wy, wz, q) in enumerate(detections_world):
            best_id, best_d = None, self.association_radius
            for tid, track in self.tracked_targets.items():
                if tid in used_tracks: continue
                d = track.distance_to(wx, wy, wz)
                if d < best_d: best_d = d; best_id = tid
            if best_id is not None:
                self.tracked_targets[best_id].update(wx, wy, wz, current_time, alpha=ema_alpha)
                used_tracks.add(best_id); used_dets.add(i)

        for i, (wx, wy, wz, q) in enumerate(detections_world):
            if i in used_dets: continue
            if any(t.distance_to(wx, wy, wz) < self.association_radius
                   for t in self.tracked_targets.values()): continue
            dr = math.sqrt((wx - self.robot_x)**2 + (wy - self.robot_y)**2)
            if dr > self.max_detection_range: continue
            if not (-0.3 <= wz <= 0.1): continue
            if not (0.5  <= wy <= self.arm_reach_y_max + 0.2): continue
            nid = self.next_target_id; self.next_target_id += 1
            self.tracked_targets[nid] = TrackedTarget(
                id=nid, world_x=wx, world_y=wy, world_z=wz,
                first_seen_time=current_time, last_seen_time=current_time)
            self.get_logger().info(
                f"NEW #{nid}: [{wx:.3f},{wy:.3f},{wz:.3f}] d={dr:.2f}m")

    def maintenance_callback(self):
        """Periodic housekeeping: prune stale completed tracks, force-reset a
        stuck TARGETING state, and re-run the queue/dispatch step."""
        now = self.get_clock().now().nanoseconds * 1e-9
        to_rm = [tid for tid, t in self.tracked_targets.items()
                 if now - t.last_seen_time > self.track_timeout
                 and tid != self.current_target_id
                 and t.beam_succeeded]
        for tid in to_rm:
            del self.tracked_targets[tid]
            if tid in self.target_queue: self.target_queue.remove(tid)

        # Force-reset if stuck in TARGETING beyond the full arm cycle. The
        # timeout must cover planning + execution + hold phase + margin so it
        # never fires before beam_done/beam_failed arrives and credits the
        # wrong target: planning + arm_movement_time + arm_hold_time_est + 5.
        if self.state == "TARGETING" and self.targeting_start_time is not None:
            elapsed = now - self.targeting_start_time
            timeout = self.planning_time_est + self.arm_movement_time + self.arm_hold_time_est + 5.0
            if elapsed > timeout:
                self.get_logger().warn(
                    f"TARGETING TIMEOUT #{self.current_target_id} "
                    f"({elapsed:.1f}s > {timeout:.1f}s) — force-resetting.")
                if self.current_target_id in self.tracked_targets:
                    track = self.tracked_targets[self.current_target_id]
                    track.beam_succeeded = True
                    self._remember_published_position(
                        track.world_x, track.world_y, track.world_z)
                    self.last_completed_x = self.robot_x
                self._reset_to_idle()

        if self.last_pose_time is not None:
            stamp = self.get_clock().now().to_msg()
            self.update_target_queue()
            self.maybe_publish_next_target(stamp)

    # ── Target publishing ─────────────────────────────────────────────────────
    def maybe_publish_next_target(self, stamp):
        """Dispatch the front-of-queue target if the gates pass, computing the
        arm aim point (x_exec) and starting speed, and entering TARGETING."""
        if self.state == "TARGETING": return
        if not self.target_queue: return
        if len(self.tracked_targets) < self.min_targets_before_publish: return

        # Gate 1: must have passed the last completed target.
        if self.last_completed_x is not None and self.robot_x > self.last_completed_x:
            return

        next_tid = self.target_queue[0]
        if next_tid not in self.tracked_targets:
            self.target_queue.pop(0)
            return

        # Gate 2: target must be within 0.8m ahead in robot-frame X.
        track = self.tracked_targets[next_tid]
        curr_rx, curr_ry, curr_rz = self._world_to_robot_at(
            track.world_x, track.world_y, track.world_z,
            self.robot_x, self.robot_y, self.robot_z)
        if curr_rx < -0.8:
            return

        # Gate 3: if the robot already drove past the target (curr_rx > 0.05)
        # there is no sweep distance left and pointing behind the robot is out
        # of IK reach, so mark it done and move on.
        if curr_rx > 0.05:
            self.get_logger().warn(
                f"Skip #{next_tid}: already past (curr_rx={curr_rx:.3f}m)")
            track.beam_succeeded = True
            self._remember_published_position(track.world_x, track.world_y, track.world_z)
            self.last_completed_x = self.robot_x
            self.target_queue.pop(0)
            return

        track.beam_attempts += 1
        track.published = True

        # x_exec is where in base_link X the arm aims the beam tip. The model
        # (empirically verified) is:
        #   tip_world_x  = robot_world_x + x_exec
        #   ball_world_x = robot_world_x + curr_rx
        #   sweep_at_arm_finish = x_exec - curr_rx - pred_speed * arm_cycle
        # For the beam to lead the ball when the arm finishes (sweep > 0) and to
        # achieve a desired sweep distance:
        #   x_exec = curr_rx + pred_speed * arm_cycle + desired_sweep
        # A fixed x_exec overshoots close targets, so we branch on curr_rx below.
        pred_speed   = min(math.sqrt(self.robot_vx**2 + self.robot_vy**2), 0.08)
        arm_cycle    = self.planning_time_est + self.arm_movement_time

        # Sweep mode only works while the beam still leads the ball at arm-finish:
        #   sweep_at_finish = x_exec - curr_rx - min_speed*arm_cycle > 0
        #   => curr_rx < default_x_exec - min_speed*arm_cycle
        # Past that threshold the beam would already be behind the ball, so we
        # switch to direct-aim: stop the robot and aim at the current ball X.
        sweep_threshold = self.default_x_exec - self.min_speed * arm_cycle
        self.direct_aim_mode = curr_rx > sweep_threshold

        if self.direct_aim_mode:
            x_exec = float(np.clip(curr_rx,
                                   self.arm_reach_x_min + 0.02,
                                   -0.02))   # keep negative so the arm points ahead
            self.get_logger().info(
                f"Direct-aim #{next_tid}: stopping robot, x_exec={x_exec:.3f}m "
                f"(curr_rx={curr_rx:.3f}m, threshold={sweep_threshold:.3f}m)")
        else:
            x_exec = self.default_x_exec  # sweep_at_finish guaranteed > 0

        # Arm target in base_link frame.
        pose = PoseStamped()
        pose.header.stamp    = stamp
        pose.header.frame_id = self.robot_frame
        pose.pose.position.x = x_exec
        pose.pose.position.y = curr_ry
        pose.pose.position.z = curr_rz
        pose.pose.orientation.w = 1.0
        self.target_pub.publish(pose)

        # Ball world position for the arm hold-phase X-sweep.
        world_msg = PointStamped()
        world_msg.header.stamp    = stamp
        world_msg.header.frame_id = self.world_frame
        world_msg.point.x = track.world_x
        world_msg.point.y = track.world_y
        world_msg.point.z = track.world_z
        self.target_world_pub.publish(world_msg)

        # Robot-frame point for other subscribers (e.g. the line follower).
        rpt = PointStamped()
        rpt.header.stamp    = stamp
        rpt.header.frame_id = self.robot_frame
        rpt.point.x = curr_rx
        rpt.point.y = curr_ry
        rpt.point.z = curr_rz
        self.target_point_pub.publish(rpt)

        speed = self.calculate_speed_for_target(track, x_exec)
        self._publish_speed(speed)

        self.state             = "TARGETING"
        self.current_target_id = next_tid
        self.current_x_exec    = x_exec
        self.targeting_start_time = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info(
            f"#{next_tid} attempt={track.beam_attempts} | "
            f"curr_rx={curr_rx:.3f} -> x_exec={x_exec:.3f}m | "
            f"robot=[{self.robot_x:.3f},{self.robot_y:.3f}] "
            f"world=[{track.world_x:.3f},{track.world_y:.3f},{track.world_z:.3f}] "
            f"spd={speed:.3f}m/s")

    def _publish_speed(self, speed: float):
        """Publish the platform slowdown speed."""
        m = Float32()
        m.data = float(speed)
        self.target_speed_pub.publish(m)

    def _is_position_published(self, x, y, z):
        """True if a target near (x,y,z) has already been completed."""
        return any(
            math.sqrt((x - px)**2 + (y - py)**2 + (z - pz)**2) < self.published_position_radius
            for px, py, pz in self.published_positions)

    def _remember_published_position(self, x, y, z):
        """Record a completed target position (bounded history) to suppress
        re-dispatch of the same ball."""
        self.published_positions.append((x, y, z))
        if len(self.published_positions) > 100:
            self.published_positions.pop(0)

    def publish_all_tracked(self, stamp):
        """Publish all current tracks as a world-frame PoseArray."""
        pa = PoseArray()
        pa.header.stamp    = stamp
        pa.header.frame_id = self.world_frame
        for t in self.tracked_targets.values():
            p = Pose()
            p.position.x = t.world_x
            p.position.y = t.world_y
            p.position.z = t.world_z
            p.orientation.w = 1.0
            pa.poses.append(p)
        self.all_targets_pub.publish(pa)

    # ── Detection helpers ─────────────────────────────────────────────────────
    def pixel_to_ray(self, u, v):
        """Back-project a pixel to a unit ray in the camera optical frame."""
        r = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        return r / np.linalg.norm(r)

    def surface_to_center(self, u, v, depth):
        """Convert a pixel + measured surface depth into the ball-centre point,
        nudging outward by one ball radius along the ray."""
        ray = self.pixel_to_ray(u, v)
        c   = ray * (depth / ray[2]) + ray * self.ball_radius
        return float(c[0]), float(c[1]), float(c[2])

    def get_gaussian_weighted_depth(self, cx, cy):
        """Gaussian-weighted depth over a small ROI around (cx,cy), ignoring
        invalid samples. Returns (depth, confidence) or falls back to the single
        centre pixel; None if no valid depth is available."""
        if self.latest_depth is None: return None
        h, w = self.latest_depth.shape
        r    = self.depth_sample_radius
        ym, yM = max(0, int(cy - r)), min(h, int(cy + r + 1))
        xm, xM = max(0, int(cx - r)), min(w, int(cx + r + 1))
        if yM <= ym or xM <= xm: return None
        roi = self.latest_depth[ym:yM, xm:xM].copy()
        rh, rw = roi.shape
        yc, xc = np.ogrid[:rh, :rw]
        sigma  = r / 2.0
        wts = np.exp(-((xc - (cx - xm))**2 + (yc - (cy - ym))**2) / (2 * sigma**2))
        vm  = np.isfinite(roi) & (roi > 0.05) & (roi < 5.0)
        if np.sum(vm) < 3:
            ix, iy = int(cx), int(cy)
            if 0 <= ix < w and 0 <= iy < h:
                d = self.latest_depth[iy, ix]
                if np.isfinite(d) and d > 0:
                    return float(d), 0.5
            return None
        wd   = np.average(roi[vm], weights=wts[vm])
        conf = max(0.3, 1.0 - min(1.0, np.std(roi[vm]) / 0.01))
        return float(wd), conf

    def find_balls(self, bgr):
        """Detect blue balls via HSV thresholding + morphology + contour
        filtering. Returns a list of detection dicts with centre, radius, area
        and a circularity-based quality score."""
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 80, 50]), np.array([130, 255, 255]))
        k    = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            a = cv2.contourArea(c)
            if not (self.min_ball_area <= a <= self.max_ball_area): continue
            if len(c) < self.min_contour_points: continue
            (cx, cy), rad = cv2.minEnclosingCircle(c)
            ca = math.pi * rad * rad
            dets.append({'cx': cx, 'cy': cy, 'radius_px': rad, 'area': a,
                         'quality': min(1., a / ca) if ca > 0 else 0.5})
        return dets

    def detection_to_world(self, det, stamp):
        """Project a detection to a world-frame point: depth -> camera-frame
        centre -> base_link via TF -> world via the (latency-compensated) pose.
        Returns (wx, wy, wz, quality) or None on failure."""
        dr = self.get_gaussian_weighted_depth(det['cx'], det['cy'])
        if dr is None: return None
        depth, dc = dr
        quality   = det['quality'] * dc
        Xc, Yc, Zc = self.surface_to_center(det['cx'], det['cy'], depth)
        pt = PointStamped()
        pt.header.stamp    = stamp
        pt.header.frame_id = self.camera_frame
        pt.point.x, pt.point.y, pt.point.z = Xc, Yc, Zc
        try:
            tf = self.tf_buffer.lookup_transform(
                self.robot_frame, self.camera_frame, rclpy.time.Time())
            pr = tf2_geometry_msgs.do_transform_point(pt, tf)
            rx, ry, rz = pr.point.x, pr.point.y, pr.point.z
            img_t = stamp.sec + stamp.nanosec * 1e-9
            hp    = self._get_pose_at(img_t)
            if hp is not None:
                wx, wy, wz = self._robot_to_world_at_pose(rx, ry, rz, hp)
            else:
                wx, wy, wz = self._robot_to_world(rx, ry, rz)
            return (wx, wy, wz, quality)
        except Exception:
            return None


def main(args=None):
    rclpy.init(args=args)
    node = PredictiveMultiTargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()