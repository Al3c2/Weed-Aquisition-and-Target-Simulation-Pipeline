#!/usr/bin/env python3


import csv
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry


class TargetStatus(Enum):
    """Status of a published target"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class PublishedTarget:
    """Record of a published target"""
    publish_time: float
    world_pos: Tuple[float, float, float]
    robot_pos: Tuple[float, float, float]
    matched_gt_name: Optional[str] = None
    matched_gt_pos: Optional[Tuple[float, float, float]] = None
    position_error_m: Optional[float] = None
    error_vector: Optional[Tuple[float, float, float]] = None  # (dx, dy, dz) vector
    status: TargetStatus = TargetStatus.PENDING
    beam_complete_time: Optional[float] = None
    pointing_duration_s: Optional[float] = None


@dataclass
class GroundTruth:
    """Ground truth target from CSV"""
    name: str
    world_pos: Tuple[float, float, float]
    detected: bool = False
    detection_time: Optional[float] = None


def dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance in 3D"""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


class DetectionEvaluatorV3(Node):
    """
    Detection evaluator with full orientation support.
    
    Now properly handles robot pitch/roll on bumpy terrain.
    """

    def __init__(self):
        super().__init__("detection_evaluator_v3")

        # ========== Parameters ==========
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter("beam_done_topic", "/beam_task_done")
        self.declare_parameter("beam_failed_topic", "/beam_task_failed")
        
        self.declare_parameter("use_ground_truth_pose", True)
        self.declare_parameter("ground_truth_topic", "/model/swincar_ur3/pose")
        self.declare_parameter("ground_truth_frame_id", "empty")
        self.declare_parameter("odom_topic", "/swincar/odom")
        
        self.declare_parameter("base_link_offset_x", 0.0)
        self.declare_parameter("base_link_offset_y", 0.2)
        self.declare_parameter("base_link_offset_z", 0.35)
        
        self.declare_parameter("gt_world_csv", "gt_world.csv")
        self.declare_parameter("match_threshold_m", 0.10)
        
        self.declare_parameter("output_dir", "/home/alex/tese_ws/src/plots")
        self.declare_parameter("summary_csv", "eval_summary.csv")
        self.declare_parameter("detailed_csv", "eval_detailed.csv")
        
        self.declare_parameter("log_interval_s", 5.0)
        self.declare_parameter("max_gt_targets", 0)

        # ========== Load Parameters ==========
        self.target_pose_topic = self.get_parameter("target_pose_topic").value
        self.beam_done_topic = self.get_parameter("beam_done_topic").value
        self.beam_failed_topic = self.get_parameter("beam_failed_topic").value
        
        self.use_ground_truth_pose = self.get_parameter("use_ground_truth_pose").value
        self.ground_truth_topic = self.get_parameter("ground_truth_topic").value
        self.ground_truth_frame_id = self.get_parameter("ground_truth_frame_id").value
        self.odom_topic = self.get_parameter("odom_topic").value
        
        self.base_link_offset_x = self.get_parameter("base_link_offset_x").value
        self.base_link_offset_y = self.get_parameter("base_link_offset_y").value
        self.base_link_offset_z = self.get_parameter("base_link_offset_z").value
        
        self.gt_csv_path = self.get_parameter("gt_world_csv").value
        self.match_threshold = float(self.get_parameter("match_threshold_m").value)
        
        self.output_dir = self.get_parameter("output_dir").value
        summary_filename = self.get_parameter("summary_csv").value
        detailed_filename = self.get_parameter("detailed_csv").value
        
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.summary_csv = os.path.join(self.output_dir, summary_filename)
        self.detailed_csv = os.path.join(self.output_dir, detailed_filename)
        
        self.log_interval = float(self.get_parameter("log_interval_s").value)
        self.max_gt_targets = int(self.get_parameter("max_gt_targets").value)

        # ========== State ==========
        # Robot pose WITH ORIENTATION
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.robot_qx = 0.0
        self.robot_qy = 0.0
        self.robot_qz = 0.0
        self.robot_qw = 1.0
        self.pose_initialized = False
        
        # Ground truth database
        self.ground_truth: Dict[str, GroundTruth] = {}
        self._load_ground_truth()
        
        # Published targets tracking
        self.published_targets: List[PublishedTarget] = []
        self.current_pending: Optional[PublishedTarget] = None
        
        # Timing
        self.run_start_time = time.time()
        self.last_log_time = 0.0
        
        # ========== Subscriptions ==========
        self.sub_target_pose = self.create_subscription(
            PoseStamped, self.target_pose_topic, self.on_target_pose, 10
        )
        
        self.sub_beam_done = self.create_subscription(
            Bool, self.beam_done_topic, self.on_beam_done, 10
        )
        
        self.sub_beam_failed = self.create_subscription(
            Bool, self.beam_failed_topic, self.on_beam_failed, 10
        )
        
        if self.use_ground_truth_pose:
            self.sub_robot_pose = self.create_subscription(
                PoseStamped, self.ground_truth_topic, self.on_ground_truth_pose, 10
            )
        else:
            self.sub_odom = self.create_subscription(
                Odometry, self.odom_topic, self.on_odom, 10
            )
        
        # Periodic logging
        self.create_timer(self.log_interval, self.periodic_log)
        
        self._print_startup()

    def _print_startup(self):
        self.get_logger().info("Using ground truth pose from: " + self.ground_truth_topic)
        self.get_logger().info("=" * 60)
        self.get_logger().info("Detection Evaluator v3 - WITH ORIENTATION SUPPORT")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Ground truth: {len(self.ground_truth)} targets from {self.gt_csv_path}")
        self.get_logger().info(f"Target pose topic: {self.target_pose_topic}")
        self.get_logger().info(f"Match threshold: {self.match_threshold * 1000:.1f} mm")
        self.get_logger().info(f"Base link offset: [{self.base_link_offset_x}, {self.base_link_offset_y}, {self.base_link_offset_z}]")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info("=" * 60)

    def _load_ground_truth(self):
        """Load ground truth targets from CSV"""
        try:
            with open(self.gt_csv_path, "r") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    if self.max_gt_targets > 0 and count >= self.max_gt_targets:
                        break
                    name = row["name"]
                    pos = (float(row["x"]), float(row["y"]), float(row["z"]))
                    self.ground_truth[name] = GroundTruth(name=name, world_pos=pos)
                    count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to load GT CSV: {e}")
            raise

    # ========== Robot Pose Callbacks (WITH ORIENTATION) ==========
    
    def on_ground_truth_pose(self, msg: PoseStamped):
        """Handle ground truth robot pose INCLUDING ORIENTATION."""
        if msg.header.frame_id != self.ground_truth_frame_id:
            return
        
        # Position
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.robot_z = msg.pose.position.z
        
        # Orientation
        self.robot_qx = msg.pose.orientation.x
        self.robot_qy = msg.pose.orientation.y
        self.robot_qz = msg.pose.orientation.z
        self.robot_qw = msg.pose.orientation.w
        
        if not self.pose_initialized:
            self.pose_initialized = True
            rot = Rotation.from_quat([self.robot_qx, self.robot_qy, self.robot_qz, self.robot_qw])
            roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
            self.get_logger().info(
                f"Robot pose initialized: [{self.robot_x:.3f}, {self.robot_y:.3f}, {self.robot_z:.3f}] "
                f"roll={roll:.1f}° pitch={pitch:.1f}°"
            )
    
    def on_odom(self, msg: Odometry):
        """Handle odometry pose INCLUDING ORIENTATION."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_z = msg.pose.pose.position.z
        
        self.robot_qx = msg.pose.pose.orientation.x
        self.robot_qy = msg.pose.pose.orientation.y
        self.robot_qz = msg.pose.pose.orientation.z
        self.robot_qw = msg.pose.pose.orientation.w
        
        if not self.pose_initialized:
            self.pose_initialized = True
            self.get_logger().info(
                f"Robot pose initialized (odom): [{self.robot_x:.3f}, {self.robot_y:.3f}, {self.robot_z:.3f}]"
            )

    def _base_link_to_world(self, x_bl: float, y_bl: float, z_bl: float) -> Tuple[float, float, float]:
        """
        Convert base_link coordinates to world coordinates.
        
        NOW USES FULL ORIENTATION for proper transform on bumpy terrain.
        """
        # Offset from swincar origin to base_link (in swincar body frame)
        offset = np.array([self.base_link_offset_x, self.base_link_offset_y, self.base_link_offset_z])
        
        # Point in base_link frame
        point_robot = np.array([x_bl, y_bl, z_bl])
        
        # Get rotation matrix from quaternion
        rot = Rotation.from_quat([self.robot_qx, self.robot_qy, self.robot_qz, self.robot_qw])
        
        # Transform: world_point = robot_pos + R * (offset + point_robot)
        point_body = offset + point_robot
        point_world = np.array([self.robot_x, self.robot_y, self.robot_z]) + rot.apply(point_body)
        
        return tuple(point_world)

    # ========== Target Pose Callback ==========
    
    def on_target_pose(self, msg: PoseStamped):
        """Handle a newly published target pose."""
        now = time.time()
        
        if not self.pose_initialized:
            self.get_logger().warn("Received target_pose but robot pose not initialized yet")
            return
        
        # Extract position in base_link frame
        x_bl = msg.pose.position.x
        y_bl = msg.pose.position.y
        z_bl = msg.pose.position.z
        
        # Convert to world frame (NOW WITH ORIENTATION!)
        world_pos = self._base_link_to_world(x_bl, y_bl, z_bl)
        
        # Match against ground truth
        matched_name, matched_pos, error_magnitude, error_vector = self._match_to_gt(world_pos)
        
        # Create target record
        target = PublishedTarget(
            publish_time=now,
            world_pos=world_pos,
            robot_pos=(x_bl, y_bl, z_bl),
            matched_gt_name=matched_name,
            matched_gt_pos=matched_pos,
            position_error_m=error_magnitude,
            error_vector=error_vector,
            status=TargetStatus.PENDING
        )
        
        self.published_targets.append(target)
        self.current_pending = target
        
        # Mark GT as detected
        if matched_name:
            gt = self.ground_truth[matched_name]
            if not gt.detected:
                gt.detected = True
                gt.detection_time = now
        
        # Log
        if matched_name:
            self.get_logger().info(
                f"📍 TARGET #{len(self.published_targets)}: "
                f"world=[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}] "
                f"→ GT '{matched_name}' (error: {error_magnitude*1000:.1f}mm)"
            )
        else:
            self.get_logger().warn(
                f"⚠️ TARGET #{len(self.published_targets)}: "
                f"world=[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}] "
                f"→ NO GT MATCH (FP)"
            )

    def _match_to_gt(self, world_pos: Tuple[float, float, float]) -> Tuple[Optional[str], Optional[Tuple[float, float, float]], Optional[float], Optional[Tuple[float, float, float]]]:
        """Match a world position to the nearest ground truth target.
        
        Returns: (matched_name, matched_pos, error_magnitude, error_vector)
        """
        best_name = None
        best_pos = None
        best_dist = float("inf")
        best_vector = None
        
        for name, gt in self.ground_truth.items():
            d = dist3(world_pos, gt.world_pos)
            if d < best_dist:
                best_dist = d
                best_name = name
                best_pos = gt.world_pos
                # Store error vector: detected - ground_truth
                best_vector = (
                    world_pos[0] - gt.world_pos[0],
                    world_pos[1] - gt.world_pos[1],
                    world_pos[2] - gt.world_pos[2]
                )
        
        if best_name and best_dist <= self.match_threshold:
            return (best_name, best_pos, best_dist, best_vector)
        else:
            return (None, None, None, None)

    # ========== Beam Result Callbacks ==========
    
    def on_beam_done(self, msg: Bool):
        """Handle successful beam completion"""
        if not msg.data:
            return
            
        now = time.time()
        
        if self.current_pending is None:
            self.get_logger().warn("Beam done received but no pending target")
            return
        
        self.current_pending.status = TargetStatus.SUCCESS
        self.current_pending.beam_complete_time = now
        self.current_pending.pointing_duration_s = now - self.current_pending.publish_time
        
        matched = self.current_pending.matched_gt_name or "FP"
        self.get_logger().info(
            f"✅ BEAM SUCCESS: target '{matched}' "
            f"(pointing time: {self.current_pending.pointing_duration_s:.2f}s)"
        )
        
        self.current_pending = None

    def on_beam_failed(self, msg: Bool):
        """Handle failed beam task"""
        if not msg.data:
            return
            
        now = time.time()
        
        if self.current_pending is None:
            self.get_logger().warn("Beam failed received but no pending target")
            return
        
        self.current_pending.status = TargetStatus.FAILED
        self.current_pending.beam_complete_time = now
        self.current_pending.pointing_duration_s = now - self.current_pending.publish_time
        
        matched = self.current_pending.matched_gt_name or "FP"
        self.get_logger().warn(
            f"❌ BEAM FAILED: target '{matched}' "
            f"(pointing time: {self.current_pending.pointing_duration_s:.2f}s)"
        )
        
        self.current_pending = None

    # ========== Metrics Computation ==========
    
    def compute_metrics(self) -> dict:
        """Compute comprehensive evaluation metrics"""
        total_published = len(self.published_targets)
        
        # Ground truth stats
        gt_total = len(self.ground_truth)
        gt_detected = sum(1 for gt in self.ground_truth.values() if gt.detected)
        gt_missed = gt_total - gt_detected
        
        # Classification
        tp_targets = [t for t in self.published_targets if t.matched_gt_name is not None]
        fp_targets = [t for t in self.published_targets if t.matched_gt_name is None]
        
        tp_count = len(tp_targets)
        fp_count = len(fp_targets)
        fn_count = gt_missed
        
        # Position errors (only for TPs)
        errors = [t.position_error_m for t in tp_targets if t.position_error_m is not None]
        
        if errors:
            mean_error = sum(errors) / len(errors)
            rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
            max_error = max(errors)
            min_error = min(errors)
        else:
            mean_error = rmse = max_error = min_error = 0.0
        
        # Beam results
        beam_success_tp = sum(1 for t in tp_targets if t.status == TargetStatus.SUCCESS)
        beam_success_fp = sum(1 for t in fp_targets if t.status == TargetStatus.SUCCESS)
        beam_failed = sum(1 for t in self.published_targets if t.status == TargetStatus.FAILED)
        beam_pending = sum(1 for t in self.published_targets if t.status == TargetStatus.PENDING)
        
        # Timing
        total_run_time = time.time() - self.run_start_time
        
        pointing_times = [t.pointing_duration_s for t in self.published_targets 
                         if t.pointing_duration_s is not None]
        avg_pointing_time = sum(pointing_times) / len(pointing_times) if pointing_times else 0.0
        max_pointing_time = max(pointing_times) if pointing_times else 0.0
        min_pointing_time = min(pointing_times) if pointing_times else 0.0
        
        # Rates
        precision = tp_count / total_published if total_published > 0 else 0.0
        recall = gt_detected / gt_total if gt_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        throughput_targets_per_min = total_published / (total_run_time / 60) if total_run_time > 0 else 0.0
        
        return {
            "gt_total": gt_total,
            "gt_detected": gt_detected,
            "gt_missed": gt_missed,
            "total_published": total_published,
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "mean_error_mm": mean_error * 1000,
            "rmse_mm": rmse * 1000,
            "max_error_mm": max_error * 1000,
            "min_error_mm": min_error * 1000,
            "beam_success_on_tp": beam_success_tp,
            "beam_success_on_fp": beam_success_fp,
            "beam_failed": beam_failed,
            "beam_pending": beam_pending,
            "total_run_time_s": total_run_time,
            "avg_pointing_time_s": avg_pointing_time,
            "max_pointing_time_s": max_pointing_time,
            "min_pointing_time_s": min_pointing_time,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "throughput_per_min": throughput_targets_per_min,
            "beam_success_rate": (beam_success_tp + beam_success_fp) / total_published if total_published > 0 else 0.0,
        }

    def periodic_log(self):
        """Periodically log current metrics"""
        metrics = self.compute_metrics()
        
        self.get_logger().info(
            f"[METRICS] GT: {metrics['gt_detected']}/{metrics['gt_total']} | "
            f"TP:{metrics['tp_count']} FP:{metrics['fp_count']} FN:{metrics['fn_count']} | "
            f"Beams: ✅{metrics['beam_success_on_tp']}+{metrics['beam_success_on_fp']} ❌{metrics['beam_failed']} ⏳{metrics['beam_pending']} | "
            f"Error: {metrics['mean_error_mm']:.1f}mm (RMSE:{metrics['rmse_mm']:.1f}mm) | "
            f"Avg point time: {metrics['avg_pointing_time_s']:.2f}s"
        )

    # ========== CSV Output ==========
    
    def save_summary_csv(self):
        """Save summary metrics to CSV"""
        metrics = self.compute_metrics()
        
        with open(self.summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            
            w.writerow(["EVALUATION SUMMARY"])
            w.writerow(["timestamp", time.strftime("%Y-%m-%d %H:%M:%S")])
            w.writerow([])
            
            w.writerow(["GROUND TRUTH"])
            w.writerow(["gt_total", metrics["gt_total"]])
            w.writerow(["gt_detected", metrics["gt_detected"]])
            w.writerow(["gt_missed", metrics["gt_missed"]])
            w.writerow([])
            
            w.writerow(["DETECTION METRICS"])
            w.writerow(["total_published", metrics["total_published"]])
            w.writerow(["true_positives", metrics["tp_count"]])
            w.writerow(["false_positives", metrics["fp_count"]])
            w.writerow(["false_negatives", metrics["fn_count"]])
            w.writerow(["precision", f"{metrics['precision']:.4f}"])
            w.writerow(["recall", f"{metrics['recall']:.4f}"])
            w.writerow(["f1_score", f"{metrics['f1_score']:.4f}"])
            w.writerow([])
            
            w.writerow(["POSITION ACCURACY (mm)"])
            w.writerow(["mean_error_mm", f"{metrics['mean_error_mm']:.2f}"])
            w.writerow(["rmse_mm", f"{metrics['rmse_mm']:.2f}"])
            w.writerow(["max_error_mm", f"{metrics['max_error_mm']:.2f}"])
            w.writerow(["min_error_mm", f"{metrics['min_error_mm']:.2f}"])
            w.writerow([])
            
            w.writerow(["BEAM RESULTS"])
            w.writerow(["beam_success_on_tp", metrics["beam_success_on_tp"]])
            w.writerow(["beam_success_on_fp", metrics["beam_success_on_fp"]])
            w.writerow(["beam_failed", metrics["beam_failed"]])
            w.writerow(["beam_pending", metrics["beam_pending"]])
            w.writerow(["beam_success_rate", f"{metrics['beam_success_rate']:.4f}"])
            w.writerow([])
            
            w.writerow(["TIMING"])
            w.writerow(["total_run_time_s", f"{metrics['total_run_time_s']:.2f}"])
            w.writerow(["avg_pointing_time_s", f"{metrics['avg_pointing_time_s']:.2f}"])
            w.writerow(["max_pointing_time_s", f"{metrics['max_pointing_time_s']:.2f}"])
            w.writerow(["min_pointing_time_s", f"{metrics['min_pointing_time_s']:.2f}"])
            w.writerow(["throughput_per_min", f"{metrics['throughput_per_min']:.2f}"])
        
        self.get_logger().info(f"Saved summary to: {self.summary_csv}")

    def save_detailed_csv(self):
        """Save per-target detailed data for graphing"""
        with open(self.detailed_csv, "w", newline="") as f:
            w = csv.writer(f)
            
            w.writerow([
                "target_num",
                "publish_time_s",
                "world_x", "world_y", "world_z",
                "robot_x", "robot_y", "robot_z",
                "matched_gt_name",
                "gt_x", "gt_y", "gt_z",
                "position_error_mm",
                "error_dx_mm", "error_dy_mm", "error_dz_mm",
                "is_true_positive",
                "beam_status",
                "pointing_duration_s"
            ])
            
            for i, t in enumerate(self.published_targets, 1):
                gt_x = t.matched_gt_pos[0] if t.matched_gt_pos else ""
                gt_y = t.matched_gt_pos[1] if t.matched_gt_pos else ""
                gt_z = t.matched_gt_pos[2] if t.matched_gt_pos else ""
                error_mm = t.position_error_m * 1000 if t.position_error_m else ""
                
                error_dx_mm = ""
                error_dy_mm = ""
                error_dz_mm = ""
                if t.error_vector:
                    error_dx_mm = f"{t.error_vector[0] * 1000:.4f}"
                    error_dy_mm = f"{t.error_vector[1] * 1000:.4f}"
                    error_dz_mm = f"{t.error_vector[2] * 1000:.4f}"
                
                w.writerow([
                    i,
                    f"{t.publish_time - self.run_start_time:.3f}",
                    f"{t.world_pos[0]:.4f}", f"{t.world_pos[1]:.4f}", f"{t.world_pos[2]:.4f}",
                    f"{t.robot_pos[0]:.4f}", f"{t.robot_pos[1]:.4f}", f"{t.robot_pos[2]:.4f}",
                    t.matched_gt_name or "",
                    gt_x, gt_y, gt_z,
                    error_mm,
                    error_dx_mm, error_dy_mm, error_dz_mm,
                    "1" if t.matched_gt_name else "0",
                    t.status.value,
                    f"{t.pointing_duration_s:.3f}" if t.pointing_duration_s else ""
                ])
        
        self.get_logger().info(f"Saved detailed data to: {self.detailed_csv}")

    def save_gt_status_csv(self):
        """Save ground truth detection status"""
        import os
        detailed_basename = os.path.basename(self.detailed_csv).replace(".csv", "_gt_status.csv")
        gt_status_file = os.path.join(self.output_dir, detailed_basename)
        
        with open(gt_status_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gt_name", "x", "y", "z", "detected", "detection_time_s"])
            
            for name, gt in self.ground_truth.items():
                det_time = ""
                if gt.detection_time:
                    det_time = f"{gt.detection_time - self.run_start_time:.3f}"
                    
                w.writerow([
                    name,
                    f"{gt.world_pos[0]:.4f}",
                    f"{gt.world_pos[1]:.4f}",
                    f"{gt.world_pos[2]:.4f}",
                    "1" if gt.detected else "0",
                    det_time
                ])
        
        self.get_logger().info(f"Saved GT status to: {gt_status_file}")

    def save_all(self):
        """Save all output files"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("FINAL EVALUATION RESULTS")
        self.get_logger().info("=" * 60)
        
        metrics = self.compute_metrics()
        
        self.get_logger().info(f"Total run time: {metrics['total_run_time_s']:.1f} seconds")
        self.get_logger().info(f"Ground truth targets: {metrics['gt_total']}")
        self.get_logger().info(f"  - Detected: {metrics['gt_detected']} ({metrics['recall']*100:.1f}%)")
        self.get_logger().info(f"  - Missed: {metrics['gt_missed']}")
        self.get_logger().info(f"Published targets: {metrics['total_published']}")
        self.get_logger().info(f"  - True positives: {metrics['tp_count']}")
        self.get_logger().info(f"  - False positives: {metrics['fp_count']}")
        self.get_logger().info(f"Precision: {metrics['precision']*100:.1f}%")
        self.get_logger().info(f"Recall: {metrics['recall']*100:.1f}%")
        self.get_logger().info(f"F1 Score: {metrics['f1_score']:.3f}")
        self.get_logger().info(f"Position error (RMSE): {metrics['rmse_mm']:.1f} mm")
        self.get_logger().info(f"Beam success rate: {metrics['beam_success_rate']*100:.1f}%")
        self.get_logger().info(f"Average pointing time: {metrics['avg_pointing_time_s']:.2f} s")
        self.get_logger().info(f"Throughput: {metrics['throughput_per_min']:.1f} targets/min")
        self.get_logger().info("=" * 60)
        
        self.save_summary_csv()
        self.save_detailed_csv()
        self.save_gt_status_csv()


def main(args=None):
    rclpy.init(args=args)
    node = DetectionEvaluatorV3()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()