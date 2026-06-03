#!/usr/bin/env python3

import csv
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry


class TargetStatus(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED  = "failed"


@dataclass
class PublishedTarget:
    publish_time:        float
    world_pos:           Tuple[float, float, float]
    robot_pos:           Tuple[float, float, float]
    robot_gt_pos:        Optional[Tuple[float, float, float]] = None  # Swincar GT position when published
    matched_gt_name:     Optional[str]                        = None
    matched_gt_pos:      Optional[Tuple[float, float, float]] = None
    position_error_yz_m: Optional[float]                      = None
    error_vector:        Optional[Tuple[float, float, float]] = None
    status:              TargetStatus                         = TargetStatus.PENDING
    beam_complete_time:  Optional[float]                      = None
    pointing_duration_s: Optional[float]                      = None
    is_retry:            bool                                  = False
    # [NEW-YZ] True beam tip YZ error measured from /beam_tip_yz vs GT
    beam_tip_yz_bl:      Optional[Tuple[float, float]]        = None  # (y, z) in base_link
    true_error_yz_m:     Optional[float]                      = None  # vs GT in base_link


@dataclass
class GroundTruth:
    name:             str
    world_pos:        Tuple[float, float, float]
    detected:         bool  = False
    detection_time:   Optional[float] = None
    beam_succeeded:   bool  = False
    beam_failed:      bool  = False
    attempt_count:    int   = 0


def dist_yz(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """YZ-only distance — X excluded because robot publishes while still moving forward."""
    return math.sqrt((a[1] - b[1])**2 + (a[2] - b[2])**2)


class DetectionEvaluatorV4(Node):

    def __init__(self):
        super().__init__("detection_evaluator_v4")

        self.declare_parameter("target_pose_topic",     "/target_pose")
        self.declare_parameter("beam_done_topic",       "/beam_task_done")
        self.declare_parameter("beam_failed_topic",     "/beam_task_failed")
        self.declare_parameter("use_ground_truth_pose", True)
        self.declare_parameter("ground_truth_topic",    "/model/swincar_ur3/pose")
        self.declare_parameter("ground_truth_frame_id", "empty")
        self.declare_parameter("odom_topic",            "/swincar/odom")
        self.declare_parameter("base_link_offset_x",    0.0)
        self.declare_parameter("base_link_offset_y",    0.2)
        self.declare_parameter("base_link_offset_z",    0.35)
        self.declare_parameter("gt_world_csv",          "gt_world.csv")
        self.declare_parameter("match_threshold_m",     0.010)
        self.declare_parameter("x_guard_m",             1.2)
        self.declare_parameter("output_dir",    "/home/alex/tese_ws/src/plots")
        self.declare_parameter("summary_csv",   "eval_summary.csv")
        self.declare_parameter("detailed_csv",  "eval_detailed_moving.csv")
        self.declare_parameter("log_interval_s",  5.0)
        self.declare_parameter("max_gt_targets",  0)

        self.target_pose_topic      = self.get_parameter("target_pose_topic").value
        self.beam_done_topic        = self.get_parameter("beam_done_topic").value
        self.beam_failed_topic      = self.get_parameter("beam_failed_topic").value
        self.use_ground_truth_pose  = self.get_parameter("use_ground_truth_pose").value
        self.ground_truth_topic     = self.get_parameter("ground_truth_topic").value
        self.ground_truth_frame_id  = self.get_parameter("ground_truth_frame_id").value
        self.odom_topic             = self.get_parameter("odom_topic").value
        self.base_link_offset_x     = self.get_parameter("base_link_offset_x").value
        self.base_link_offset_y     = self.get_parameter("base_link_offset_y").value
        self.base_link_offset_z     = self.get_parameter("base_link_offset_z").value
        self.gt_csv_path            = self.get_parameter("gt_world_csv").value
        self.match_threshold        = float(self.get_parameter("match_threshold_m").value)
        self.x_guard                = float(self.get_parameter("x_guard_m").value)
        self.output_dir             = self.get_parameter("output_dir").value
        summary_filename            = self.get_parameter("summary_csv").value
        detailed_filename           = self.get_parameter("detailed_csv").value

        import os
        os.makedirs(self.output_dir, exist_ok=True)
        self.summary_csv  = os.path.join(self.output_dir, summary_filename)
        self.detailed_csv = os.path.join(self.output_dir, detailed_filename)
        self.log_interval   = float(self.get_parameter("log_interval_s").value)
        self.max_gt_targets = int(self.get_parameter("max_gt_targets").value)

        self.robot_x = self.robot_y = self.robot_z = 0.0
        self.robot_qx = self.robot_qy = self.robot_qz = 0.0
        self.robot_qw = 1.0
        self.pose_initialized = False

        self.ground_truth: Dict[str, GroundTruth] = {}
        self._load_ground_truth()
        self.published_targets: List[PublishedTarget] = []
        self.current_pending: Optional[PublishedTarget] = None
        self.run_start_time = time.time()

        # [NEW-YZ] Latest beam tip YZ snapshot from Beamastar (/beam_tip_yz).
        # Published once per trajectory, right after execution completes.
        # Stored here and consumed in on_beam_done to compute true pointing error.
        self._latest_beam_tip_yz: Optional[Tuple[float, float]] = None

        self.create_subscription(Bool, self.beam_done_topic,   self.on_beam_done,   10)
        self.create_subscription(Bool, self.beam_failed_topic, self.on_beam_failed, 10)

        # [NEW-YZ] Subscribe to beam tip YZ snapshot
        self.create_subscription(PointStamped, '/beam_tip_yz', self._on_beam_tip_yz, 10)

        self._latest_world_pos: Optional[Tuple[float, float, float]] = None
        self.create_subscription(PointStamped, '/target_world_pos',
                                 self.on_target_world_pos_primary, 10)

        self._latest_bl: Optional[Tuple[float, float, float]] = None
        self.create_subscription(PoseStamped, self.target_pose_topic, self._on_arm_pose_log, 10)

        if self.use_ground_truth_pose:
            self.create_subscription(PoseStamped, self.ground_truth_topic,
                                     self.on_ground_truth_pose, 10)
        else:
            self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)

        self.create_timer(self.log_interval, self.periodic_log)
        self._print_startup()

    def _print_startup(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("Detection Evaluator v4 — exclusive YZ+X matching + true beam YZ")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Ground truth: {len(self.ground_truth)} targets from {self.gt_csv_path}")
        self.get_logger().info(f"Pose source:  {self.ground_truth_topic}")
        self.get_logger().info(f"Match threshold (YZ): {self.match_threshold * 1000:.1f} mm")
        self.get_logger().info(f"X-guard window:       ±{self.x_guard:.2f} m")
        self.get_logger().info(f"True beam YZ:         /beam_tip_yz (base_link)")
        self.get_logger().info(f"Output dir:   {self.output_dir}")
        self.get_logger().info("=" * 60)

    def _load_ground_truth(self):
        try:
            with open(self.gt_csv_path, "r") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    if self.max_gt_targets > 0 and count >= self.max_gt_targets:
                        break
                    name = row["name"]
                    pos  = (float(row["x"]), float(row["y"]), float(row["z"]))
                    self.ground_truth[name] = GroundTruth(name=name, world_pos=pos)
                    count += 1
            self.get_logger().info(f"Loaded {len(self.ground_truth)} GT targets")
        except Exception as e:
            self.get_logger().error(f"Failed to load GT CSV: {e}")
            raise

    # ── Pose callbacks ────────────────────────────────────────────────────

    def on_ground_truth_pose(self, msg: PoseStamped):
        if msg.header.frame_id != self.ground_truth_frame_id:
            return
        self.robot_x  = msg.pose.position.x
        self.robot_y  = msg.pose.position.y
        self.robot_z  = msg.pose.position.z
        self.robot_qx = msg.pose.orientation.x
        self.robot_qy = msg.pose.orientation.y
        self.robot_qz = msg.pose.orientation.z
        self.robot_qw = msg.pose.orientation.w
        if not self.pose_initialized:
            self.pose_initialized = True
            rot = Rotation.from_quat([self.robot_qx, self.robot_qy,
                                      self.robot_qz, self.robot_qw])
            roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
            self.get_logger().info(
                f"Robot pose init: [{self.robot_x:.3f}, {self.robot_y:.3f}, "
                f"{self.robot_z:.3f}] roll={roll:.1f}° pitch={pitch:.1f}° yaw={yaw:.1f}°")

    def on_odom(self, msg: Odometry):
        self.robot_x  = msg.pose.pose.position.x
        self.robot_y  = msg.pose.pose.position.y
        self.robot_z  = msg.pose.pose.position.z
        self.robot_qx = msg.pose.pose.orientation.x
        self.robot_qy = msg.pose.pose.orientation.y
        self.robot_qz = msg.pose.pose.orientation.z
        self.robot_qw = msg.pose.pose.orientation.w
        if not self.pose_initialized:
            self.pose_initialized = True

    # ── Transform helpers ─────────────────────────────────────────────────

    def _base_link_to_world(self, x_bl, y_bl, z_bl) -> Tuple[float, float, float]:
        rot    = Rotation.from_quat([self.robot_qx, self.robot_qy,
                                     self.robot_qz, self.robot_qw])
        offset = np.array([self.base_link_offset_x,
                           self.base_link_offset_y,
                           self.base_link_offset_z])
        world  = np.array([self.robot_x, self.robot_y, self.robot_z]) \
                 + rot.apply(offset + np.array([x_bl, y_bl, z_bl]))
        return tuple(world)

    def _world_to_base_link(self, wx, wy, wz) -> Tuple[float, float, float]:
        """Inverse of _base_link_to_world — used to bring GT into base_link for true error."""
        rot    = Rotation.from_quat([self.robot_qx, self.robot_qy,
                                     self.robot_qz, self.robot_qw])
        offset = np.array([self.base_link_offset_x,
                           self.base_link_offset_y,
                           self.base_link_offset_z])
        bl = rot.inv().apply(np.array([wx, wy, wz])
                             - np.array([self.robot_x, self.robot_y, self.robot_z])) - offset
        return tuple(bl)

    # ── [NEW-YZ] Beam tip YZ callback ─────────────────────────────────────

    def _on_beam_tip_yz(self, msg: PointStamped):
        """
        Store latest beam tip YZ snapshot (base_link frame, x=0 always).
        Published by Beamastar once per trajectory, right after execution+dwell.
        Consumed in on_beam_done to compute true pointing error vs GT.
        """
        self._latest_beam_tip_yz = (msg.point.y, msg.point.z)

    # ── Target pose callbacks ─────────────────────────────────────────────

    def _on_arm_pose_log(self, msg: PoseStamped):
        self._latest_bl = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def on_target_world_pos_primary(self, msg: PointStamped):
        if not self.pose_initialized:
            self.get_logger().warn("target_world_pos received but robot pose not yet initialised")
            return

        now = time.time()
        world_pos = (msg.point.x, msg.point.y, msg.point.z)
        bl = self._latest_bl or (0.0, 0.0, 0.0)
        self._latest_bl = None

        matched_name, matched_pos, error_yz, error_vector = self._match_to_gt(world_pos)

        is_retry = (
            matched_name is not None
            and len(self.published_targets) > 0
            and self.published_targets[-1].matched_gt_name == matched_name
        )

        target = PublishedTarget(
            publish_time        = now,
            world_pos           = world_pos,
            robot_pos           = (self.robot_x, self.robot_y, self.robot_z),
            robot_gt_pos        = (self.robot_x, self.robot_y, self.robot_z),  # Capture swincar GT pose
            matched_gt_name     = matched_name,
            matched_gt_pos      = matched_pos,
            position_error_yz_m = error_yz,
            error_vector        = error_vector,
            status              = TargetStatus.PENDING,
            is_retry            = is_retry,
        )
        self.published_targets.append(target)
        self.current_pending = target

        if matched_name:
            gt = self.ground_truth[matched_name]
            gt.attempt_count += 1
            if not gt.detected:
                gt.detected       = True
                gt.detection_time = now
            retry_tag = " [RETRY]" if is_retry else ""
            self.get_logger().info(
                f"📍 TARGET #{len(self.published_targets)}{retry_tag}: "
                f"world=[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}] "
                f"bl=[{bl[0]:.3f}, {bl[1]:.3f}, {bl[2]:.3f}] "
                f"→ GT '{matched_name}' (YZ error: {error_yz*1000:.1f} mm)"
            )
        else:
            self.get_logger().warn(
                f"⚠️ TARGET #{len(self.published_targets)}: "
                f"world=[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}] "
                f"→ NO GT MATCH (FP)"
            )

    # ── Matching ──────────────────────────────────────────────────────────

    def _match_to_gt(self, world_pos: Tuple[float, float, float]):
        RETRY_POSITION_THRESHOLD = 0.05

        def _best_in(candidates):
            best_name = best_pos = best_vector = None
            best_dist = float("inf")
            for name, gt in candidates:
                if abs(world_pos[0] - gt.world_pos[0]) > self.x_guard:
                    continue
                d = dist_yz(world_pos, gt.world_pos)
                if d < best_dist:
                    best_dist = d
                    best_name = name
                    best_pos  = gt.world_pos
                    best_vector = (
                        world_pos[0] - gt.world_pos[0],
                        world_pos[1] - gt.world_pos[1],
                        world_pos[2] - gt.world_pos[2],
                    )
            return best_name, best_pos, best_dist, best_vector

        unbeamed = [(n, g) for n, g in self.ground_truth.items() if not g.beam_succeeded]
        name, pos, dist, vec = _best_in(unbeamed)
        if name and dist <= self.match_threshold:
            return (name, pos, dist, vec)

        if self.published_targets:
            prev_pos = self.published_targets[-1].world_pos
            dist_to_prev = math.sqrt(
                (world_pos[0] - prev_pos[0])**2 +
                (world_pos[1] - prev_pos[1])**2 +
                (world_pos[2] - prev_pos[2])**2
            )
            if dist_to_prev <= RETRY_POSITION_THRESHOLD:
                name, pos, dist, vec = _best_in(self.ground_truth.items())
                if name and dist <= self.match_threshold:
                    return (name, pos, dist, vec)

        return (None, None, None, None)

    # ── Beam callbacks ────────────────────────────────────────────────────

    def on_beam_done(self, msg: Bool):
        if not msg.data:
            return
        if self.current_pending is None:
            self.get_logger().warn("Beam done but no pending target")
            return
        now = time.time()
        self.current_pending.status             = TargetStatus.SUCCESS
        self.current_pending.beam_complete_time  = now
        self.current_pending.pointing_duration_s = now - self.current_pending.publish_time

        matched = self.current_pending.matched_gt_name
        if matched and matched in self.ground_truth:
            self.ground_truth[matched].beam_succeeded = True

        # [NEW-YZ] Compute true beam tip YZ error vs GT in base_link frame.
        # _latest_beam_tip_yz was published by Beamastar right after execution,
        # so it reflects the arm's actual resting position before the hold sweep.
        true_err_str = ""
        if self._latest_beam_tip_yz is not None and matched is not None:
            tip_y, tip_z = self._latest_beam_tip_yz
            self.current_pending.beam_tip_yz_bl = (tip_y, tip_z)

            # Convert matched GT world pos → base_link at current robot pose.
            # The robot has moved since dispatch, so use current pose for the
            # conversion — we want the GT position in the arm's current frame.
            gt_pos = self.ground_truth[matched].world_pos
            gt_bl  = self._world_to_base_link(gt_pos[0], gt_pos[1], gt_pos[2])
            gt_y, gt_z = gt_bl[1], gt_bl[2]

            true_err = math.sqrt((tip_y - gt_y)**2 + (tip_z - gt_z)**2)
            self.current_pending.true_error_yz_m = true_err
            true_err_str = f" | true_yz_err={true_err*1000:.1f}mm (tip_bl=[{tip_y:.3f},{tip_z:.3f}] gt_bl=[{gt_y:.3f},{gt_z:.3f}])"

            # Consume so next target doesn't accidentally reuse this snapshot
            self._latest_beam_tip_yz = None

        self.get_logger().info(
            f"✅ BEAM SUCCESS: '{matched or 'FP'}' "
            f"(pointing time: {self.current_pending.pointing_duration_s:.2f}s)"
            f"{true_err_str}"
        )
        self.current_pending = None

    def on_beam_failed(self, msg: Bool):
        if not msg.data:
            return
        if self.current_pending is None:
            self.get_logger().warn("Beam failed but no pending target")
            return
        now = time.time()
        self.current_pending.status             = TargetStatus.FAILED
        self.current_pending.beam_complete_time  = now
        self.current_pending.pointing_duration_s = now - self.current_pending.publish_time

        matched = self.current_pending.matched_gt_name
        if matched and matched in self.ground_truth:
            self.ground_truth[matched].beam_failed = True

        # Consume beam_tip_yz on failure too (don't carry stale value to next target)
        self._latest_beam_tip_yz = None

        self.get_logger().warn(
            f"❌ BEAM FAILED: '{matched or 'FP'}' "
            f"(pointing time: {self.current_pending.pointing_duration_s:.2f}s)"
        )
        self.current_pending = None

    # ── Metrics ───────────────────────────────────────────────────────────

    def compute_metrics(self) -> dict:
        total_published = len(self.published_targets)
        gt_total   = len(self.ground_truth)
        gt_detected = sum(1 for gt in self.ground_truth.values() if gt.detected)
        gt_beamed   = sum(1 for gt in self.ground_truth.values() if gt.beam_succeeded)
        gt_missed   = gt_total - gt_detected

        seen_gt = set()
        tp_unique  = 0
        tp_retries = 0
        fp_targets = []
        for t in self.published_targets:
            if t.matched_gt_name is None:
                fp_targets.append(t)
            elif t.matched_gt_name not in seen_gt:
                seen_gt.add(t.matched_gt_name)
                tp_unique += 1
            else:
                tp_retries += 1

        fp_count = len(fp_targets)
        fn_count = gt_total - gt_detected

        # Detection YZ errors (from first attempts, based on detector world position)
        first_tp = [t for t in self.published_targets
                    if t.matched_gt_name is not None and not t.is_retry]
        errors_yz = [t.position_error_yz_m for t in first_tp
                     if t.position_error_yz_m is not None]
        mean_error_yz = sum(errors_yz) / len(errors_yz) if errors_yz else 0.0
        rmse_yz = math.sqrt(sum(e**2 for e in errors_yz) / len(errors_yz)) if errors_yz else 0.0
        max_error_yz = max(errors_yz) if errors_yz else 0.0

        # [NEW-YZ] True beam tip YZ errors (from /beam_tip_yz vs GT in base_link)
        true_errors = [t.true_error_yz_m for t in self.published_targets
                       if t.true_error_yz_m is not None]
        true_mean_yz = sum(true_errors) / len(true_errors) if true_errors else 0.0
        true_rmse_yz = math.sqrt(sum(e**2 for e in true_errors) / len(true_errors)) if true_errors else 0.0
        true_max_yz  = max(true_errors) if true_errors else 0.0

        beam_success = sum(1 for t in self.published_targets if t.status == TargetStatus.SUCCESS)
        beam_failed  = sum(1 for t in self.published_targets if t.status == TargetStatus.FAILED)
        beam_pending = sum(1 for t in self.published_targets if t.status == TargetStatus.PENDING)

        total_run_time    = time.time() - self.run_start_time
        pointing_times    = [t.pointing_duration_s for t in self.published_targets
                             if t.pointing_duration_s is not None]
        avg_pointing_time = sum(pointing_times) / len(pointing_times) if pointing_times else 0.0
        max_pointing_time = max(pointing_times) if pointing_times else 0.0

        precision  = tp_unique / total_published if total_published > 0 else 0.0
        recall     = gt_beamed / gt_total        if gt_total > 0        else 0.0
        f1         = (2*precision*recall / (precision+recall)
                      if (precision+recall) > 0 else 0.0)
        throughput = total_published / (total_run_time / 60) if total_run_time > 0 else 0.0

        return {
            "gt_total": gt_total, "gt_detected": gt_detected,
            "gt_beamed": gt_beamed, "gt_missed": gt_missed,
            "total_published": total_published,
            "tp_unique": tp_unique, "tp_retries": tp_retries,
            "fp_count": fp_count, "fn_count": fn_count,
            # Detection error (detector world pos vs GT)
            "mean_error_yz_mm": mean_error_yz * 1000,
            "rmse_yz_mm":       rmse_yz       * 1000,
            "max_error_yz_mm":  max_error_yz  * 1000,
            # True beam tip error (Beamastar /beam_tip_yz vs GT in base_link)
            "true_mean_yz_mm":  true_mean_yz * 1000,
            "true_rmse_yz_mm":  true_rmse_yz * 1000,
            "true_max_yz_mm":   true_max_yz  * 1000,
            "true_sample_count": len(true_errors),
            "beam_success": beam_success, "beam_failed": beam_failed,
            "beam_pending": beam_pending,
            "total_run_time_s": total_run_time,
            "avg_pointing_time_s": avg_pointing_time,
            "max_pointing_time_s": max_pointing_time,
            "precision": precision, "recall": recall, "f1_score": f1,
            "throughput_per_min": throughput,
            "beam_success_rate": beam_success / total_published if total_published > 0 else 0.0,
        }

    def periodic_log(self):
        m = self.compute_metrics()
        self.get_logger().info(
            f"[METRICS] GT: {m['gt_beamed']}/{m['gt_total']} beamed "
            f"({m['gt_detected']} attempted) | "
            f"TP:{m['tp_unique']}(+{m['tp_retries']} retries) FP:{m['fp_count']} FN:{m['fn_count']} | "
            f"Beams: ✅{m['beam_success']} ❌{m['beam_failed']} ⏳{m['beam_pending']} | "
            f"YZ err: {m['mean_error_yz_mm']:.1f}mm (RMSE:{m['rmse_yz_mm']:.1f}mm) | "
            f"True beam YZ: {m['true_mean_yz_mm']:.1f}mm (RMSE:{m['true_rmse_yz_mm']:.1f}mm, n={m['true_sample_count']}) | "
            f"Avg point: {m['avg_pointing_time_s']:.2f}s"
        )

    def save_summary_csv(self):
        m = self.compute_metrics()
        with open(self.summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["EVALUATION SUMMARY — v4"])
            w.writerow(["timestamp", time.strftime("%Y-%m-%d %H:%M:%S")])
            w.writerow(["note", "YZ-only errors; X excluded (robot moves in X during arm execution)"])
            w.writerow([])
            w.writerow(["GROUND TRUTH"])
            w.writerow(["gt_total",     m["gt_total"]])
            w.writerow(["gt_attempted", m["gt_detected"]])
            w.writerow(["gt_beamed",    m["gt_beamed"]])
            w.writerow(["gt_missed",    m["gt_missed"]])
            w.writerow([])
            w.writerow(["DETECTION METRICS"])
            w.writerow(["total_published",  m["total_published"]])
            w.writerow(["tp_unique",        m["tp_unique"]])
            w.writerow(["tp_retries",       m["tp_retries"]])
            w.writerow(["false_positives",  m["fp_count"]])
            w.writerow(["false_negatives",  m["fn_count"]])
            w.writerow(["precision",        f"{m['precision']:.4f}"])
            w.writerow(["recall",           f"{m['recall']:.4f}"])
            w.writerow(["f1_score",         f"{m['f1_score']:.4f}"])
            w.writerow([])
            w.writerow(["DETECTION POSITION ACCURACY — YZ only (mm), detector world pos vs GT"])
            w.writerow(["mean_error_yz_mm", f"{m['mean_error_yz_mm']:.2f}"])
            w.writerow(["rmse_yz_mm",       f"{m['rmse_yz_mm']:.2f}"])
            w.writerow(["max_error_yz_mm",  f"{m['max_error_yz_mm']:.2f}"])
            w.writerow([])
            w.writerow(["TRUE BEAM POINTING ACCURACY — YZ only (mm), /beam_tip_yz vs GT in base_link"])
            w.writerow(["true_mean_yz_mm",  f"{m['true_mean_yz_mm']:.2f}"])
            w.writerow(["true_rmse_yz_mm",  f"{m['true_rmse_yz_mm']:.2f}"])
            w.writerow(["true_max_yz_mm",   f"{m['true_max_yz_mm']:.2f}"])
            w.writerow(["true_sample_count", m["true_sample_count"]])
            w.writerow([])
            w.writerow(["BEAM RESULTS"])
            w.writerow(["beam_success",      m["beam_success"]])
            w.writerow(["beam_failed",       m["beam_failed"]])
            w.writerow(["beam_pending",      m["beam_pending"]])
            w.writerow(["beam_success_rate", f"{m['beam_success_rate']:.4f}"])
            w.writerow([])
            w.writerow(["TIMING"])
            w.writerow(["total_run_time_s",    f"{m['total_run_time_s']:.2f}"])
            w.writerow(["avg_pointing_time_s", f"{m['avg_pointing_time_s']:.2f}"])
            w.writerow(["throughput_per_min",  f"{m['throughput_per_min']:.2f}"])
        self.get_logger().info(f"Saved summary → {self.summary_csv}")

    def save_detailed_csv(self):
        with open(self.detailed_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "target_num", "publish_time_s",
                "world_x", "world_y", "world_z",
                "robot_x", "robot_y", "robot_z",
                "swincar_gt_x", "swincar_gt_y", "swincar_gt_z",
                "matched_gt_name", "gt_x", "gt_y", "gt_z",
                "detect_error_yz_mm",
                "error_dx_mm", "error_dy_mm", "error_dz_mm",
                "beam_tip_y_bl", "beam_tip_z_bl",
                "true_error_yz_mm",      # [NEW-YZ] true arm pointing error
                "is_true_positive", "is_retry", "beam_status", "pointing_duration_s",
            ])
            for i, t in enumerate(self.published_targets, 1):
                gt_x   = t.matched_gt_pos[0] if t.matched_gt_pos else ""
                gt_y   = t.matched_gt_pos[1] if t.matched_gt_pos else ""
                gt_z   = t.matched_gt_pos[2] if t.matched_gt_pos else ""
                swincar_gt_x = f"{t.robot_gt_pos[0]:.4f}" if t.robot_gt_pos else ""
                swincar_gt_y = f"{t.robot_gt_pos[1]:.4f}" if t.robot_gt_pos else ""
                swincar_gt_z = f"{t.robot_gt_pos[2]:.4f}" if t.robot_gt_pos else ""
                err_yz = f"{t.position_error_yz_m*1000:.4f}" if t.position_error_yz_m is not None else ""
                dx = f"{t.error_vector[0]*1000:.4f}" if t.error_vector else ""
                dy = f"{t.error_vector[1]*1000:.4f}" if t.error_vector else ""
                dz = f"{t.error_vector[2]*1000:.4f}" if t.error_vector else ""
                tip_y  = f"{t.beam_tip_yz_bl[0]:.4f}" if t.beam_tip_yz_bl else ""
                tip_z  = f"{t.beam_tip_yz_bl[1]:.4f}" if t.beam_tip_yz_bl else ""
                true_e = f"{t.true_error_yz_m*1000:.4f}" if t.true_error_yz_m is not None else ""
                w.writerow([
                    i, f"{t.publish_time - self.run_start_time:.3f}",
                    f"{t.world_pos[0]:.4f}", f"{t.world_pos[1]:.4f}", f"{t.world_pos[2]:.4f}",
                    f"{t.robot_pos[0]:.4f}", f"{t.robot_pos[1]:.4f}", f"{t.robot_pos[2]:.4f}",
                    swincar_gt_x, swincar_gt_y, swincar_gt_z,
                    t.matched_gt_name or "", gt_x, gt_y, gt_z,
                    err_yz, dx, dy, dz,
                    tip_y, tip_z, true_e,
                    "1" if t.matched_gt_name else "0",
                    "1" if t.is_retry else "0",
                    t.status.value,
                    f"{t.pointing_duration_s:.3f}" if t.pointing_duration_s else "",
                ])
        self.get_logger().info(f"Saved detailed → {self.detailed_csv}")

    def save_gt_status_csv(self):
        import os
        gt_status_file = os.path.join(
            self.output_dir,
            os.path.basename(self.detailed_csv).replace(".csv", "_gt_status.csv"))
        with open(gt_status_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gt_name", "x", "y", "z",
                        "attempted", "beam_succeeded", "beam_failed",
                        "attempt_count", "detection_time_s"])
            for name, gt in self.ground_truth.items():
                det_time = (f"{gt.detection_time - self.run_start_time:.3f}"
                            if gt.detection_time else "")
                w.writerow([
                    name,
                    f"{gt.world_pos[0]:.4f}", f"{gt.world_pos[1]:.4f}", f"{gt.world_pos[2]:.4f}",
                    "1" if gt.detected else "0",
                    "1" if gt.beam_succeeded else "0",
                    "1" if gt.beam_failed else "0",
                    gt.attempt_count, det_time,
                ])
        self.get_logger().info(f"Saved GT status → {gt_status_file}")

    def save_all(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("FINAL EVALUATION RESULTS")
        self.get_logger().info("=" * 60)
        m = self.compute_metrics()
        self.get_logger().info(f"Run time:                {m['total_run_time_s']:.1f}s")
        self.get_logger().info(f"GT targets:              {m['gt_total']}")
        self.get_logger().info(f"  Attempted:             {m['gt_detected']}")
        self.get_logger().info(f"  Successfully beamed:   {m['gt_beamed']} ({m['recall']*100:.1f}%)")
        self.get_logger().info(f"  Never attempted:       {m['gt_missed']}")
        self.get_logger().info(f"Total publishes:         {m['total_published']}")
        self.get_logger().info(f"  Unique TP:             {m['tp_unique']}")
        self.get_logger().info(f"  Retries:               {m['tp_retries']}")
        self.get_logger().info(f"  FP:                    {m['fp_count']}")
        self.get_logger().info(f"Precision:               {m['precision']*100:.1f}%")
        self.get_logger().info(f"Recall (beamed/gt):      {m['recall']*100:.1f}%")
        self.get_logger().info(f"F1:                      {m['f1_score']:.3f}")
        self.get_logger().info(f"Detection YZ error RMSE: {m['rmse_yz_mm']:.1f} mm")
        self.get_logger().info(f"True beam YZ error RMSE: {m['true_rmse_yz_mm']:.1f} mm  (n={m['true_sample_count']})")
        self.get_logger().info(f"Beam success rate:       {m['beam_success_rate']*100:.1f}%")
        self.get_logger().info(f"Avg pointing time:       {m['avg_pointing_time_s']:.2f}s")
        self.get_logger().info(f"Throughput:              {m['throughput_per_min']:.1f} targets/min")
        self.get_logger().info("=" * 60)
        self.save_summary_csv()
        self.save_detailed_csv()
        self.save_gt_status_csv()


def main(args=None):
    rclpy.init(args=args)
    node = DetectionEvaluatorV4()
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