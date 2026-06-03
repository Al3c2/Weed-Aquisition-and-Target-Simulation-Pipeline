// Core ROS2
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

// MoveIt
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit/robot_state/robot_state.h>

// TF2 math & utils
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

// Angles
#include <angles/angles.h>

// Collision mesh helpers
#include <shape_msgs/msg/mesh.hpp>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shapes.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

class SimplePoseCommander : public rclcpp::Node
{
public:
  SimplePoseCommander()
  : Node("simple_pose_commander")
  {
    // --- Params you can override from launch/CLI ---
    swincar_mesh_uri_ = this->declare_parameter<std::string>(
        "swincar_mesh_uri",
        "file:///home/alex/.ignition/gazebo/models/swincar/meshes/swincar_collision.dae");

    // The car lives in 'world' in your launch
    world_frame_       = this->declare_parameter<std::string>("world_frame", "world");
    swincar_pose_frame_= this->declare_parameter<std::string>("swincar_pose_frame", "world");
    swincar_pose_xyz_  = this->declare_parameter<std::vector<double>>("swincar_pose_xyz", {0.0, 0.0, 0.0});
    swincar_pose_rpy_  = this->declare_parameter<std::vector<double>>("swincar_pose_rpy", {0.0, 0.0, 0.0});

    // Tool geometry
    beam_length_       = this->declare_parameter<double>("beam_length",   0.50);

    // Step control
    max_step_m_  = this->declare_parameter<double>("max_step_m",    0.05);
    eef_step_m_  = this->declare_parameter<double>("eef_step_m",    0.01);   // a tad coarser, more robust
    error_tol_m_ = this->declare_parameter<double>("error_tol_m",   0.01);
    max_iters_   = this->declare_parameter<int>("max_iters",        20);

    // Debug/toggles
    cart_retry_no_collisions_ = this->declare_parameter<bool>("cartesian_retry_no_collisions", true);
    enable_swincar_collision_ = this->declare_parameter<bool>("enable_swincar_collision", true);
    tf_timeout_s_             = this->declare_parameter<double>("tf_timeout_s", 0.8);

    // TF
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // MoveIt (PLAN IN WORLD to match your launch)
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node*){}),
      "ur3_manipulator");

    move_group_->setEndEffectorLink("tool0");
    move_group_->setPoseReferenceFrame(world_frame_);   // ✅ plan in world
    move_group_->setPlanningTime(8.0);
    move_group_->setPlannerId("RRTConnectkConfigDefault");
    move_group_->setMaxVelocityScalingFactor(0.25);
    move_group_->setMaxAccelerationScalingFactor(0.25);
    move_group_->setNumPlanningAttempts(20);
    move_group_->setGoalPositionTolerance(0.01);
    move_group_->setGoalOrientationTolerance(0.35);     // ~20°
    move_group_->startStateMonitor();

    if (!waitForFreshState(2.0))
      RCLCPP_WARN(this->get_logger(), "Current robot state not received in time. Planning may fail.");

    // Make sure TF world<->base_link exists (you publish it in launch)
    try {
      tf_buffer_->lookupTransform(world_frame_, "base_link", tf2::TimePointZero, tf2::durationFromSec(0.5));
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN(this->get_logger(), "No TF %s↔base_link yet: %s", world_frame_.c_str(), ex.what());
    }

    if (enable_swincar_collision_) addSwincarCollision();

    // Target subscription
    sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "target_pose", 10, std::bind(&SimplePoseCommander::poseCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "✅ SimplePoseCommander up (planning frame: %s). Publish /target_pose.",
                world_frame_.c_str());
  }

private:
  // ---- Members ----
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::string swincar_mesh_uri_;
  std::string world_frame_{"world"};
  std::string swincar_pose_frame_{"world"};
  std::vector<double> swincar_pose_xyz_;
  std::vector<double> swincar_pose_rpy_;

  double beam_length_{0.5};

  double max_step_m_{0.05};
  double eef_step_m_{0.01};
  double error_tol_m_{0.01};
  int    max_iters_{20};
  bool   cart_retry_no_collisions_{true};
  bool   enable_swincar_collision_{true};
  double tf_timeout_s_{0.8};

  // ---------------- helpers ----------------
  bool waitForFreshState(double timeout_seconds) const
  {
    auto st = move_group_->getCurrentState(timeout_seconds);
    return static_cast<bool>(st);
  }

  static tf2::Quaternion slerp(const tf2::Quaternion& a_in, const tf2::Quaternion& b_in, double t)
  {
    tf2::Quaternion a = a_in; a.normalize();
    tf2::Quaternion b = b_in; b.normalize();
    if (a.dot(b) < 0.0) b = tf2::Quaternion(-b.x(), -b.y(), -b.z(), -b.w());
    double dot = a.dot(b);
    if (dot > 0.9995) { tf2::Quaternion q = a*(1.0 - t) + b*t; q.normalize(); return q; }
    double theta0 = std::acos(dot);
    double theta  = theta0 * t;
    tf2::Quaternion c = (b - a*dot); c.normalize();
    tf2::Quaternion q = a*std::cos(theta) + c*std::sin(theta);
    q.normalize();
    return q;
  }

  // ---- Swincar collision anchored in WORLD ----
  void addSwincarCollision()
  {
    auto mesh = std::unique_ptr<shapes::Mesh>(shapes::createMeshFromResource(swincar_mesh_uri_));
    if (!mesh) {
      RCLCPP_WARN(this->get_logger(), "⚠️ Could not load mesh '%s' — no Swincar collision.",
                  swincar_mesh_uri_.c_str());
      return;
    }

    shapes::ShapeMsg shape_msg;
    shapes::constructMsgFromShape(mesh.get(), shape_msg);
    shape_msgs::msg::Mesh mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg);

    moveit_msgs::msg::CollisionObject obj;
    obj.id = "swincar";
    obj.operation = obj.ADD;

    // Pose in the car's frame (default 'world' from your launch)
    geometry_msgs::msg::PoseStamped car_pose;
    car_pose.header.frame_id = swincar_pose_frame_;
    car_pose.header.stamp    = this->now();
    car_pose.pose.position.x = swincar_pose_xyz_.size() > 0 ? swincar_pose_xyz_[0] : 0.0;
    car_pose.pose.position.y = swincar_pose_xyz_.size() > 1 ? swincar_pose_xyz_[1] : 0.0;
    car_pose.pose.position.z = swincar_pose_xyz_.size() > 2 ? swincar_pose_xyz_[2] : 0.0;

    double rr = swincar_pose_rpy_.size() > 0 ? swincar_pose_rpy_[0] : 0.0;
    double pp = swincar_pose_rpy_.size() > 1 ? swincar_pose_rpy_[1] : 0.0;
    double yy = swincar_pose_rpy_.size() > 2 ? swincar_pose_rpy_[2] : 0.0;
    tf2::Quaternion q; q.setRPY(rr, pp, yy); q.normalize();
    car_pose.pose.orientation = tf2::toMsg(q);

    // Keep it in WORLD (exactly like your sim)
    obj.header.frame_id = car_pose.header.frame_id;   // 'world'
    obj.header.stamp    = this->now();
    obj.meshes          = {mesh_msg};
    obj.mesh_poses      = {car_pose.pose};

    planning_scene_interface_.applyCollisionObjects({obj});
    rclcpp::sleep_for(std::chrono::milliseconds(300));
    RCLCPP_INFO(this->get_logger(), "🚗 Swincar collision added in frame '%s'.", obj.header.frame_id.c_str());
  }

  // Current beam tip in WORLD
  bool getCurrentBeamTip(tf2::Vector3& tip_out) const
  {
    try {
      auto tfStamped = tf_buffer_->lookupTransform(
          world_frame_, move_group_->getEndEffectorLink(),
          tf2::TimePointZero, tf2::durationFromSec(0.5));
      tf2::Transform tf; tf2::fromMsg(tfStamped.transform, tf);
      tf2::Vector3 p = tf.getOrigin();
      tf2::Matrix3x3 R = tf.getBasis();
      tf2::Vector3 tip = p + R * tf2::Vector3(0,0,beam_length_);
      tip_out = tip;                 // no extra Z hack — world↔base_link TF handles base height
      return true;
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN(this->get_logger(), "TF error getting beam tip: %s", ex.what());
      return false;
    }
  }

  // Orientation that points +Z to dir, but keeps current roll about +Z (reduces wrist spin)
  tf2::Quaternion alignZ_keepCurrentRoll(const tf2::Vector3& dir)
  {
    tf2::Vector3 up(0,0,1), z_axis = dir.normalized();
    if (std::fabs(z_axis.dot(up)) > 0.99) up = tf2::Vector3(1,0,0);
    tf2::Vector3 x_axis = up.cross(z_axis).normalized();
    tf2::Vector3 y_axis = z_axis.cross(x_axis).normalized();
    tf2::Matrix3x3 R_align; R_align[0]=x_axis; R_align[1]=y_axis; R_align[2]=z_axis;
    tf2::Quaternion q_align; R_align.getRotation(q_align); q_align.normalize();

    auto curr_pose = move_group_->getCurrentPose(move_group_->getEndEffectorLink()).pose;
    tf2::Quaternion q_curr; tf2::fromMsg(curr_pose.orientation, q_curr);

    tf2::Quaternion q_rel = q_align.inverse() * q_curr;
    double r,p,y; tf2::Matrix3x3(q_rel).getRPY(r,p,y);

    tf2::Quaternion q_z; q_z.setRPY(0,0,y); q_z.normalize();
    tf2::Quaternion q_goal = q_align * q_z; q_goal.normalize();
    return q_goal;
  }

  // Wrist pose whose +Z points at target and beam_tip hits target; preserves current roll
  geometry_msgs::msg::PoseStamped wristPoseForTarget(const tf2::Vector3& target_in)
  {
    // Use current wrist in WORLD so direction is local and stable
    const auto curr = move_group_->getCurrentPose(move_group_->getEndEffectorLink()).pose;
    tf2::Vector3 wrist_now(curr.position.x, curr.position.y, curr.position.z);

    tf2::Vector3 dir = (target_in - wrist_now).normalized();
    tf2::Quaternion q_primary = alignZ_keepCurrentRoll(dir);
    tf2::Quaternion q_spin180; q_spin180.setRPY(0,0,M_PI);
    tf2::Quaternion q = q_primary;

    tf2::Matrix3x3 R(q);
    tf2::Vector3 wrist_pos = target_in - R * tf2::Vector3(0,0,beam_length_);

    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = world_frame_;
    pose.pose.position.x = wrist_pos.x();
    pose.pose.position.y = wrist_pos.y();
    pose.pose.position.z = wrist_pos.z();
    pose.pose.orientation = tf2::toMsg(q);

    // Try 180° spin if IK doesn’t like primary
    if (!solvableIK(pose.pose)) {
      q = (q_primary * q_spin180).normalized();
      R = tf2::Matrix3x3(q);
      wrist_pos = target_in - R * tf2::Vector3(0,0,beam_length_);
      pose.pose.position.x = wrist_pos.x();
      pose.pose.position.y = wrist_pos.y();
      pose.pose.position.z = wrist_pos.z();
      pose.pose.orientation = tf2::toMsg(q);
    }
    return pose;
  }

  // Quick IK feasibility check (Humble-compatible)
  bool solvableIK(const geometry_msgs::msg::Pose& pose, double timeout = 0.25)
  {
    moveit::core::RobotModelConstPtr kmodel = move_group_->getRobotModel();
    moveit::core::RobotState kstate(kmodel);
    kstate.setToRandomPositions();
    kstate.update();
    const auto* jmg = kmodel->getJointModelGroup(move_group_->getName());
    return kstate.setFromIK(jmg, pose, move_group_->getEndEffectorLink(), timeout);
  }

  // IK -> joint-target fallback planner (no path constraints)
  bool planToPoseViaIK(const geometry_msgs::msg::PoseStamped& goal)
  {
    moveit::core::RobotModelConstPtr kmodel = move_group_->getRobotModel();
    moveit::core::RobotState rs(*move_group_->getCurrentState());
    const auto* jmg = kmodel->getJointModelGroup(move_group_->getName());

    bool ok = rs.setFromIK(jmg, goal.pose, move_group_->getEndEffectorLink(), 0.5);
    if (!ok) {
      RCLCPP_WARN(this->get_logger(), "IK failed for goal pose.");
      return false;
    }

    move_group_->setStartStateToCurrentState();
    move_group_->setJointValueTarget(rs);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto code = move_group_->plan(plan);
    RCLCPP_INFO(this->get_logger(), "IK→plan result: %d", code.val);
    if (code != moveit::core::MoveItErrorCode::SUCCESS) return false;

    return (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  }

  // Cartesian execution of a short hop to a pose (with optional no-collision retry)
  bool cartesianTo(const geometry_msgs::msg::PoseStamped& goal)
  {
    if (!waitForFreshState(1.0)) {
      RCLCPP_ERROR(this->get_logger(), "No fresh robot state — aborting cartesian hop.");
      return false;
    }

    move_group_->setStartStateToCurrentState();

    std::vector<geometry_msgs::msg::Pose> wps;
    const auto curr = move_group_->getCurrentPose(move_group_->getEndEffectorLink()).pose;

    tf2::Quaternion q0, q1;
    tf2::fromMsg(curr.orientation, q0);
    tf2::fromMsg(goal.pose.orientation, q1);

    for (int i=1;i<=3;++i) {
      double t = i/3.0;
      geometry_msgs::msg::Pose p;
      p.position.x = curr.position.x + t*(goal.pose.position.x - curr.position.x);
      p.position.y = curr.position.y + t*(goal.pose.position.y - curr.position.y);
      p.position.z = curr.position.z + t*(goal.pose.position.z - curr.position.z);
      p.orientation = tf2::toMsg(slerp(q0, q1, t));
      wps.push_back(p);
    }

    moveit_msgs::msg::RobotTrajectory traj;
    double fraction = move_group_->computeCartesianPath(
        wps, eef_step_m_, /*jump_threshold=*/0.0, traj, /*avoid_collisions=*/true);

    if (fraction < 0.95 && cart_retry_no_collisions_) {
      RCLCPP_WARN(this->get_logger(),
          "Cartesian fraction %.2f — retrying without collision checking (debug).", fraction);
      fraction = move_group_->computeCartesianPath(
          wps, eef_step_m_, 0.0, traj, /*avoid_collisions=*/false);
    }

    if (fraction >= 0.95) {
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      plan.trajectory_ = traj;
      return (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    }

    RCLCPP_WARN(this->get_logger(), "Cartesian fraction %.2f — using IK→joint fallback.", fraction);
    return planToPoseViaIK(goal);
  }

  // Move to a named pose if we're on the wrong side; otherwise keep current
  void ensureSideReady(const tf2::Vector3& target)
  {
    const std::string left = "ready_left";
    const std::string right = "ready_right";
    const std::string wanted = (target.y() >= 0.0) ? left : right;

    tf2::Vector3 tip;
    if (getCurrentBeamTip(tip)) {
      if ((target.y() >= 0.0 && tip.y() >= 0.0) ||
          (target.y() <  0.0 && tip.y() <  0.0))
      {
        return; // already on correct side
      }
    }

    if (!waitForFreshState(1.0)) {
      RCLCPP_WARN(this->get_logger(), "No fresh state for side switch; skipping.");
      return;
    }

    move_group_->clearPathConstraints();

    if (!move_group_->setNamedTarget(wanted)) {
      RCLCPP_WARN(this->get_logger(), "Named target '%s' not found in SRDF", wanted.c_str());
      return;
    }

    move_group_->setStartStateToCurrentState();
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto code = move_group_->plan(plan);
    RCLCPP_INFO(this->get_logger(), "Side switch plan result: %d", code.val);
    if (code == moveit::core::MoveItErrorCode::SUCCESS) {
      move_group_->execute(plan);
      RCLCPP_INFO(this->get_logger(), "➡️  Moved to '%s'", wanted.c_str());
    } else {
      RCLCPP_WARN(this->get_logger(), "Couldn't reach named pose '%s' — continuing without side switch", wanted.c_str());
    }
  }

  // ---- Main callback: tolerant TF to WORLD + robust fallbacks ----
  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    geometry_msgs::msg::PoseStamped target_w;

    // Already world?
    if (msg->header.frame_id.empty() || msg->header.frame_id == world_frame_) {
      target_w = *msg;
      target_w.header.frame_id = world_frame_;
    } else {
      // Try TF; if it fails, warn and proceed using raw pose as world (keeps commands flowing)
      try {
        target_w = tf_buffer_->transform(*msg, world_frame_, tf2::durationFromSec(tf_timeout_s_));
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(),
          "TF %s→%s failed (%.2fs). Using raw pose as %s: %s",
          msg->header.frame_id.c_str(), world_frame_.c_str(), tf_timeout_s_, world_frame_.c_str(), ex.what());
        target_w = *msg;
        target_w.header.frame_id = world_frame_;
      }
    }

    tf2::Vector3 target(target_w.pose.position.x,
                        target_w.pose.position.y,
                        target_w.pose.position.z);

    RCLCPP_INFO(this->get_logger(), "🎯 Target (%s): [%.3f, %.3f, %.3f]",
                world_frame_.c_str(), target.x(), target.y(), target.z());

    if (!waitForFreshState(1.0)) {
      RCLCPP_ERROR(this->get_logger(), "No fresh robot state — aborting this target.");
      return;
    }

    ensureSideReady(target);

    tf2::Vector3 tip;
    if (!getCurrentBeamTip(tip)) {
      auto goal = wristPoseForTarget(target);
      if (!cartesianTo(goal)) {
        RCLCPP_ERROR(this->get_logger(), "Failed single-hop approach.");
        return;
      }
      getCurrentBeamTip(tip);
    }

    int it = 0;
    while (it < max_iters_) {
      tf2::Vector3 delta = target - tip;
      double err = delta.length();
      if (err <= error_tol_m_) {
        RCLCPP_INFO(this->get_logger(), "✅ Done. Error %.3f m ≤ %.3f m", err, error_tol_m_);
        break;
      }

      tf2::Vector3 step_dir = delta.normalized();
      double step_len = std::min(err, max_step_m_);
      if (err < 2.0*beam_length_) step_len = std::min(step_len, 0.03); // gentler near goal
      tf2::Vector3 mid_target = tip + step_dir * step_len;

      auto goal = wristPoseForTarget(mid_target);
      if (!cartesianTo(goal)) {
        RCLCPP_ERROR(this->get_logger(), "❌ Cartesian step %d failed.", it+1);
        break;
      }

      if (!getCurrentBeamTip(tip)) {
        RCLCPP_WARN(this->get_logger(), "TF lost after step %d.", it+1);
        break;
      }
      ++it;
    }

    // Debug
    tf2::Vector3 final_tip;
    if (getCurrentBeamTip(final_tip)) {
      double final_err = (final_tip - target).length();
      RCLCPP_INFO(this->get_logger(),
        "📏 Beam tip now at [%.3f, %.3f, %.3f], error = %.3f m",
        final_tip.x(), final_tip.y(), final_tip.z(), final_err);
    }
    try {
      auto tfStamped = tf_buffer_->lookupTransform(
        world_frame_, move_group_->getEndEffectorLink(),
        tf2::TimePointZero, tf2::durationFromSec(0.5));
      tf2::Transform tf; tf2::fromMsg(tfStamped.transform, tf);
      double r,p,y; tf.getBasis().getRPY(r,p,y);
      RCLCPP_INFO(this->get_logger(), "🧭 EE RPY (world): roll=%.1f°, pitch=%.1f°, yaw=%.1f°",
                  angles::to_degrees(r), angles::to_degrees(p), angles::to_degrees(y));
    } catch (...) {}

    move_group_->clearPoseTargets();
    move_group_->clearPathConstraints();
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimplePoseCommander>());
  rclcpp::shutdown();
  return 0;
}
