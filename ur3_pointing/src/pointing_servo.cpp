// UR3 pointing servo (ROS 2 Humble)
// Publishes tiny steps to joint_trajectory_controller.
// Uses MoveIt IK; prefers collision-checked IK if PlanningScene is up.
// Adds Swincar collision mesh from a fixed URI.

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <moveit/kinematics_base/kinematics_base.h>  // KinematicsQueryOptions

#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <Eigen/Geometry>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <boost/variant/get.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using std::placeholders::_1;
using namespace std::chrono_literals;

class PointingServo : public rclcpp::Node
{
public:
  PointingServo()
  : Node("pointing_servo"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // --- Parameters ---
    world_frame_  = this->declare_parameter<std::string>("world_frame", "world");
    base_link_    = this->declare_parameter<std::string>("base_link", "base_link");
    ee_link_      = this->declare_parameter<std::string>("ee_link", "tool0");
    group_name_   = this->declare_parameter<std::string>("move_group", "ur3_manipulator");

    // Default to +Z; we try ±axis in code anyway
    tool_axis_    = this->declare_parameter<std::vector<double>>("tool_axis_xyz", {0.0, 0.0, 1.0});
    if (tool_axis_.size() != 3) tool_axis_ = {0.0, 0.0, 1.0};

    beam_length_  = this->declare_parameter<double>("beam_length", 0.50);
    err_tol_m_    = this->declare_parameter<double>("error_tol_m", 0.01);
    rate_hz_      = this->declare_parameter<double>("rate_hz", 100.0);

    max_joint_step_ = this->declare_parameter<double>("max_joint_step", 0.02);
    min_time_step_  = this->declare_parameter<double>("min_time_step", 0.02);
    coll_check_every_ = this->declare_parameter<int>("coll_check_every", 5); // every N ticks do collision IK

    joint_names_ = this->declare_parameter<std::vector<std::string>>(
      "joint_names",
      {"shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
       "wrist_1_joint","wrist_2_joint","wrist_3_joint"});

    // Swincar collision mesh toggle + pose
    swc_enabled_  = this->declare_parameter<bool>("swincar.enabled", false);
    swc_pose_xyz_ = this->declare_parameter<std::vector<double>>("swincar.pose_xyz", {0.0, 0.0, 0.45});
    swc_pose_rpy_ = this->declare_parameter<std::vector<double>>("swincar.pose_rpy", {0.0, 0.0, 0.0});

    // --- IO ---
    traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/joint_trajectory_controller/joint_trajectory", 10);

    target_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "target_pose", 10, std::bind(&PointingServo::targetCb, this, _1));

    // Control loop
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(1.0, rate_hz_)),
      std::bind(&PointingServo::step, this));

    // Bring-up timer (retry until PlanningScene is ready)
    init_timer_ = this->create_wall_timer(250ms, std::bind(&PointingServo::tryInitMoveIt, this));

    // Periodic “I’m alive but waiting” log (1 Hz)
    heartbeat_timer_ = this->create_wall_timer(1s, [this]{
      if (!moveit_ready_) {
        RCLCPP_INFO(get_logger(), "⏳ Waiting for PlanningScene... (have model=%s)", kmodel_ ? "yes":"no");
      } else if (!psm_ready_) {
        RCLCPP_INFO(get_logger(), "⏳ PlanningSceneMonitor started, waiting for first state...");
      }
    });

    RCLCPP_INFO(this->get_logger(),
      "✅ PointingServo up. frame=%s base=%s ee=%s group=%s axis=[%.2f %.2f %.2f] beam=%.2fm",
      world_frame_.c_str(), base_link_.c_str(), ee_link_.c_str(), group_name_.c_str(),
      tool_axis_[0], tool_axis_[1], tool_axis_[2], beam_length_);
  }

private:
  // ---------- Utility: FK + quaternion error ----------
  static bool fkPose(moveit::core::RobotState& st,   // <-- non-const
                     const std::string& link,
                     tf2::Vector3& p, tf2::Quaternion& q)
  {
    st.update();  // ensure transforms are fresh
    const auto* tip = st.getLinkModel(link);
    if (!tip) return false;
    const Eigen::Isometry3d& T = st.getGlobalLinkTransform(tip);
    p = tf2::Vector3(T.translation().x(), T.translation().y(), T.translation().z());
    Eigen::Quaterniond qe(T.rotation());
    q = tf2::Quaternion(qe.x(), qe.y(), qe.z(), qe.w()); q.normalize();
    return true;
  }

  static double quatAngularErrorRad(const tf2::Quaternion& qa, const tf2::Quaternion& qb)
  {
    tf2::Quaternion dq = qa * qb.inverse(); dq.normalize();
    const double s = std::sqrt(dq.x()*dq.x()+dq.y()*dq.y()+dq.z()*dq.z());
    double angle = 2.0 * std::atan2(s, std::abs(dq.w()));
    if (angle > M_PI) angle = 2*M_PI - angle;
    return angle;
  }

  // ---------- MoveIt bring-up ----------
  void tryInitMoveIt()
  {
    // Build model once
    if (!kmodel_) {
      try {
        robot_model_loader::RobotModelLoader::Options opts("robot_description");
        robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(this->shared_from_this(), opts);
        kmodel_ = robot_model_loader_->getModel();
        if (!kmodel_) {
          RCLCPP_WARN(get_logger(), "MoveIt: robot model not yet available.");
          return;
        }
        jmg_ = kmodel_->getJointModelGroup(group_name_);
        if (!jmg_) {
          RCLCPP_ERROR(get_logger(), "MoveIt: group '%s' not found.", group_name_.c_str());
          return;
        }
        // Tip-link sanity
        const auto* tip = jmg_->getOnlyOneEndEffectorTip();
        if (tip && tip->getName() != ee_link_) {
          RCLCPP_WARN(get_logger(), "SRDF tip is '%s' but ee_link param is '%s' — using '%s' for IK.",
                      tip->getName().c_str(), ee_link_.c_str(), tip->getName().c_str());
          ee_link_ = tip->getName();
        }

        kstate_ = std::make_shared<moveit::core::RobotState>(kmodel_);
        kstate_->setToDefaultValues();
        RCLCPP_INFO(get_logger(), "✅ MoveIt model ready (group: %s).", group_name_.c_str());
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "MoveIt model init failed: %s", e.what());
        return;
      }
    }

    // Start PlanningSceneMonitor once
    if (!psm_) {
      psm_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(this->shared_from_this(), "robot_description");
      if (!psm_ || !psm_->getPlanningScene()) {
        RCLCPP_WARN(get_logger(), "PlanningSceneMonitor not ready yet.");
        return;
      }
      psm_->startSceneMonitor();
      psm_->startStateMonitor("/joint_states");
      RCLCPP_INFO(get_logger(), "✅ PlanningSceneMonitor started.");
      moveit_ready_ = true;
    }

    // Wait until we have a complete state at least once
    if (psm_->getStateMonitor() && psm_->getStateMonitor()->haveCompleteState()) {
      if (!psm_ready_) {
        psm_ready_ = true;

        if (swc_enabled_) addSwincarCollision();

        // Ensure joint_names_ match the group's active joints (order AND names)
        const auto& group_joint_names = jmg_->getActiveJointModelNames();
        if (joint_names_.size() != group_joint_names.size() ||
            !std::equal(joint_names_.begin(), joint_names_.end(), group_joint_names.begin())) {
          std::ostringstream os_old, os_new;
          for (auto& n : joint_names_) os_old << " " << n;
          for (auto& n : group_joint_names) os_new << " " << n;
          RCLCPP_WARN(get_logger(),
            "JTC joint_names param doesn’t match MoveIt group. Overriding.\n param:%s\n group:%s",
            os_old.str().c_str(), os_new.str().c_str());
          joint_names_ = group_joint_names;
        }

        // Print controller joint list once
        std::ostringstream os;
        for (auto& n : joint_names_) os << " " << n;
        RCLCPP_INFO(get_logger(), "JTC joints:%s", os.str().c_str());

        RCLCPP_INFO(get_logger(), "✅ Planning scene ready.");
      }
      init_timer_->cancel();
      heartbeat_timer_->cancel();
    }
  }

  void addSwincarCollision()
  {
    // Pose from params
    geometry_msgs::msg::Pose pose;
    pose.position.x = swc_pose_xyz_[0];
    pose.position.y = swc_pose_xyz_[1];
    pose.position.z = swc_pose_xyz_[2];
    tf2::Quaternion q; q.setRPY(swc_pose_rpy_[0], swc_pose_rpy_[1], swc_pose_rpy_[2]); q.normalize();
    pose.orientation = tf2::toMsg(q);

    moveit_msgs::msg::CollisionObject co;
    co.id = "swincar";
    co.header.frame_id = world_frame_;

    try {
      shapes::Mesh* mesh = shapes::createMeshFromResource(
          swc_mesh_fixed_uri_, Eigen::Vector3d(1.0, 1.0, 1.0));
      if (!mesh) {
        RCLCPP_ERROR(get_logger(), "Swincar mesh failed to load: %s", swc_mesh_fixed_uri_.c_str());
        return;
      }

      shape_msgs::msg::Mesh mesh_msg;
      shapes::ShapeMsg shape_msg;
      shapes::constructMsgFromShape(mesh, shape_msg);
      mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg);

      co.meshes.push_back(mesh_msg);
      co.mesh_poses.push_back(pose);
      co.operation = co.ADD;

      psi_.applyCollisionObject(co);
      RCLCPP_INFO(get_logger(), "🧱 Added 'swincar' collision MESH: %s", swc_mesh_fixed_uri_.c_str());

      delete mesh;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Swincar mesh load exception: %s", e.what());
    }
  }

  // ---------- Sub / Pub ----------
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
  rclcpp::TimerBase::SharedPtr timer_, init_timer_, heartbeat_timer_;

  // ---------- TF ----------
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // ---------- IK / Scene ----------
  std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
  moveit::core::RobotModelPtr kmodel_;
  const moveit::core::JointModelGroup* jmg_{nullptr};
  moveit::core::RobotStatePtr kstate_;
  planning_scene_monitor::PlanningSceneMonitorPtr psm_;
  moveit::planning_interface::PlanningSceneInterface psi_;
  bool moveit_ready_{false};
  bool psm_ready_{false};

  // ---------- Params / state ----------
  std::string world_frame_, base_link_, ee_link_, group_name_;
  std::vector<std::string> joint_names_;
  std::vector<double> tool_axis_;
  double beam_length_{0.5};
  double err_tol_m_{0.01};
  double rate_hz_{100.0};
  double max_joint_step_{0.02};
  double min_time_step_{0.02};
  int    coll_check_every_{5};

  bool swc_enabled_{false};
  std::vector<double> swc_pose_xyz_, swc_pose_rpy_;

  // Hard-wired mesh (no fallback box, no params)
  const std::string swc_mesh_fixed_uri_ =
    "file:///home/alex/.ignition/gazebo/models/swincar/meshes/swincar_collision.dae";

  bool have_target_{false};
  tf2::Vector3 target_w_;

  // ---------- Helpers ----------
  static tf2::Vector3 norm(const tf2::Vector3& v)
  {
    double n = v.length();
    if (n < 1e-12) return tf2::Vector3(0,0,1);
    return v / n;
  }

  static tf2::Quaternion quatBetween(const tf2::Vector3& a_raw, const tf2::Vector3& b_raw)
  {
    tf2::Vector3 a = norm(a_raw), b = norm(b_raw);
    double c = a.dot(b);
    if (c < -0.999999)
    {
      tf2::Vector3 axis = a.cross(tf2::Vector3(1,0,0));
      if (axis.length2() < 1e-9) axis = a.cross(tf2::Vector3(0,1,0));
      axis = norm(axis);
      tf2::Quaternion q; q.setRotation(axis, M_PI); q.normalize(); return q;
    }
    tf2::Vector3 v = a.cross(b);
    tf2::Quaternion q(v.x(), v.y(), v.z(), 1.0 + c); q.normalize(); return q;
  }

  static tf2::Quaternion angleAxis(double angle, const tf2::Vector3& axis)
  {
    tf2::Quaternion q; q.setRotation(norm(axis), angle); q.normalize(); return q;
  }

  void targetCb(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    geometry_msgs::msg::PoseStamped in = *msg, w;
    if (in.header.frame_id.empty() || in.header.frame_id == world_frame_) {
      w = in; w.header.frame_id = world_frame_;
    } else {
      try {
        w = tf_buffer_.transform(in, world_frame_, tf2::durationFromSec(0.5));
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(get_logger(), "TF %s→%s failed: %s. Using raw pose as %s.",
                    in.header.frame_id.c_str(), world_frame_.c_str(), ex.what(), world_frame_.c_str());
        w = in; w.header.frame_id = world_frame_;
      }
    }
    target_w_ = tf2::Vector3(w.pose.position.x, w.pose.position.y, w.pose.position.z);
    have_target_ = true;
    RCLCPP_INFO(get_logger(), "🎯 New target (%s): [%.3f, %.3f, %.3f]",
                world_frame_.c_str(), target_w_.x(), target_w_.y(), target_w_.z());
  }

  bool getEEPoseWorld(tf2::Vector3& p_out, tf2::Matrix3x3& R_out)
  {
    try {
      auto tfStamped = tf_buffer_.lookupTransform(world_frame_, ee_link_, tf2::TimePointZero, tf2::durationFromSec(0.5));
      tf2::Transform T; tf2::fromMsg(tfStamped.transform, T);
      p_out = T.getOrigin(); R_out = T.getBasis(); return true;
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN(get_logger(), "TF world→%s failed: %s", ee_link_.c_str(), ex.what());
      return false;
    }
  }

  // NOTE: beam_len argument lets us "wiggle" the beam length during sampling
  bool solveOnce(const tf2::Vector3& tcp_axis_local,
                 const tf2::Vector3& target,
                 double roll_rad,
                 double radial_jitter,
                 double jitter_angle_rad,
                 double beam_len,
                 tf2::Vector3& wrist_out,
                 tf2::Quaternion& q_out)
  {
    // Current EE position
    tf2::Vector3 p_w; tf2::Matrix3x3 R_w;
    if (!getEEPoseWorld(p_w, R_w)) return false;

    // Direction from EE to (jittered) target (now with optional Z jitter via 'target')
    tf2::Vector3 jitter_offset(std::cos(jitter_angle_rad)*radial_jitter,
                               std::sin(jitter_angle_rad)*radial_jitter, 0.0);
    tf2::Vector3 tgt_j = target + jitter_offset;

    tf2::Vector3 dir = norm(tgt_j - p_w);
    tf2::Quaternion q_align = quatBetween(tcp_axis_local, dir);
    tf2::Quaternion q = angleAxis(roll_rad, dir) * q_align; q.normalize();

    tf2::Matrix3x3 R(q);
    tf2::Vector3 wrist = tgt_j - R * (tcp_axis_local * beam_len);

    // Guard: don't bury the wrist below floor (z<0). Adjust for your world as needed.
    if (wrist.z() < 0.0) return false;

    geometry_msgs::msg::Pose pose;
    pose.position.x = wrist.x(); pose.position.y = wrist.y(); pose.position.z = wrist.z();
    pose.orientation = tf2::toMsg(q);

    moveit::core::RobotState test_state(*kstate_);

    // IK options: allow approximate solution; we'll verify by FK
    kinematics::KinematicsQueryOptions kopt;
    kopt.return_approximate_solution = true;

    // More generous IK time
    constexpr double IK_TIMEOUT = 0.10; // 100 ms

    auto validity = [&](moveit::core::RobotState* state,
                        const moveit::core::JointModelGroup* group,
                        const double* ik_solution) -> bool
    {
      if (!psm_ready_) return true;
      state->setJointGroupPositions(group, ik_solution);
      state->update();
      collision_detection::CollisionRequest req;
      collision_detection::CollisionResult res;
      req.group_name = group->getName();
      req.max_contacts = 0;
      req.contacts = false;
      psm_->getPlanningScene()->checkCollision(req, res, *state);
      return !res.collision;
    };

    bool ok = false;
    if (psm_ready_) {
      ok = test_state.setFromIK(jmg_, pose, ee_link_, IK_TIMEOUT, validity, kopt);
    } else {
      ok = test_state.setFromIK(jmg_, pose, ee_link_, IK_TIMEOUT,
                                moveit::core::GroupStateValidityCallbackFn(), kopt);
    }
    if (!ok) return false;

    // Ensure transforms are fresh before FK
    test_state.update();

    // FK validation: ensure the solved pose points as requested within small tolerances
    tf2::Vector3 p_ach; tf2::Quaternion q_ach;
    if (!fkPose(test_state, ee_link_, p_ach, q_ach)) return false;

    constexpr double POS_TOL_POINT = 0.002;                     // 2 mm on wrist pose
    constexpr double ANG_TOL_POINT = 2.0 * M_PI / 180.0;        // 2 deg

    const double pos_err = (p_ach - wrist).length();
    const double ang_err = quatAngularErrorRad(q, q_ach);

    if (pos_err > POS_TOL_POINT || ang_err > ANG_TOL_POINT) return false;

    wrist_out = wrist; q_out = q; 
    return true;
  }

  bool buildPointingPose(tf2::Vector3& wrist_pos_w, tf2::Quaternion& q_w)
  {
    // Try both +axis and −axis
    tf2::Vector3 ax(tool_axis_[0], tool_axis_[1], tool_axis_[2]);
    tf2::Vector3 candidates_axes[2] = { ax, tf2::Vector3(-ax.x(), -ax.y(), -ax.z()) };

    // Denser sampling, with Z jitter and beam wiggle
    const double rolls_deg[] = {
      -180,-165,-150,-135,-120,-105,-90,-75,-60,-45,-30,-15,0,
      15,30,45,60,75,90,105,120,135,150,165,180
    };
    const double jit_r_xy[]   = { 0.00, 0.02, 0.04, 0.06 };
    const double jit_ang[]    = { 0, M_PI_2, M_PI, 3*M_PI_2 };
    const double jit_z[]      = { 0.00, 0.02, -0.02 };                 // up/down 2 cm
    const double beam_wiggle[] = { std::max(0.05, beam_length_ - 0.05), beam_length_, beam_length_ + 0.05 };

    for (const auto& axis_local : candidates_axes) {
      for (double bl : beam_wiggle) {
        for (double rd : rolls_deg) {
          const double roll = rd * M_PI / 180.0;
          for (double rj : jit_r_xy) {
            for (double aj : jit_ang) {
              for (double zj : jit_z) {
                tf2::Vector3 wrist; tf2::Quaternion q;
                const tf2::Vector3 tgt = target_w_ + tf2::Vector3(0,0,zj);
                if (solveOnce(axis_local, tgt, roll, rj, aj, bl, wrist, q)) {
                  if (wrist.z() < 0.0) continue;
                  wrist_pos_w = wrist; q_w = q;
                  return true;
                }
              }
            }
          }
        }
      }
    }
    return false;
  }

  void step()
  {
    if (!kmodel_ || !jmg_ || !kstate_ || !have_target_) return;

    // Keep seed in sync with current robot state if available
    if (psm_ready_ && psm_->getStateMonitor() && psm_->getStateMonitor()->haveCompleteState()) {
      auto cs = psm_->getStateMonitor()->getCurrentState();
      if (cs) {
        kstate_->setVariablePositions(cs->getVariablePositions());
        kstate_->update();
      }
    }

    tf2::Vector3 wrist_w; tf2::Quaternion q_w;
    if (!buildPointingPose(wrist_w, q_w)) {
      static int tick = 0; 
      if ((++tick % 50) == 0) { // ~0.5s @ 100 Hz
        RCLCPP_INFO(get_logger(), "…still searching feasible pointing pose (IK %s collision, ±axis, jitter, beam wiggle)…",
                    psm_ready_ ? "with" : "without");
      }
      return;
    }

    geometry_msgs::msg::Pose goal;
    goal.position.x = wrist_w.x(); 
    goal.position.y = wrist_w.y(); 
    goal.position.z = wrist_w.z();
    goal.orientation = tf2::toMsg(q_w);

    moveit::core::RobotState target_state(*kstate_);
    bool ok = false;

    // Occasionally do a collision-checked IK refine on the found pose
    static int tick_counter = 0;
    const bool do_collision_check = psm_ready_ && (coll_check_every_ > 0) &&
                                    ((tick_counter++ % std::max(1, coll_check_every_)) == 0);

    kinematics::KinematicsQueryOptions kopt;
    kopt.return_approximate_solution = true;   // allow approx, we validate by FK
    constexpr double IK_TIMEOUT = 0.10;        // match solveOnce()

    auto validity = [&](moveit::core::RobotState* state,
                        const moveit::core::JointModelGroup* group,
                        const double* ik_solution) -> bool
    {
      if (!do_collision_check) return true;
      state->setJointGroupPositions(group, ik_solution);
      state->update();
      collision_detection::CollisionRequest req;
      collision_detection::CollisionResult res;
      req.group_name = group->getName();
      req.max_contacts = 0;
      req.contacts = false;
      psm_->getPlanningScene()->checkCollision(req, res, *state);
      return !res.collision;
    };

    auto accept_solution = [&](moveit::core::RobotState& st) -> bool {
      tf2::Vector3 p_ach; tf2::Quaternion q_ach;
      if (!fkPose(st, ee_link_, p_ach, q_ach)) return false;
      constexpr double POS_TOL_POINT = 0.002;                    // 2 mm
      constexpr double ANG_TOL_POINT = 2.0 * M_PI / 180.0;       // 2 deg
      const double pos_err = (p_ach - wrist_w).length();
      const double ang_err = quatAngularErrorRad(q_w, q_ach);
      return (pos_err <= POS_TOL_POINT && ang_err <= ANG_TOL_POINT);
    };

    ok = target_state.setFromIK(jmg_, goal, ee_link_, IK_TIMEOUT, validity, kopt);
    if (ok) target_state.update();                  // <-- refresh transforms after IK
    ok = ok && accept_solution(target_state);

    if (!ok) {
      // fallback: try again without collision filtering (diagnostic)
      ok = target_state.setFromIK(jmg_, goal, ee_link_, IK_TIMEOUT,
                                  moveit::core::GroupStateValidityCallbackFn(), kopt);
      if (ok) target_state.update();                // <-- refresh transforms after IK
      ok = ok && accept_solution(target_state);
      if (!ok) {
        RCLCPP_DEBUG(get_logger(), "IK refine rejected (approx outside tolerance or failed).");
        return;
      }
    }

    // Small step towards target
    std::vector<double> q_curr, q_goal;
    kstate_->copyJointGroupPositions(jmg_, q_curr);
    target_state.copyJointGroupPositions(jmg_, q_goal);
    if (q_curr.size() != q_goal.size()) return;

    // pre-clamp infinity norm (max abs component of raw dq)
    double pre_inf = 0.0;

    std::vector<double> q_next = q_curr;
    double max_adq = 0.0;
    for (size_t i = 0; i < q_curr.size(); ++i) {
      double dq = q_goal[i] - q_curr[i];
      while (dq >  M_PI) dq -= 2*M_PI;
      while (dq < -M_PI) dq += 2*M_PI;
      pre_inf = std::max(pre_inf, std::abs(dq));
      double adq = std::clamp(dq, -max_joint_step_, max_joint_step_);
      q_next[i] = q_curr[i] + adq;
      max_adq = std::max(max_adq, std::abs(adq));
    }

    // If even the raw dq is tiny, skip publishing noise
    if (pre_inf < 1e-7 && max_adq < 1e-7) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 1000,
                           "IK returned current state (no motion). Continuing search.");
      return;
    }

    // Update seed for next tick
    kstate_->setJointGroupPositions(jmg_, q_next);
    kstate_->update();

    // ===== PUBLISH TRAJECTORY (well-formed for JTC) =====
    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = this->get_clock()->now();
    traj.joint_names = joint_names_;

    trajectory_msgs::msg::JointTrajectoryPoint p0, p1;
    p0.positions   = q_curr;
    p0.velocities  = std::vector<double>(q_curr.size(), 0.0);
    p0.time_from_start.sec = 0;
    p0.time_from_start.nanosec = 0;

    p1.positions   = q_next;
    p1.velocities  = std::vector<double>(q_next.size(), 0.0);
    const double sec_d = std::max(1e-3, min_time_step_);
    const int32_t sec_i = static_cast<int32_t>(sec_d);
    const uint32_t nsec_i = static_cast<uint32_t>((sec_d - static_cast<double>(sec_i)) * 1e9);
    p1.time_from_start.sec = sec_i; 
    p1.time_from_start.nanosec = nsec_i;

    traj.points = {p0, p1};
    traj_pub_->publish(traj);

    RCLCPP_INFO(get_logger(), "→ published step (pre ||Δq||_∞=%.4g, post ||Δq||_∞=%.4g) (max |Δq|=%.4f rad, %s collision check)",
                pre_inf, max_adq, max_adq, do_collision_check ? "WITH" : "SKIPPED");

    // Beam tip error (for stop condition)
    tf2::Vector3 ee_p; tf2::Matrix3x3 ee_R;
    if (getEEPoseWorld(ee_p, ee_R))
    {
      tf2::Vector3 axis(tool_axis_[0], tool_axis_[1], tool_axis_[2]);
      tf2::Vector3 tip = ee_p + ee_R * axis * beam_length_;
      double err = (tip - target_w_).length();
      if (err < err_tol_m_) {
        RCLCPP_INFO(get_logger(), "✅ Target reached (err %.3f m).", err);
        have_target_ = false;
      }
    }
  }

};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointingServo>());
  rclcpp::shutdown();
  return 0;
}
