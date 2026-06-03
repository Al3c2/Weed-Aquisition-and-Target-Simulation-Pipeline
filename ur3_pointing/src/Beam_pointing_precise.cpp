// Beam_pointing_robust.cpp

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

// MoveIt2
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>

// Collision mesh
#include <geometric_shapes/shapes.h>
#include <geometric_shapes/shape_operations.h>
#include <shape_msgs/msg/mesh.hpp>

// TF2
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <deque>
#include <mutex>

static const char BASE_FRAME[]  = "base_link";
static const char WORLD_FRAME[] = "world";
static const char EE_LINK[]     = "tool0";
static const char GROUP_NAME[]  = "ur3_manipulator";

// === Joint angle normalization to prevent 2π wrap-around ===
inline double normalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

inline double closestAngle(double angle, double reference) {
  double normalized = normalizeAngle(angle);
  double options[3] = { normalized, normalized + 2*M_PI, normalized - 2*M_PI };
  
  double best = normalized;
  double best_diff = std::fabs(normalized - reference);
  
  for (int i = 1; i < 3; i++) {
    double diff = std::fabs(options[i] - reference);
    if (diff < best_diff) {
      best_diff = diff;
      best = options[i];
    }
  }
  return best;
}

class RobustBeamPointingNode : public rclcpp::Node
{
public:
  RobustBeamPointingNode()
  : Node("Beam_pointing_precise"),
    tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
    tf_listener_(*tf_buffer_)
  {
    RCLCPP_INFO(get_logger(), "🚀 Robust Beam Pointing Node starting...");

    // === Beam parameters ===
    beam_length_ = declare_parameter<double>("beam_length", 0.65);
    ray_L_min_   = declare_parameter<double>("ray_L_min", 0.62);
    ray_L_max_   = declare_parameter<double>("ray_L_max", 0.68);
    ray_L_step_  = declare_parameter<double>("ray_L_step", 0.03);

    // === Performance parameters (CONSERVATIVE for stability) ===
    cartesian_step_size_ = declare_parameter<double>("cartesian_step_size", 0.01);
    cartesian_min_fraction_ = declare_parameter<double>("cartesian_min_fraction", 0.90);
    
    planning_timeout_ = declare_parameter<double>("planning_timeout", 2.0);
    num_planning_attempts_ = declare_parameter<int>("num_planning_attempts", 5);
    
    // Conservative velocity for stability
    velocity_scale_ = declare_parameter<double>("velocity_scale", 0.25);
    acceleration_scale_ = declare_parameter<double>("acceleration_scale", 0.15);

    // === Collision parameters ===
    enable_swincar_collision_ = declare_parameter<bool>("enable_swincar_collision", true);
    swincar_mesh_uri_ = declare_parameter<std::string>(
        "swincar_mesh_uri",
        "file:///home/alex/.ignition/gazebo/models/swincar/meshes/swincar_collision.dae");
    swincar_pose_xyz_ = declare_parameter<std::vector<double>>(
        "swincar_pose_xyz", std::vector<double>{0.0, 0.0, 0.0});
    swincar_pose_rpy_ = declare_parameter<std::vector<double>>(
        "swincar_pose_rpy", std::vector<double>{0.0, 0.0, 0.0});
    
    enable_ground_collision_ = declare_parameter<bool>("enable_ground_collision", true);
    ground_clearance_ = declare_parameter<double>("ground_clearance", 0.05);
    
    // Camera view protection
    enable_camera_protection_ = declare_parameter<bool>("enable_camera_protection", false);
    camera_protection_length_ = declare_parameter<double>("camera_protection_length", 0.6);
    camera_protection_radius_ = declare_parameter<double>("camera_protection_radius", 0.08);
    camera_protection_y_offset_ = declare_parameter<double>("camera_protection_y_offset", 0.35);
    
    // Ground truth for collision updates
    ground_truth_topic_ = declare_parameter<std::string>(
        "ground_truth_topic", "/model/swincar_ur3/pose");
    ground_truth_frame_id_ = declare_parameter<std::string>(
        "ground_truth_frame_id", "empty");
    
    // Result topics
    std::string done_topic = declare_parameter<std::string>("arm_done_topic", "/beam_task_done");
    std::string failed_topic = declare_parameter<std::string>("arm_failed_topic", "/beam_task_failed");
    beam_done_pub_ = create_publisher<std_msgs::msg::Bool>(done_topic, 10);
    beam_failed_pub_ = create_publisher<std_msgs::msg::Bool>(failed_topic, 10);
    
    beam_error_threshold_ = declare_parameter<double>("beam_error_threshold", 0.02);

    // === Subscriptions ===
    target_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "target_pose", 10,
      std::bind(&RobustBeamPointingNode::targetCallback, this, std::placeholders::_1));

    multi_target_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "target_poses", 10,
      std::bind(&RobustBeamPointingNode::multiTargetCallback, this, std::placeholders::_1));

    ground_truth_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      ground_truth_topic_, 10,
      std::bind(&RobustBeamPointingNode::groundTruthCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "📡 Subscribed to 'target_pose' (base_link frame expected)");

    // Delayed initialization
    init_timer_ = create_wall_timer(
      std::chrono::seconds(2),
      std::bind(&RobustBeamPointingNode::initializeMoveIt, this));
  }

private:
  // Parameters
  double beam_length_, ray_L_min_, ray_L_max_, ray_L_step_;
  double cartesian_step_size_, cartesian_min_fraction_;
  double planning_timeout_, velocity_scale_, acceleration_scale_;
  int num_planning_attempts_;
  
  bool enable_swincar_collision_, enable_ground_collision_;
  std::string swincar_mesh_uri_;
  std::vector<double> swincar_pose_xyz_, swincar_pose_rpy_;
  double ground_clearance_;
  
  // Camera protection
  bool enable_camera_protection_;
  double camera_protection_length_, camera_protection_radius_, camera_protection_y_offset_;
  
  std::string ground_truth_topic_, ground_truth_frame_id_;
  double beam_error_threshold_;

  // ROS/MoveIt
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
  
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr multi_target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ground_truth_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr beam_done_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr beam_failed_pub_;
  
  rclcpp::TimerBase::SharedPtr init_timer_;
  rclcpp::TimerBase::SharedPtr collision_timer_;
  bool moveit_ready_ = false;
  
  // Ground truth
  geometry_msgs::msg::Pose ground_truth_pose_;
  geometry_msgs::msg::Pose last_collision_pose_;
  bool ground_truth_received_ = false;
  std::mutex pose_mutex_;
  
  // Statistics
  int total_targets_ = 0;
  int successful_targets_ = 0;
  int consecutive_failures_ = 0;
  static constexpr int MAX_CONSECUTIVE_FAILURES = 3;
  
  // Track the beam length used for current target
  double current_beam_length_ = 0.65;

  void groundTruthCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (msg->header.frame_id != ground_truth_frame_id_) return;
    std::lock_guard<std::mutex> lock(pose_mutex_);
    ground_truth_pose_ = msg->pose;
    ground_truth_received_ = true;
  }

  void initializeMoveIt()
  {
    init_timer_->cancel();
    RCLCPP_INFO(get_logger(), "🔧 Initializing MoveIt...");
    
    try {
      move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), GROUP_NAME);
      
      planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
      
      move_group_->setPoseReferenceFrame(BASE_FRAME);
      move_group_->setEndEffectorLink(EE_LINK);
      
      move_group_->setPlanningTime(planning_timeout_);
      move_group_->setNumPlanningAttempts(num_planning_attempts_);
      move_group_->setPlannerId("RRTConnectkConfigDefault");
      
      move_group_->setGoalPositionTolerance(0.005);
      move_group_->setGoalOrientationTolerance(0.03);
      
      move_group_->setMaxVelocityScalingFactor(velocity_scale_);
      move_group_->setMaxAccelerationScalingFactor(acceleration_scale_);
      
      move_group_->startStateMonitor();

      if (!move_group_->getCurrentState(5.0)) {
        RCLCPP_WARN(get_logger(), "Could not get robot state");
      }

      initializeCollisions();

      moveit_ready_ = true;
      RCLCPP_INFO(get_logger(), "✅ MoveIt ready! Beam: %.2fm, Vel: %.0f%%, Frame: %s",
        beam_length_, velocity_scale_ * 100, BASE_FRAME);

      collision_timer_ = create_wall_timer(
        std::chrono::milliseconds(250),
        std::bind(&RobustBeamPointingNode::updateCollisions, this));
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "❌ MoveIt init failed: %s", e.what());
      init_timer_ = create_wall_timer(
        std::chrono::seconds(2),
        std::bind(&RobustBeamPointingNode::initializeMoveIt, this));
    }
  }

  void initializeCollisions()
  {
    try {
      auto tf = tf_buffer_->lookupTransform(WORLD_FRAME, BASE_FRAME, 
        tf2::TimePointZero, tf2::durationFromSec(2.0));
      ground_truth_pose_.position.x = tf.transform.translation.x;
      ground_truth_pose_.position.y = tf.transform.translation.y;
      ground_truth_pose_.position.z = tf.transform.translation.z;
      ground_truth_pose_.orientation = tf.transform.rotation;
    } catch (...) {
      ground_truth_pose_.orientation.w = 1.0;
    }
    
    last_collision_pose_ = ground_truth_pose_;
    
    if (enable_ground_collision_) addGroundCollision();
    if (enable_swincar_collision_) addSwincarCollision();
    
    addCameraViewProtection();
  }

  void addCameraViewProtection()
  {
    if (!enable_camera_protection_) return;
    
    RCLCPP_INFO(get_logger(), "🎥 Adding camera view protection zone...");
    
    moveit_msgs::msg::CollisionObject camera_zone;
    camera_zone.id = "camera_view_zone";
    camera_zone.operation = camera_zone.ADD;
    camera_zone.header.frame_id = BASE_FRAME;
    camera_zone.header.stamp = now();

    shape_msgs::msg::SolidPrimitive cylinder;
    cylinder.type = cylinder.CYLINDER;
    cylinder.dimensions.resize(2);
    cylinder.dimensions[0] = camera_protection_length_;
    cylinder.dimensions[1] = camera_protection_radius_;

    geometry_msgs::msg::Pose cylinder_pose;
    cylinder_pose.position.x = -0.1;
    cylinder_pose.position.y = camera_protection_y_offset_;
    cylinder_pose.position.z = -0.05;
    
    tf2::Quaternion q;
    q.setRPY(M_PI/2, 0, 0);
    cylinder_pose.orientation = tf2::toMsg(q);

    camera_zone.primitives.push_back(cylinder);
    camera_zone.primitive_poses.push_back(cylinder_pose);
    
    planning_scene_interface_->applyCollisionObjects({camera_zone});
    
    RCLCPP_INFO(get_logger(), "   Camera view protection: cylinder at y=%.2f, radius=%.2fm, length=%.2fm",
      camera_protection_y_offset_, camera_protection_radius_, camera_protection_length_);
  }

  void addGroundCollision()
  {
    moveit_msgs::msg::CollisionObject ground;
    ground.id = "ground_plane";
    ground.operation = ground.ADD;
    ground.header.frame_id = WORLD_FRAME;
    ground.header.stamp = now();

    shape_msgs::msg::SolidPrimitive prim;
    prim.type = prim.BOX;
    prim.dimensions = {20.0, 20.0, 0.1};

    geometry_msgs::msg::Pose pose;
    pose.position.x = ground_truth_pose_.position.x;
    pose.position.y = ground_truth_pose_.position.y;
    pose.position.z = ground_clearance_ - 0.05;
    pose.orientation.w = 1.0;

    ground.primitives.push_back(prim);
    ground.primitive_poses.push_back(pose);
    planning_scene_interface_->applyCollisionObjects({ground});
    
    RCLCPP_INFO(get_logger(), "🛡️ Ground collision added");
  }

  void addSwincarCollision()
  {
    RCLCPP_INFO(get_logger(), "🔧 Loading Swincar collision mesh from: %s", swincar_mesh_uri_.c_str());
    
    auto mesh = std::unique_ptr<shapes::Mesh>(shapes::createMeshFromResource(swincar_mesh_uri_));
    if (!mesh) {
      RCLCPP_ERROR(get_logger(), "❌ Could not load swincar mesh from: %s", swincar_mesh_uri_.c_str());
      return;
    }
    
    RCLCPP_INFO(get_logger(), "   Mesh loaded: %u vertices, %u triangles", 
      mesh->vertex_count, mesh->triangle_count);

    shapes::ShapeMsg shape_msg;
    shapes::constructMsgFromShape(mesh.get(), shape_msg);
    auto mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg);

    moveit_msgs::msg::CollisionObject obj;
    obj.id = "swincar";
    obj.operation = obj.ADD;
    obj.header.frame_id = WORLD_FRAME;
    obj.header.stamp = now();

    geometry_msgs::msg::Pose pose;
    pose.position.x = ground_truth_pose_.position.x + 
      (swincar_pose_xyz_.size() > 0 ? swincar_pose_xyz_[0] : 0.0);
    pose.position.y = ground_truth_pose_.position.y + 
      (swincar_pose_xyz_.size() > 1 ? swincar_pose_xyz_[1] : 0.0);
    pose.position.z = ground_truth_pose_.position.z +
      (swincar_pose_xyz_.size() > 2 ? swincar_pose_xyz_[2] : 0.0);

    tf2::Quaternion robot_q;
    tf2::fromMsg(ground_truth_pose_.orientation, robot_q);
    
    double r = swincar_pose_rpy_.size() > 0 ? swincar_pose_rpy_[0] : 0.0;
    double p = swincar_pose_rpy_.size() > 1 ? swincar_pose_rpy_[1] : 0.0;
    double y = swincar_pose_rpy_.size() > 2 ? swincar_pose_rpy_[2] : 0.0;
    tf2::Quaternion offset_q;
    offset_q.setRPY(r, p, y);
    
    tf2::Quaternion combined_q = robot_q * offset_q;
    combined_q.normalize();
    pose.orientation = tf2::toMsg(combined_q);

    RCLCPP_INFO(get_logger(), "   Swincar collision pose: [%.2f, %.2f, %.2f]",
      pose.position.x, pose.position.y, pose.position.z);

    obj.meshes = {mesh_msg};
    obj.mesh_poses = {pose};

    planning_scene_interface_->applyCollisionObjects({obj});
    rclcpp::sleep_for(std::chrono::milliseconds(500));
    
    RCLCPP_INFO(get_logger(), "🧱 Swincar collision mesh added to planning scene");
  }

  void updateCollisions()
  {
    if (!moveit_ready_ || !ground_truth_received_) return;
    
    geometry_msgs::msg::Pose current;
    {
      std::lock_guard<std::mutex> lock(pose_mutex_);
      current = ground_truth_pose_;
    }
    
    double dx = current.position.x - last_collision_pose_.position.x;
    double dy = current.position.y - last_collision_pose_.position.y;
    if (std::sqrt(dx*dx + dy*dy) < 0.05) return;
    
    last_collision_pose_ = current;
    
    {
      moveit_msgs::msg::CollisionObject ground;
      ground.id = "ground_plane";
      ground.operation = ground.MOVE;
      ground.header.frame_id = WORLD_FRAME;
      ground.header.stamp = now();

      geometry_msgs::msg::Pose pose;
      pose.position.x = current.position.x;
      pose.position.y = current.position.y;
      pose.position.z = ground_clearance_ - 0.05;
      pose.orientation.w = 1.0;
      ground.primitive_poses.push_back(pose);
      
      planning_scene_interface_->applyCollisionObjects({ground});
    }
    
    if (enable_swincar_collision_) {
      moveit_msgs::msg::CollisionObject obj;
      obj.id = "swincar";
      obj.operation = obj.MOVE;
      obj.header.frame_id = WORLD_FRAME;
      obj.header.stamp = now();

      geometry_msgs::msg::Pose pose;
      pose.position.x = current.position.x + (swincar_pose_xyz_.size() > 0 ? swincar_pose_xyz_[0] : 0.0);
      pose.position.y = current.position.y + (swincar_pose_xyz_.size() > 1 ? swincar_pose_xyz_[1] : 0.0);
      pose.position.z = current.position.z + (swincar_pose_xyz_.size() > 2 ? swincar_pose_xyz_[2] : 0.0);
      
      tf2::Quaternion robot_q;
      tf2::fromMsg(current.orientation, robot_q);
      
      double r = swincar_pose_rpy_.size() > 0 ? swincar_pose_rpy_[0] : 0.0;
      double p = swincar_pose_rpy_.size() > 1 ? swincar_pose_rpy_[1] : 0.0;
      double y = swincar_pose_rpy_.size() > 2 ? swincar_pose_rpy_[2] : 0.0;
      tf2::Quaternion offset_q;
      offset_q.setRPY(r, p, y);
      
      tf2::Quaternion combined_q = robot_q * offset_q;
      combined_q.normalize();
      pose.orientation = tf2::toMsg(combined_q);
      
      obj.mesh_poses.push_back(pose);
      
      planning_scene_interface_->applyCollisionObjects({obj});
    }
  }

  bool getTool0Pose(tf2::Vector3& pos, tf2::Quaternion& orient)
  {
    try {
      auto tf = tf_buffer_->lookupTransform(BASE_FRAME, EE_LINK, 
        tf2::TimePointZero, tf2::durationFromSec(0.5));
      pos.setValue(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
      tf2::fromMsg(tf.transform.rotation, orient);
      return true;
    } catch (const tf2::TransformException& e) {
      RCLCPP_ERROR(get_logger(), "TF error: %s", e.what());
      return false;
    }
  }

  tf2::Quaternion alignZToDirection(const tf2::Vector3& dir)
  {
    tf2::Vector3 z_axis(0.0, 0.0, 1.0);
    tf2::Vector3 v = dir.normalized();
    double dot = z_axis.dot(v);

    tf2::Quaternion q;
    if (dot > 0.9999) {
      q.setValue(0, 0, 0, 1);
    } else if (dot < -0.9999) {
      q.setRotation(tf2::Vector3(1, 0, 0), M_PI);
    } else {
      tf2::Vector3 axis = z_axis.cross(v);
      axis.normalize();
      double angle = std::acos(std::clamp(dot, -1.0, 1.0));
      q.setRotation(axis, angle);
    }
    q.normalize();
    return q;
  }

  std::vector<double> candidateLengths() const
  {
    std::vector<double> Ls;
    for (double L = ray_L_min_; L <= ray_L_max_ + 1e-6; L += ray_L_step_) {
      Ls.push_back(L);
    }
    std::sort(Ls.begin(), Ls.end(), [this](double a, double b) {
      return std::fabs(a - beam_length_) < std::fabs(b - beam_length_);
    });
    return Ls;
  }

  bool computeGoalPose(const tf2::Vector3& target, double L,
                       geometry_msgs::msg::Pose& goal_pose,
                       tf2::Vector3& direction)
  {
    tf2::Vector3 tcp_pos;
    tf2::Quaternion tcp_orient;
    if (!getTool0Pose(tcp_pos, tcp_orient)) return false;

    direction = target - tcp_pos;
    double dist = direction.length();
    if (dist < 0.05) {
      RCLCPP_WARN(get_logger(), "Target too close (%.2fm)", dist);
      return false;
    }
    direction.normalize();

    tf2::Vector3 goal_pos = target - direction * L;
    for (int i = 0; i < 3; ++i) {
      tf2::Vector3 new_dir = (target - goal_pos).normalized();
      goal_pos = target - new_dir * L;
      if ((new_dir - direction).length() < 0.001) break;
      direction = new_dir;
    }

    tf2::Quaternion goal_q = alignZToDirection(direction);

    goal_pose.position.x = goal_pos.x();
    goal_pose.position.y = goal_pos.y();
    goal_pose.position.z = goal_pos.z();
    goal_pose.orientation = tf2::toMsg(goal_q);

    return true;
  }

  bool isStateCollisionFree(const moveit::core::RobotState& state)
  {
    auto robot_model = move_group_->getRobotModel();
    const auto* jmg = robot_model->getJointModelGroup(GROUP_NAME);
    if (!jmg) return true;
    
    std::vector<double> joint_values;
    state.copyJointGroupPositions(jmg, joint_values);
    
    for (size_t i = 0; i < joint_values.size(); i++) {
      const auto& bounds = jmg->getActiveJointModelsBounds()[i];
      for (const auto& bound : *bounds) {
        if (bound.position_bounded_) {
          if (joint_values[i] < bound.min_position_ || joint_values[i] > bound.max_position_) {
            RCLCPP_DEBUG(get_logger(), "Joint %zu out of bounds: %.3f not in [%.3f, %.3f]",
              i, joint_values[i], bound.min_position_, bound.max_position_);
            return false;
          }
        }
      }
    }
    
    return true;
  }
  
  bool isIKSolutionValid(const moveit::core::RobotState& goal_state)
  {
    auto robot_model = move_group_->getRobotModel();
    const auto* jmg = robot_model->getJointModelGroup(GROUP_NAME);
    if (!jmg) return false;
    
    double orig_planning_time = planning_timeout_;
    int orig_attempts = num_planning_attempts_;
    
    move_group_->setPlanningTime(0.2);
    move_group_->setNumPlanningAttempts(1);
    move_group_->setStartStateToCurrentState();
    move_group_->setJointValueTarget(goal_state);
    
    moveit::planning_interface::MoveGroupInterface::Plan test_plan;
    auto result = move_group_->plan(test_plan);
    
    move_group_->setPlanningTime(orig_planning_time);
    move_group_->setNumPlanningAttempts(orig_attempts);
    move_group_->clearPoseTargets();
    
    return (result == moveit::core::MoveItErrorCode::SUCCESS);
  }

  bool computeIK(const geometry_msgs::msg::Pose& goal_pose,
                 moveit::core::RobotState& result_state,
                 double timeout = 0.3,
                 bool check_validity = true)
  {
    auto robot_model = move_group_->getRobotModel();
    const auto* jmg = robot_model->getJointModelGroup(GROUP_NAME);
    if (!jmg) return false;
    
    auto current_state = move_group_->getCurrentState();
    if (!current_state) return false;
    
    result_state = *current_state;
    
    bool success = result_state.setFromIK(jmg, goal_pose, EE_LINK, timeout);
    
    if (!success) return false;
    
    normalizeJointAngles(result_state, *current_state);
    result_state.update();
    
    if (!isStateCollisionFree(result_state)) {
      RCLCPP_DEBUG(get_logger(), "   IK solution violates joint limits");
      return false;
    }
    
    if (check_validity && !isIKSolutionValid(result_state)) {
      RCLCPP_DEBUG(get_logger(), "   IK solution fails planning check (likely collision)");
      return false;
    }
    
    return true;
  }

  bool computeIKRobust(const geometry_msgs::msg::Pose& goal_pose,
                       moveit::core::RobotState& result_state)
  {
    if (computeIK(goal_pose, result_state, 0.2, true)) {
      RCLCPP_INFO(get_logger(), "   IK succeeded with original orientation");
      return true;
    }
    
    tf2::Quaternion base_q;
    tf2::fromMsg(goal_pose.orientation, base_q);
    
    tf2::Matrix3x3 R(base_q);
    tf2::Vector3 beam_axis(R[0][2], R[1][2], R[2][2]);
    
    double rotations[] = {M_PI/2, -M_PI/2, M_PI, M_PI/4, -M_PI/4, 3*M_PI/4, -3*M_PI/4};
    
    for (double rot : rotations) {
      tf2::Quaternion rot_q;
      rot_q.setRotation(beam_axis, rot);
      tf2::Quaternion perturbed_q = base_q * rot_q;
      perturbed_q.normalize();
      
      geometry_msgs::msg::Pose perturbed_pose = goal_pose;
      perturbed_pose.orientation = tf2::toMsg(perturbed_q);
      
      if (computeIK(perturbed_pose, result_state, 0.15, true)) {
        RCLCPP_INFO(get_logger(), "   IK succeeded with %.0f° rotation", rot * 180 / M_PI);
        return true;
      }
    }
    
    RCLCPP_WARN(get_logger(), "   No valid IK solution found after trying all orientations");
    return false;
  }

  void normalizeJointAngles(moveit::core::RobotState& target_state,
                            const moveit::core::RobotState& current_state)
  {
    const auto* jmg = target_state.getJointModelGroup(GROUP_NAME);
    if (!jmg) return;
    
    std::vector<double> target_vals, current_vals;
    target_state.copyJointGroupPositions(jmg, target_vals);
    current_state.copyJointGroupPositions(jmg, current_vals);

    bool modified = false;
    for (size_t i = 0; i < target_vals.size() && i < current_vals.size(); i++) {
      double diff = std::fabs(target_vals[i] - current_vals[i]);
      if (diff > M_PI) {
        target_vals[i] = closestAngle(target_vals[i], current_vals[i]);
        modified = true;
      }
    }

    if (modified) {
      target_state.setJointGroupPositions(jmg, target_vals);
      RCLCPP_DEBUG(get_logger(), "Normalized joint angles");
    }
  }

  void normalizeTrajectory(trajectory_msgs::msg::JointTrajectory& traj)
  {
    if (traj.points.empty()) return;
    
    auto current_state = move_group_->getCurrentState();
    if (!current_state) return;
    
    const auto* jmg = current_state->getJointModelGroup(GROUP_NAME);
    if (!jmg) return;
    
    std::vector<double> reference;
    current_state->copyJointGroupPositions(jmg, reference);
    
    for (auto& point : traj.points) {
      for (size_t i = 0; i < point.positions.size() && i < reference.size(); i++) {
        double diff = std::fabs(point.positions[i] - reference[i]);
        if (diff > M_PI) {
          point.positions[i] = closestAngle(point.positions[i], reference[i]);
        }
      }
      reference = point.positions;
    }
  }

  bool tryCartesianPath(const geometry_msgs::msg::Pose& goal_pose,
                        moveit::planning_interface::MoveGroupInterface::Plan& plan)
  {
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(goal_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = move_group_->computeCartesianPath(
      waypoints, cartesian_step_size_, 0.0, trajectory, true);

    if (fraction < cartesian_min_fraction_) {
      RCLCPP_DEBUG(get_logger(), "   Cartesian: %.0f%% (need %.0f%%)", 
        fraction * 100, cartesian_min_fraction_ * 100);
      return false;
    }

    robot_trajectory::RobotTrajectory rt(move_group_->getRobotModel(), GROUP_NAME);
    rt.setRobotTrajectoryMsg(*move_group_->getCurrentState(), trajectory);
    
    trajectory_processing::IterativeParabolicTimeParameterization iptp;
    if (!iptp.computeTimeStamps(rt, velocity_scale_, acceleration_scale_)) {
      RCLCPP_WARN(get_logger(), "   Time parameterization failed");
      return false;
    }

    rt.getRobotTrajectoryMsg(plan.trajectory_);
    normalizeTrajectory(plan.trajectory_.joint_trajectory);
    
    if (!plan.trajectory_.joint_trajectory.points.empty()) {
      auto& last = plan.trajectory_.joint_trajectory.points.back();
      double duration = last.time_from_start.sec + last.time_from_start.nanosec * 1e-9;
      if (duration > 8.0) {
        RCLCPP_WARN(get_logger(), "   Trajectory too long: %.1fs", duration);
        return false;
      }
    }
    
    RCLCPP_INFO(get_logger(), "   ✓ Cartesian path (%.0f%%)", fraction * 100);
    return true;
  }

  bool tryJointSpacePlan(const moveit::core::RobotState& goal_state,
                         moveit::planning_interface::MoveGroupInterface::Plan& plan)
  {
    move_group_->setStartStateToCurrentState();
    move_group_->setJointValueTarget(goal_state);
    
    auto result = move_group_->plan(plan);
    move_group_->clearPoseTargets();
    
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_DEBUG(get_logger(), "   Joint space planning failed: %d", result.val);
      return false;
    }
    
    normalizeTrajectory(plan.trajectory_.joint_trajectory);
    RCLCPP_INFO(get_logger(), "   ✓ Joint space plan");
    return true;
  }

  double verifyBeamTip(const tf2::Vector3& target, double beam_length)
  {
    tf2::Vector3 tcp_pos;
    tf2::Quaternion tcp_orient;
    
    if (!getTool0Pose(tcp_pos, tcp_orient)) {
      return 1.0;
    }

    tf2::Matrix3x3 R(tcp_orient);
    tf2::Vector3 beam_dir(R[0][2], R[1][2], R[2][2]);
    beam_dir.normalize();
    
    tf2::Vector3 beam_tip = tcp_pos + beam_dir * beam_length;
    double error = (beam_tip - target).length();
    
    RCLCPP_INFO(get_logger(), "   📍 Verification:");
    RCLCPP_INFO(get_logger(), "      TCP: [%.3f, %.3f, %.3f]", tcp_pos.x(), tcp_pos.y(), tcp_pos.z());
    RCLCPP_INFO(get_logger(), "      Beam dir: [%.3f, %.3f, %.3f]", beam_dir.x(), beam_dir.y(), beam_dir.z());
    RCLCPP_INFO(get_logger(), "      Beam length: %.3fm", beam_length);
    RCLCPP_INFO(get_logger(), "      Beam tip: [%.3f, %.3f, %.3f]", beam_tip.x(), beam_tip.y(), beam_tip.z());
    RCLCPP_INFO(get_logger(), "      Target: [%.3f, %.3f, %.3f]", target.x(), target.y(), target.z());
    RCLCPP_INFO(get_logger(), "      Error: %.1fmm", error * 1000);
    
    return error;
  }

  bool recoverToSafePosition()
  {
    RCLCPP_INFO(get_logger(), "🔄 Attempting recovery to safe position...");
    
    std::vector<std::vector<double>> safe_positions = {
      {0.0, -1.57, 0.0, -1.57, 0.0, 0.0},
      {0.0, -2.0, 1.5, -1.07, -1.57, 0.0},
      {0.5, -1.8, 1.2, -0.97, -1.57, 0.0},
      {-0.5, -1.8, 1.2, -0.97, -1.57, 0.0},
    };
    
    double orig_vel = velocity_scale_;
    double orig_acc = acceleration_scale_;
    double orig_time = planning_timeout_;
    int orig_attempts = num_planning_attempts_;
    
    for (size_t i = 0; i < safe_positions.size(); i++) {
      RCLCPP_INFO(get_logger(), "   Trying safe position %zu/%zu...", i + 1, safe_positions.size());
      
      move_group_->setMaxVelocityScalingFactor(0.2);
      move_group_->setMaxAccelerationScalingFactor(0.1);
      move_group_->setPlanningTime(5.0);
      move_group_->setNumPlanningAttempts(10);
      
      move_group_->setStartStateToCurrentState();
      move_group_->setJointValueTarget(safe_positions[i]);
      
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      auto result = move_group_->plan(plan);
      
      if (result == moveit::core::MoveItErrorCode::SUCCESS) {
        normalizeTrajectory(plan.trajectory_.joint_trajectory);
        
        auto exec_result = move_group_->execute(plan);
        if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) {
          RCLCPP_INFO(get_logger(), "✅ Recovery successful with position %zu!", i + 1);
          
          move_group_->setMaxVelocityScalingFactor(orig_vel);
          move_group_->setMaxAccelerationScalingFactor(orig_acc);
          move_group_->setPlanningTime(orig_time);
          move_group_->setNumPlanningAttempts(orig_attempts);
          
          rclcpp::sleep_for(std::chrono::milliseconds(500));
          
          return true;
        }
      }
    }
    
    RCLCPP_ERROR(get_logger(), "❌ All recovery attempts failed");
    
    move_group_->stop();
    
    move_group_->setMaxVelocityScalingFactor(orig_vel);
    move_group_->setMaxAccelerationScalingFactor(orig_acc);
    move_group_->setPlanningTime(orig_time);
    move_group_->setNumPlanningAttempts(orig_attempts);
    
    return false;
  }

  bool attemptTargetWithRecovery(const tf2::Vector3& target, int attempt_num = 1)
  {
    auto lengths = candidateLengths();
    
    geometry_msgs::msg::Pose goal_pose;
    moveit::core::RobotState goal_state(move_group_->getRobotModel());
    tf2::Vector3 direction;
    bool found_ik = false;
    double chosen_L = 0.0;

    for (double L : lengths) {
      if (!computeGoalPose(target, L, goal_pose, direction)) continue;

      if (computeIKRobust(goal_pose, goal_state)) {
        chosen_L = L;
        found_ik = true;
        RCLCPP_INFO(get_logger(), "   Found valid IK for L=%.3fm", L);
        break;
      }
    }

    if (!found_ik) {
      RCLCPP_WARN(get_logger(), "❌ No valid IK solution found");
      
      if (attempt_num == 1) {
        RCLCPP_WARN(get_logger(), "   No IK found - attempting recovery and retry...");
        if (recoverToSafePosition()) {
          return attemptTargetWithRecovery(target, 2);
        }
      }
      
      return false;
    }

    current_beam_length_ = chosen_L;

    RCLCPP_INFO(get_logger(), "   Goal TCP: [%.3f, %.3f, %.3f]",
      goal_pose.position.x, goal_pose.position.y, goal_pose.position.z);
    RCLCPP_INFO(get_logger(), "   Direction: [%.3f, %.3f, %.3f]", 
      direction.x(), direction.y(), direction.z());

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool planned = false;

    if (tryCartesianPath(goal_pose, plan)) {
      planned = true;
    }
    else if (tryJointSpacePlan(goal_state, plan)) {
      planned = true;
    }
    
    if (!planned) {
      RCLCPP_ERROR(get_logger(), "❌ Planning failed");
      
      if (attempt_num == 1) {
        RCLCPP_WARN(get_logger(), "   Planning failed - attempting recovery and retry...");
        if (recoverToSafePosition()) {
          return attemptTargetWithRecovery(target, 2);
        }
      }
      
      return false;
    }

    RCLCPP_INFO(get_logger(), "   Executing...");
    auto exec_result = move_group_->execute(plan);
    
    if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "❌ Execution failed: %d", exec_result.val);
      return false;
    }

    rclcpp::sleep_for(std::chrono::milliseconds(150));
    
    double error = verifyBeamTip(target, current_beam_length_);
    
    if (error < beam_error_threshold_) {
      RCLCPP_INFO(get_logger(), "✅ SUCCESS! Error: %.1fmm", error * 1000);
      return true;
    } else {
      RCLCPP_WARN(get_logger(), "⚠️ Large error: %.1fmm > %.1fmm threshold",
        error * 1000, beam_error_threshold_ * 1000);
      
      if (error > 0.15 && attempt_num == 1) {
        RCLCPP_WARN(get_logger(), "   Very large error - attempting recovery...");
        recoverToSafePosition();
      }
      
      return false;
    }
  }

  bool executeTarget(const tf2::Vector3& target)
  {
    total_targets_++;
    
    tf2::Vector3 tcp_pos;
    tf2::Quaternion tcp_orient;
    getTool0Pose(tcp_pos, tcp_orient);
    
    RCLCPP_INFO(get_logger(), "🎯 Target %d: [%.3f, %.3f, %.3f]", 
      total_targets_, target.x(), target.y(), target.z());
    RCLCPP_INFO(get_logger(), "   Current TCP: [%.3f, %.3f, %.3f]", 
      tcp_pos.x(), tcp_pos.y(), tcp_pos.z());

    if (consecutive_failures_ >= MAX_CONSECUTIVE_FAILURES) {
      RCLCPP_WARN(get_logger(), "⚠️ %d consecutive failures - forcing recovery",
        consecutive_failures_);
      if (recoverToSafePosition()) {
        consecutive_failures_ = 0;
        getTool0Pose(tcp_pos, tcp_orient);
        RCLCPP_INFO(get_logger(), "   Recovered TCP: [%.3f, %.3f, %.3f]", 
          tcp_pos.x(), tcp_pos.y(), tcp_pos.z());
      } else {
        RCLCPP_ERROR(get_logger(), "❌ Recovery failed - skipping target");
        publishFailed();
        return false;
      }
    }

    if (tcp_pos.z() < -0.1) {
      RCLCPP_WARN(get_logger(), "⚠️ TCP Z=%.3f is very low - recovering", tcp_pos.z());
      if (recoverToSafePosition()) {
        consecutive_failures_ = 0;
        getTool0Pose(tcp_pos, tcp_orient);
      }
    }

    bool success = attemptTargetWithRecovery(target);
    
    if (success) {
      successful_targets_++;
      consecutive_failures_ = 0;
      RCLCPP_INFO(get_logger(), "✅ Overall success rate: %.1f%%",
        100.0 * successful_targets_ / total_targets_);
      
      std_msgs::msg::Bool msg;
      msg.data = true;
      beam_done_pub_->publish(msg);
      return true;
    } else {
      consecutive_failures_++;
      publishFailed();
      return false;
    }
  }

  void publishFailed()
  {
    std_msgs::msg::Bool msg;
    msg.data = true;
    beam_failed_pub_->publish(msg);
  }

  void targetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (!moveit_ready_) {
      RCLCPP_WARN(get_logger(), "⏳ MoveIt not ready");
      return;
    }

    tf2::Vector3 target;
    
    if (msg->header.frame_id == BASE_FRAME || msg->header.frame_id.empty()) {
      target.setValue(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    } else {
      try {
        auto transformed = tf_buffer_->transform(*msg, BASE_FRAME, tf2::durationFromSec(0.5));
        target.setValue(transformed.pose.position.x, transformed.pose.position.y, transformed.pose.position.z);
      } catch (const tf2::TransformException& e) {
        RCLCPP_ERROR(get_logger(), "Transform error: %s", e.what());
        publishFailed();
        return;
      }
    }

    executeTarget(target);
  }

  void multiTargetCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "📦 Received %zu targets", msg->poses.size());
    
    for (const auto& pose : msg->poses) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = msg->header;
      ps.pose = pose;
      targetCallback(std::make_shared<geometry_msgs::msg::PoseStamped>(ps));
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RobustBeamPointingNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}