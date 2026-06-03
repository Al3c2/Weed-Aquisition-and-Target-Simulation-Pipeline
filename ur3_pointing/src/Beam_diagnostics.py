// beam_pointing_fixed.cpp

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

// MoveIt2
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/msg/collision_object.hpp>

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

static const char WORLD_FRAME[] = "world";
static const char EE_LINK[]     = "tool0";
static const char GROUP_NAME[]  = "ur3_manipulator";

// === CRITICAL: Joint angle normalization to prevent 2π wrap-around errors ===
inline double normalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

inline double closestAngle(double angle, double reference) {
  double normalized = normalizeAngle(angle);
  double options[3] = { normalized, normalized + 2.0 * M_PI, normalized - 2.0 * M_PI };
  double best = normalized;
  double best_diff = std::fabs(normalized - reference);
  for (int i = 1; i < 3; i++) {
    double diff = std::fabs(options[i] - reference);
    if (diff < best_diff) { best_diff = diff; best = options[i]; }
  }
  return best;
}

class OptimizedBeamPointingNode : public rclcpp::Node
{
public:
  OptimizedBeamPointingNode()
  : Node("Beam_pointing_precise"),
    tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
    tf_listener_(*tf_buffer_)
  {
    RCLCPP_INFO(this->get_logger(), "🚀 Optimized Beam Pointing Node starting...");

    // === Beam parameters ===
    beam_length_ = this->declare_parameter<double>("beam_length", 0.65);
    ray_L_min_   = this->declare_parameter<double>("ray_L_min", 0.55);  // Wider range
    ray_L_max_   = this->declare_parameter<double>("ray_L_max", 0.75);
    ray_L_step_  = this->declare_parameter<double>("ray_L_step", 0.01);

    // === Performance parameters ===
    use_cartesian_path_ = this->declare_parameter<bool>("use_cartesian_path", true);
    cartesian_step_size_ = this->declare_parameter<double>("cartesian_step_size", 0.01);
    cartesian_jump_threshold_ = this->declare_parameter<double>("cartesian_jump_threshold", 0.0);
    
    enable_target_queue_ = this->declare_parameter<bool>("enable_target_queue", true);
    max_queue_size_ = this->declare_parameter<int>("max_queue_size", 10);
    
    adaptive_velocity_ = this->declare_parameter<bool>("adaptive_velocity", true);
    min_velocity_scale_ = this->declare_parameter<double>("min_velocity_scale", 0.8);
    max_velocity_scale_ = this->declare_parameter<double>("max_velocity_scale", 1.5);
    
    planning_timeout_ = this->declare_parameter<double>("planning_timeout", 5.0);
    num_planning_attempts_ = this->declare_parameter<int>("num_planning_attempts", 10);

    // === Collision parameters ===
    enable_swincar_collision_ = this->declare_parameter<bool>("enable_swincar_collision", true);
    swincar_mesh_uri_ = this->declare_parameter<std::string>(
        "swincar_mesh_uri",
        "file:///home/alex/.ignition/gazebo/models/swincar/meshes/swincar_collision.dae");
    swincar_pose_xyz_ = this->declare_parameter<std::vector<double>>(
        "swincar_pose_xyz", std::vector<double>{0.0, 0.0, 0.0});
    swincar_pose_rpy_ = this->declare_parameter<std::vector<double>>(
        "swincar_pose_rpy", std::vector<double>{0.0, 0.0, 0.0});
    
    enable_ground_collision_ = this->declare_parameter<bool>("enable_ground_collision", true);
    ground_clearance_ = this->declare_parameter<double>("ground_clearance", 0.02);
    spawn_height_ = this->declare_parameter<double>("spawn_height", 0.35);
    
    // === GROUND TRUTH POSE (from Gazebo) ===
    ground_truth_topic_ = this->declare_parameter<std::string>(
        "ground_truth_topic", "/model/swincar_ur3/pose");
    ground_truth_frame_id_ = this->declare_parameter<std::string>(
        "ground_truth_frame_id", "empty");
    
    // === Beam-done and beam-failed topics ===
    std::string arm_done_topic = this->declare_parameter<std::string>("arm_done_topic", "/beam_task_done");
    std::string arm_failed_topic = this->declare_parameter<std::string>("arm_failed_topic", "/beam_task_failed");
    beam_done_pub_ = this->create_publisher<std_msgs::msg::Bool>(arm_done_topic, 10);
    beam_failed_pub_ = this->create_publisher<std_msgs::msg::Bool>(arm_failed_topic, 10);
    
    // Error threshold
    beam_error_threshold_ = this->declare_parameter<double>("beam_error_threshold", 0.03);
    RCLCPP_INFO(this->get_logger(), "📊 Beam error threshold: %.1fmm", beam_error_threshold_ * 1000.0);

    // === SUBSCRIPTIONS FIRST - BEFORE MOVEIT ===
    single_target_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "target_pose", 10,
      std::bind(&OptimizedBeamPointingNode::singleTargetCallback, this, std::placeholders::_1));

    multi_target_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
      "target_poses", 10,
      std::bind(&OptimizedBeamPointingNode::multiTargetCallback, this, std::placeholders::_1));

    // === GROUND TRUTH SUBSCRIPTION (for accurate collision tracking) ===
    ground_truth_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      ground_truth_topic_, 10,
      std::bind(&OptimizedBeamPointingNode::groundTruthCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "📡 Subscribed to 'target_pose' and 'target_poses'");
    RCLCPP_INFO(this->get_logger(), "📡 Subscribed to ground truth: %s (frame_id filter: %s)", 
      ground_truth_topic_.c_str(), ground_truth_frame_id_.c_str());

    // === DELAYED MOVEIT INITIALIZATION ===
    RCLCPP_INFO(this->get_logger(), "⏳ Will initialize MoveIt in 2 seconds...");
    init_timer_ = this->create_wall_timer(
      std::chrono::seconds(2),
      std::bind(&OptimizedBeamPointingNode::initializeMoveIt, this));
  }

private:
  // === Parameters ===
  double beam_length_, ray_L_min_, ray_L_max_, ray_L_step_;
  bool use_cartesian_path_, enable_target_queue_, adaptive_velocity_;
  double cartesian_step_size_, cartesian_jump_threshold_;
  int max_queue_size_, num_planning_attempts_;
  double min_velocity_scale_, max_velocity_scale_, planning_timeout_;
  
  bool enable_swincar_collision_, enable_ground_collision_;
  std::string swincar_mesh_uri_;
  std::vector<double> swincar_pose_xyz_, swincar_pose_rpy_;
  double ground_clearance_, spawn_height_;
  
  std::string ground_truth_topic_, ground_truth_frame_id_;

  // === ROS/MoveIt ===
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
  
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr single_target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr multi_target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ground_truth_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr beam_done_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr beam_failed_pub_;
  double beam_error_threshold_;
  
  rclcpp::TimerBase::SharedPtr init_timer_;
  rclcpp::TimerBase::SharedPtr collision_update_timer_;
  bool moveit_ready_ = false;
  
  // Ground truth pose from Gazebo (most accurate!)
  geometry_msgs::msg::Pose ground_truth_pose_;
  bool ground_truth_received_ = false;
  std::mutex pose_mutex_;
  
  // Track last collision update position
  geometry_msgs::msg::Pose last_collision_pose_;
  bool collision_initialized_ = false;
  
  std::deque<geometry_msgs::msg::PoseStamped> pending_targets_;
  std::mutex targets_mutex_;
  
  // Failure tracking
  int consecutive_failures_ = 0;
  static const int MAX_CONSECUTIVE_FAILURES = 3;

  // === GROUND TRUTH CALLBACK ===
  void groundTruthCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (msg->header.frame_id != ground_truth_frame_id_) {
      return;
    }
    
    static int log_count = 0;
    if (log_count < 3) {
      RCLCPP_INFO(this->get_logger(), "📍 Ground truth: [%.2f, %.2f, %.2f] frame='%s'",
        msg->pose.position.x, msg->pose.position.y, msg->pose.position.z,
        msg->header.frame_id.c_str());
      log_count++;
    }
    
    std::lock_guard<std::mutex> lock(pose_mutex_);
    ground_truth_pose_ = msg->pose;
    ground_truth_received_ = true;
  }

  // === Get current robot base position in world frame ===
  bool getRobotWorldPosition(geometry_msgs::msg::Pose& pose_out)
  {
    // Prefer ground truth
    {
      std::lock_guard<std::mutex> lock(pose_mutex_);
      if (ground_truth_received_) {
        pose_out = ground_truth_pose_;
        return true;
      }
    }
    
    // Fallback to TF
    try {
      auto transform = tf_buffer_->lookupTransform(
        WORLD_FRAME, "base_link", tf2::TimePointZero, tf2::durationFromSec(0.1));
      pose_out.position.x = transform.transform.translation.x;
      pose_out.position.y = transform.transform.translation.y;
      pose_out.position.z = transform.transform.translation.z;
      pose_out.orientation = transform.transform.rotation;
      return true;
    } catch (const tf2::TransformException& ex) {
      return false;
    }
  }

  // === Transform point from base_link to world using GROUND TRUTH ===
  tf2::Vector3 transformToWorld(const tf2::Vector3& point_base_link)
  {
    geometry_msgs::msg::Pose robot_pose;
    if (!getRobotWorldPosition(robot_pose)) {
      RCLCPP_WARN(this->get_logger(), "Could not get robot position for transform!");
      return point_base_link;
    }
    
    // Get robot orientation
    tf2::Quaternion robot_q;
    tf2::fromMsg(robot_pose.orientation, robot_q);
    tf2::Matrix3x3 R(robot_q);
    
    // Rotate the point by robot orientation
    tf2::Vector3 rotated(
      R[0][0] * point_base_link.x() + R[0][1] * point_base_link.y() + R[0][2] * point_base_link.z(),
      R[1][0] * point_base_link.x() + R[1][1] * point_base_link.y() + R[1][2] * point_base_link.z(),
      R[2][0] * point_base_link.x() + R[2][1] * point_base_link.y() + R[2][2] * point_base_link.z()
    );
    
    // Add robot position (translation)
    tf2::Vector3 world_point(
      rotated.x() + robot_pose.position.x,
      rotated.y() + robot_pose.position.y,
      rotated.z() + robot_pose.position.z
    );
    
    return world_point;
  }

  void initializeMoveIt()
  {
    init_timer_->cancel();
    
    RCLCPP_INFO(this->get_logger(), "🔧 Initializing MoveIt...");
    
    try {
      move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), GROUP_NAME);
      
      planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
      
      // === MoveIt configuration ===
      move_group_->setEndEffectorLink(EE_LINK);
      move_group_->setPoseReferenceFrame(WORLD_FRAME);
      move_group_->setPlanningTime(planning_timeout_);
      move_group_->setPlannerId("RRTConnectkConfigDefault");
      move_group_->setNumPlanningAttempts(num_planning_attempts_);
      move_group_->setGoalPositionTolerance(0.005);
      move_group_->setGoalOrientationTolerance(0.02);
      move_group_->startStateMonitor();
      move_group_->allowReplanning(true);

      RCLCPP_INFO(this->get_logger(), "⏳ Waiting for robot state...");
      if (!move_group_->getCurrentState(5.0)) {
        RCLCPP_WARN(this->get_logger(), "Could not get robot state, continuing anyway...");
      }

      // === Wait for valid ground truth OR valid TF ===
      RCLCPP_INFO(this->get_logger(), "📍 Waiting for valid robot position...");
      
      bool got_valid_position = false;
      int max_attempts = 50;
      
      for (int attempt = 0; attempt < max_attempts && !got_valid_position; attempt++) {
        {
          std::lock_guard<std::mutex> lock(pose_mutex_);
          if (ground_truth_received_) {
            double pos_magnitude = std::sqrt(
              ground_truth_pose_.position.x * ground_truth_pose_.position.x +
              ground_truth_pose_.position.y * ground_truth_pose_.position.y +
              ground_truth_pose_.position.z * ground_truth_pose_.position.z);
            
            if (spawn_height_ < 0.1 || pos_magnitude > 0.05) {
              RCLCPP_INFO(this->get_logger(), "📍 Got ground truth position: [%.2f, %.2f, %.2f]",
                ground_truth_pose_.position.x, ground_truth_pose_.position.y, ground_truth_pose_.position.z);
              got_valid_position = true;
              break;
            }
          }
        }
        
        try {
          auto base_transform = tf_buffer_->lookupTransform(
            WORLD_FRAME, "base_link", tf2::TimePointZero, tf2::durationFromSec(0.1));
          
          double x = base_transform.transform.translation.x;
          double y = base_transform.transform.translation.y;
          double z = base_transform.transform.translation.z;
          
          double pos_magnitude = std::sqrt(x*x + y*y + z*z);
          if (spawn_height_ < 0.1 || pos_magnitude > 0.05 || z > 0.05) {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            ground_truth_pose_.position.x = x;
            ground_truth_pose_.position.y = y;
            ground_truth_pose_.position.z = z;
            ground_truth_pose_.orientation = base_transform.transform.rotation;
            ground_truth_received_ = true;
            
            RCLCPP_INFO(this->get_logger(), "📍 Got TF position: [%.2f, %.2f, %.2f]", x, y, z);
            got_valid_position = true;
            break;
          }
        } catch (const tf2::TransformException&) {
          // TF not ready yet
        }
        
        RCLCPP_DEBUG(this->get_logger(), "⏳ Waiting for valid position (attempt %d/%d)...", 
          attempt + 1, max_attempts);
        rclcpp::sleep_for(std::chrono::milliseconds(100));
      }
      
      if (!got_valid_position) {
        RCLCPP_ERROR(this->get_logger(), "❌ Could not get valid robot position after 5 seconds!");
        RCLCPP_ERROR(this->get_logger(), "   This will cause incorrect beam pointing. Retrying...");
        init_timer_ = this->create_wall_timer(
          std::chrono::seconds(2),
          std::bind(&OptimizedBeamPointingNode::initializeMoveIt, this));
        return;
      }
      
      {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        last_collision_pose_ = ground_truth_pose_;
      }
      
      if (enable_swincar_collision_) addSwincarCollision();
      if (enable_ground_collision_) addGroundCollision();
      collision_initialized_ = true;

      moveit_ready_ = true;
      
      RCLCPP_INFO(this->get_logger(),
        "✅ Ready! Beam: %.2fm, L_range: [%.2f, %.2f], Cartesian: %s",
        beam_length_, ray_L_min_, ray_L_max_, use_cartesian_path_ ? "ON" : "OFF");

      collision_update_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(200),
        std::bind(&OptimizedBeamPointingNode::updateCollisionObjects, this));
      
      RCLCPP_INFO(this->get_logger(), "🔄 Started dynamic collision update timer (200ms)");

      processPendingTargets();
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "❌ MoveIt init failed: %s", e.what());
      RCLCPP_INFO(this->get_logger(), "Retrying in 2 seconds...");
      init_timer_ = this->create_wall_timer(
        std::chrono::seconds(2),
        std::bind(&OptimizedBeamPointingNode::initializeMoveIt, this));
    }
  }

  void processPendingTargets()
  {
    std::lock_guard<std::mutex> lock(targets_mutex_);
    RCLCPP_INFO(this->get_logger(), "Processing %zu pending targets...", pending_targets_.size());
    while (!pending_targets_.empty()) {
      auto target = pending_targets_.front();
      pending_targets_.pop_front();
      executeBeamTarget(target);
    }
  }

  void updateCollisionObjects()
  {
    if (!moveit_ready_ || !planning_scene_interface_ || !collision_initialized_) return;
    
    geometry_msgs::msg::Pose current_pose;
    if (!getRobotWorldPosition(current_pose)) return;
    
    double dx = current_pose.position.x - last_collision_pose_.position.x;
    double dy = current_pose.position.y - last_collision_pose_.position.y;
    double dz = current_pose.position.z - last_collision_pose_.position.z;
    double dist_moved = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    tf2::Quaternion q1, q2;
    tf2::fromMsg(current_pose.orientation, q1);
    tf2::fromMsg(last_collision_pose_.orientation, q2);
    double angle_diff = q1.angleShortestPath(q2);
    
    if (dist_moved < 0.03 && angle_diff < 0.05) {
      return;
    }
    
    last_collision_pose_ = current_pose;
    
    RCLCPP_DEBUG(this->get_logger(), "🔄 Updating collisions at [%.2f, %.2f, %.2f]",
      current_pose.position.x, current_pose.position.y, current_pose.position.z);
    
    updateGroundCollision(current_pose);
    if (enable_swincar_collision_) {
      updateSwincarCollision(current_pose);
    }
  }
  
  void updateGroundCollision(const geometry_msgs::msg::Pose& robot_pose)
  {
    moveit_msgs::msg::CollisionObject ground_plane;
    ground_plane.id = "ground_plane";
    ground_plane.operation = ground_plane.MOVE;
    ground_plane.header.frame_id = WORLD_FRAME;
    ground_plane.header.stamp = this->now();

    double ground_z = ground_clearance_;

    geometry_msgs::msg::Pose box_pose;
    box_pose.position.x = robot_pose.position.x;
    box_pose.position.y = robot_pose.position.y;
    box_pose.position.z = ground_z - 0.05;
    box_pose.orientation.w = 1.0;

    ground_plane.primitive_poses.push_back(box_pose);
    planning_scene_interface_->applyCollisionObjects({ground_plane});
  }
  
  void updateSwincarCollision(const geometry_msgs::msg::Pose& robot_pose)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = "swincar";
    obj.operation = obj.MOVE;
    obj.header.frame_id = WORLD_FRAME;
    obj.header.stamp = this->now();

    geometry_msgs::msg::Pose pose;
    pose.position.x = robot_pose.position.x + 
      (swincar_pose_xyz_.size() > 0 ? swincar_pose_xyz_[0] : 0.0);
    pose.position.y = robot_pose.position.y + 
      (swincar_pose_xyz_.size() > 1 ? swincar_pose_xyz_[1] : 0.0);
    pose.position.z = robot_pose.position.z +
      (swincar_pose_xyz_.size() > 2 ? swincar_pose_xyz_[2] : 0.0);

    tf2::Quaternion robot_q;
    tf2::fromMsg(robot_pose.orientation, robot_q);
    
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

  void addSwincarCollision()
  {
    RCLCPP_INFO(this->get_logger(), "🔧 Loading Swincar mesh from: %s", swincar_mesh_uri_.c_str());
    
    auto mesh = std::unique_ptr<shapes::Mesh>(
        shapes::createMeshFromResource(swincar_mesh_uri_));
    if (!mesh) {
      RCLCPP_ERROR(this->get_logger(), "❌ Could not load Swincar mesh from: %s", swincar_mesh_uri_.c_str());
      return;
    }
    
    RCLCPP_INFO(this->get_logger(), "   Mesh loaded: %u vertices, %u triangles", 
      mesh->vertex_count, mesh->triangle_count);

    shapes::ShapeMsg shape_msg;
    shapes::constructMsgFromShape(mesh.get(), shape_msg);
    shape_msgs::msg::Mesh mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg);

    moveit_msgs::msg::CollisionObject obj;
    obj.id = "swincar";
    obj.operation = obj.ADD;
    obj.header.frame_id = WORLD_FRAME;
    obj.header.stamp = this->now();

    geometry_msgs::msg::Pose pose;
    {
      std::lock_guard<std::mutex> lock(pose_mutex_);
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
    }

    RCLCPP_INFO(this->get_logger(), "   Swincar pose: [%.2f, %.2f, %.2f]",
      pose.position.x, pose.position.y, pose.position.z);

    obj.meshes = {mesh_msg};
    obj.mesh_poses = {pose};

    planning_scene_interface_->applyCollisionObjects({obj});
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    RCLCPP_INFO(this->get_logger(), "🧱 Swincar collision mesh added to planning scene");
  }

  void addGroundCollision()
  {
    moveit_msgs::msg::CollisionObject ground_plane;
    ground_plane.id = "ground_plane";
    ground_plane.operation = ground_plane.ADD;
    ground_plane.header.frame_id = WORLD_FRAME;
    ground_plane.header.stamp = this->now();

    double box_height = 0.1;
    double ground_z = ground_clearance_;
    
    geometry_msgs::msg::Pose robot_pose;
    {
      std::lock_guard<std::mutex> lock(pose_mutex_);
      robot_pose = ground_truth_pose_;
    }
    
    RCLCPP_INFO(this->get_logger(), "🔧 Adding ground plane at robot pos [%.2f, %.2f], ground_z=%.2f",
      robot_pose.position.x, robot_pose.position.y, ground_z);
    
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[primitive.BOX_X] = 20.0;
    primitive.dimensions[primitive.BOX_Y] = 20.0;
    primitive.dimensions[primitive.BOX_Z] = box_height;

    geometry_msgs::msg::Pose box_pose;
    box_pose.position.x = robot_pose.position.x;
    box_pose.position.y = robot_pose.position.y;
    box_pose.position.z = ground_z - box_height / 2.0;
    box_pose.orientation.w = 1.0;

    ground_plane.primitives.push_back(primitive);
    ground_plane.primitive_poses.push_back(box_pose);

    planning_scene_interface_->applyCollisionObjects({ground_plane});
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    RCLCPP_INFO(this->get_logger(), 
      "🛡️  Ground collision plane added - top surface at Z=%.3fm", ground_z);
  }

  tf2::Quaternion alignZToDirection(const tf2::Vector3 &dir_in)
  {
    tf2::Vector3 z_axis(0.0, 0.0, 1.0);
    tf2::Vector3 v = dir_in.normalized();
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
    std::sort(Ls.begin(), Ls.end(),
              [this](double a, double b) {
                return std::fabs(a - beam_length_) < std::fabs(b - beam_length_);
              });
    return Ls;
  }

  bool computeIKWithSeed(const geometry_msgs::msg::Pose &target_pose,
                         moveit::core::RobotState &result_state,
                         double timeout = 0.5)
  {
    moveit::core::RobotModelConstPtr kmodel = move_group_->getRobotModel();
    if (!kmodel) return false;

    const auto* jmg = kmodel->getJointModelGroup(GROUP_NAME);
    if (!jmg) return false;

    auto current_state = move_group_->getCurrentState();
    result_state = *current_state;
    
    // Try multiple times with current seed
    for (int attempt = 0; attempt < 3; attempt++) {
      bool ik_success = result_state.setFromIK(jmg, target_pose, EE_LINK, timeout);
      if (ik_success) {
        normalizeJointState(result_state, *current_state);
        result_state.update();
        return true;
      }
    }
    
    // Try with different seeds
    std::vector<std::vector<double>> seeds = {
      {0.0, -1.57, 0.0, -1.57, -1.57, 0.0},
      {0.5, -1.57, 0.5, -1.57, -1.57, 0.0},
      {-0.5, -1.57, 0.5, -1.57, -1.57, 0.0},
      {0.0, -2.0, 1.0, -0.57, -1.57, 0.0},
      {1.0, -1.57, 0.5, -1.0, -1.57, 0.0},
    };
    
    for (const auto& seed : seeds) {
      result_state.setJointGroupPositions(jmg, seed);
      result_state.update();
      
      if (result_state.setFromIK(jmg, target_pose, EE_LINK, timeout)) {
        RCLCPP_DEBUG(this->get_logger(), "   IK succeeded with alternate seed");
        normalizeJointState(result_state, *current_state);
        result_state.update();
        return true;
      }
    }
    
    return false;
  }

  void normalizeJointState(moveit::core::RobotState &target_state,
                           const moveit::core::RobotState &current_state)
  {
    const auto* jmg = target_state.getJointModelGroup(GROUP_NAME);
    if (!jmg) return;

    std::vector<double> target_values;
    target_state.copyJointGroupPositions(jmg, target_values);
    
    std::vector<double> current_values;
    current_state.copyJointGroupPositions(jmg, current_values);

    bool modified = false;
    
    for (size_t i = 0; i < target_values.size() && i < current_values.size(); i++) {
      double current = current_values[i];
      double target = target_values[i];
      double diff = std::fabs(target - current);
      
      if (diff > M_PI) {
        double new_target = closestAngle(target, current);
        if (std::fabs(new_target - target) > 0.01) {
          target_values[i] = new_target;
          modified = true;
        }
      }
    }

    if (modified) {
      target_state.setJointGroupPositions(jmg, target_values);
      RCLCPP_INFO(this->get_logger(), "✓ Joint state normalized to avoid 2π wrap-around");
    }
  }

  double computeVelocityScale(const geometry_msgs::msg::Pose &current,
                              const geometry_msgs::msg::Pose &target)
  {
    if (!adaptive_velocity_) return max_velocity_scale_;

    double dx = target.position.x - current.position.x;
    double dy = target.position.y - current.position.y;
    double dz = target.position.z - current.position.z;
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    double scale = min_velocity_scale_ + 
                   (max_velocity_scale_ - min_velocity_scale_) * 
                   std::min(dist / 0.3, 1.0);

    return std::clamp(scale, min_velocity_scale_, max_velocity_scale_);
  }

  bool planCartesianPath(const geometry_msgs::msg::Pose &target_pose,
                         moveit::planning_interface::MoveGroupInterface::Plan &plan,
                         double /* velocity_scale */)
  {
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = move_group_->computeCartesianPath(
        waypoints, cartesian_step_size_, cartesian_jump_threshold_,
        trajectory, true);

    RCLCPP_INFO(this->get_logger(), "   Cartesian path: %.0f%% complete", fraction * 100);

    if (fraction < 0.95) {
      RCLCPP_WARN(this->get_logger(), "⚠️ Cartesian path blocked (%.0f%%)", fraction * 100);
      return false;
    }

    robot_trajectory::RobotTrajectory rt(move_group_->getRobotModel(), GROUP_NAME);
    rt.setRobotTrajectoryMsg(*move_group_->getCurrentState(), trajectory);
    
    double safe_vel = 0.20;
    double safe_acc = 0.15;
    
    trajectory_processing::IterativeParabolicTimeParameterization iptp;
    if (!iptp.computeTimeStamps(rt, safe_vel, safe_acc)) {
      RCLCPP_WARN(this->get_logger(), "⚠️ Time parameterization failed");
      return false;
    }

    rt.getRobotTrajectoryMsg(plan.trajectory_);
    plan.planning_time_ = 0.0;
    
    normalizeTrajectory(plan.trajectory_.joint_trajectory);
    
    if (!plan.trajectory_.joint_trajectory.points.empty()) {
      double duration = plan.trajectory_.joint_trajectory.points.back().time_from_start.sec +
        plan.trajectory_.joint_trajectory.points.back().time_from_start.nanosec * 1e-9;
      RCLCPP_INFO(this->get_logger(), "   Trajectory duration: %.2f seconds", duration);
      
      const double MAX_TRAJECTORY_DURATION = 8.0;
      if (duration > MAX_TRAJECTORY_DURATION) {
        RCLCPP_WARN(this->get_logger(), "⚠️ Trajectory too long (%.1fs > %.1fs max) - rejecting",
          duration, MAX_TRAJECTORY_DURATION);
        return false;
      }
    }
    
    return true;
  }

  void normalizeTrajectory(trajectory_msgs::msg::JointTrajectory &trajectory)
  {
    if (trajectory.points.empty()) return;
    
    auto current_state = move_group_->getCurrentState();
    const auto* jmg = current_state->getJointModelGroup(GROUP_NAME);
    if (!jmg) return;
    
    std::vector<double> current_values;
    current_state->copyJointGroupPositions(jmg, current_values);
    
    bool any_normalized = false;
    std::vector<double> reference = current_values;
    
    for (auto &point : trajectory.points) {
      for (size_t i = 0; i < point.positions.size() && i < reference.size(); i++) {
        double target = point.positions[i];
        double ref = reference[i];
        double diff = std::fabs(target - ref);
        
        if (diff > M_PI) {
          double new_target = closestAngle(target, ref);
          if (std::fabs(new_target - target) > 0.01) {
            point.positions[i] = new_target;
            any_normalized = true;
          }
        }
      }
      reference = point.positions;
    }
    
    if (any_normalized) {
      RCLCPP_INFO(this->get_logger(), "✓ Trajectory normalized to avoid 2π wrap-around");
    }
  }

  bool planJointSpace(const moveit::core::RobotState &target_state,
                      moveit::planning_interface::MoveGroupInterface::Plan &plan,
                      double /* velocity_scale */)
  {
    move_group_->setStartStateToCurrentState();
    move_group_->setJointValueTarget(target_state);
    
    move_group_->setMaxVelocityScalingFactor(0.20);
    move_group_->setMaxAccelerationScalingFactor(0.15);

    auto code = move_group_->plan(plan);
    
    if (code == moveit::core::MoveItErrorCode::SUCCESS) {
      normalizeTrajectory(plan.trajectory_.joint_trajectory);
      RCLCPP_INFO(this->get_logger(), "   Joint space plan: SUCCESS");
      return true;
    } else {
      RCLCPP_WARN(this->get_logger(), "   Joint space plan FAILED: %d", code.val);
      return false;
    }
  }

  bool computeGoalPose(const tf2::Vector3& target, double L,
                       geometry_msgs::msg::Pose& goal_pose)
  {
    auto curr_pose = move_group_->getCurrentPose(EE_LINK).pose;
    tf2::Vector3 P_curr(curr_pose.position.x, curr_pose.position.y, curr_pose.position.z);

    tf2::Vector3 dir = (target - P_curr);
    double dist = dir.length();
    if (dist < 0.01) {
      RCLCPP_WARN(this->get_logger(), "Target too close to current TCP");
      return false;
    }
    dir /= dist;

    tf2::Vector3 P_goal;
    for (int i = 0; i < 5; ++i) {
      P_goal = target - dir * L;
      tf2::Vector3 new_dir = (target - P_goal);
      double new_dist = new_dir.length();
      if (new_dist < 0.001) break;
      new_dir /= new_dist;
      if ((new_dir - dir).length() < 0.0001) break;
      dir = new_dir;
    }

    // === Workspace check ===
    geometry_msgs::msg::Pose robot_pose;
    if (getRobotWorldPosition(robot_pose)) {
      double dx = P_goal.x() - robot_pose.position.x;
      double dy = P_goal.y() - robot_pose.position.y;
      double dz = P_goal.z() - robot_pose.position.z;
      double horiz_dist = std::sqrt(dx*dx + dy*dy);
      
      // UR3 workspace limits (approximate)
      if (horiz_dist > 0.55 || horiz_dist < 0.10 || dz < 0.05 || dz > 0.85) {
        RCLCPP_DEBUG(this->get_logger(), 
          "   TCP [%.3f, %.3f, %.3f] may be outside workspace (horiz=%.2f, dz=%.2f)",
          P_goal.x(), P_goal.y(), P_goal.z(), horiz_dist, dz);
        // Don't reject - let IK decide
      }
    }

    tf2::Quaternion q = alignZToDirection(dir);

    goal_pose.position.x = P_goal.x();
    goal_pose.position.y = P_goal.y();
    goal_pose.position.z = P_goal.z();
    goal_pose.orientation = tf2::toMsg(q);

    return true;
  }

  bool executeBeamTarget(const geometry_msgs::msg::PoseStamped &spot_world)
  {
    tf2::Vector3 T(spot_world.pose.position.x,
                   spot_world.pose.position.y,
                   spot_world.pose.position.z);

    RCLCPP_INFO(this->get_logger(), "🎯 Processing target [%.3f, %.3f, %.3f] (frame: %s)",
      T.x(), T.y(), T.z(), spot_world.header.frame_id.c_str());

    if (isCurrentStateInCollision()) {
      RCLCPP_WARN(this->get_logger(), "⚠️ Current arm state is in collision! Attempting recovery...");
      if (!recoverToSafePosition()) {
        RCLCPP_ERROR(this->get_logger(), "❌ Could not recover from collision state");
        std_msgs::msg::Bool fail_msg;
        fail_msg.data = true;
        beam_failed_pub_->publish(fail_msg);
        return false;
      }
      RCLCPP_INFO(this->get_logger(), "✅ Recovered to safe position");
    }

    auto lengths = candidateLengths();
    geometry_msgs::msg::Pose goal_pose;
    moveit::core::RobotState goal_state(move_group_->getRobotModel());
    bool found_ik = false;
    double chosen_L = 0.0;

    for (double L : lengths) {
      if (!computeGoalPose(T, L, goal_pose)) continue;

      if (computeIKWithSeed(goal_pose, goal_state, 0.5)) {
        chosen_L = L;
        found_ik = true;
        RCLCPP_INFO(this->get_logger(), "   Found IK for L=%.3fm, TCP: [%.3f, %.3f, %.3f]", 
          L, goal_pose.position.x, goal_pose.position.y, goal_pose.position.z);
        break;
      }
    }

    if (!found_ik) {
      RCLCPP_WARN(this->get_logger(), "❌ No valid IK solution for target [%.2f, %.2f, %.2f]",
                  T.x(), T.y(), T.z());
      
      std_msgs::msg::Bool fail_msg;
      fail_msg.data = true;
      beam_failed_pub_->publish(fail_msg);
      RCLCPP_WARN(this->get_logger(), "📡 Published /beam_task_failed (no IK solution)");
      
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "📍 Goal TCP: [%.3f, %.3f, %.3f] L=%.3fm",
      goal_pose.position.x, goal_pose.position.y, goal_pose.position.z, chosen_L);

    auto curr_pose = move_group_->getCurrentPose(EE_LINK).pose;
    double vel_scale = computeVelocityScale(curr_pose, goal_pose);
    vel_scale = std::min(vel_scale, 0.5);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool planned = false;

    if (use_cartesian_path_) {
      planned = planCartesianPath(goal_pose, plan, vel_scale);
      if (planned) {
        RCLCPP_INFO(this->get_logger(), "✓ Cartesian path planned");
      } else {
        RCLCPP_WARN(this->get_logger(), "⚠️ Cartesian failed, trying joint space...");
      }
    }

    if (!planned) {
      planned = planJointSpace(goal_state, plan, vel_scale);
      if (planned) {
        RCLCPP_INFO(this->get_logger(), "✓ Joint space plan succeeded");
      }
    }

    if (!planned) {
      RCLCPP_WARN(this->get_logger(), "❌ Planning failed for this target");
      incrementFailureCounter();
      
      std_msgs::msg::Bool fail_msg;
      fail_msg.data = true;
      beam_failed_pub_->publish(fail_msg);
      RCLCPP_WARN(this->get_logger(), "📡 Published /beam_task_failed (planning failed)");
      
      move_group_->clearPoseTargets();
      return false;
    }

    resetFailureCounter();

    RCLCPP_INFO(this->get_logger(), "📍 Executing trajectory...");

    auto exec_code = move_group_->execute(plan);
    bool success = (exec_code == moveit::core::MoveItErrorCode::SUCCESS);

    if (!success) {
      RCLCPP_WARN(this->get_logger(), "❌ Execution failed");
      incrementFailureCounter();
      
      std_msgs::msg::Bool fail_msg;
      fail_msg.data = true;
      beam_failed_pub_->publish(fail_msg);
      RCLCPP_WARN(this->get_logger(), "📡 Published /beam_task_failed (execution failed)");
      
      recoverToSafePosition();
      move_group_->clearPoseTargets();
      return false;
    }

    resetFailureCounter();

    rclcpp::sleep_for(std::chrono::milliseconds(100));
    double error = verifyBeamTip(T);
    
    std_msgs::msg::Bool result_msg;
    result_msg.data = true;
    
    if (error < beam_error_threshold_) {
      beam_done_pub_->publish(result_msg);
      RCLCPP_INFO(this->get_logger(), "✅ SUCCESS: Error %.1fmm < %.1fmm threshold",
                  error * 1000.0, beam_error_threshold_ * 1000.0);
    } else {
      beam_failed_pub_->publish(result_msg);
      RCLCPP_WARN(this->get_logger(), "❌ FAILED: Error %.1fmm >= %.1fmm threshold",
                  error * 1000.0, beam_error_threshold_ * 1000.0);
    }

    move_group_->clearPoseTargets();
    return success;
  }

  double verifyBeamTip(const tf2::Vector3& target)
  {
    try {
      auto ee_transform = tf_buffer_->lookupTransform(
          WORLD_FRAME, EE_LINK, tf2::TimePointZero, tf2::durationFromSec(0.5));
      
      tf2::Vector3 tcp(
          ee_transform.transform.translation.x,
          ee_transform.transform.translation.y,
          ee_transform.transform.translation.z);
      
      tf2::Quaternion quat;
      tf2::fromMsg(ee_transform.transform.rotation, quat);
      tf2::Matrix3x3 R(quat);
      
      tf2::Vector3 beam_dir(R[0][2], R[1][2], R[2][2]);
      beam_dir.normalize();
      
      tf2::Vector3 beam_tip = tcp + beam_dir * beam_length_;
      
      double error = (beam_tip - target).length();
      
      RCLCPP_INFO(this->get_logger(),
        "📍 BEAM TIP [%.3f, %.3f, %.3f] | Target [%.3f, %.3f, %.3f] | Error: %.1fmm",
        beam_tip.x(), beam_tip.y(), beam_tip.z(),
        target.x(), target.y(), target.z(),
        error * 1000.0);
      
      return error;
        
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Beam tip verification failed: %s", ex.what());
      return 1.0;
    }
  }

  bool isCurrentStateInCollision()
  {
    return consecutive_failures_ >= MAX_CONSECUTIVE_FAILURES;
  }
  
  void resetFailureCounter()
  {
    consecutive_failures_ = 0;
  }
  
  void incrementFailureCounter()
  {
    consecutive_failures_++;
    if (consecutive_failures_ >= MAX_CONSECUTIVE_FAILURES) {
      RCLCPP_WARN(this->get_logger(), "🔴 %d consecutive failures - arm likely stuck",
        consecutive_failures_);
    }
  }

  bool recoverToSafePosition()
  {
    RCLCPP_WARN(this->get_logger(), "🔄 Attempting recovery to safe position...");
    
    std::vector<std::vector<double>> recovery_positions = {
      {0.0, -1.57, 0.0, -1.57, -1.57, 0.0},
      {0.0, -2.0, 1.0, -0.57, -1.57, 0.0},
      {1.57, -1.57, 0.0, -1.57, -1.57, 0.0},
    };
    
    for (size_t i = 0; i < recovery_positions.size(); i++) {
      RCLCPP_INFO(this->get_logger(), "🔄 Trying recovery position %zu/%zu...", 
        i + 1, recovery_positions.size());
      
      try {
        move_group_->setStartStateToCurrentState();
        move_group_->setJointValueTarget(recovery_positions[i]);
        move_group_->setMaxVelocityScalingFactor(0.15);
        move_group_->setMaxAccelerationScalingFactor(0.1);
        move_group_->setPlanningTime(3.0);
        move_group_->setNumPlanningAttempts(10);
        
        moveit::planning_interface::MoveGroupInterface::Plan recovery_plan;
        auto result = move_group_->plan(recovery_plan);
        
        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
          RCLCPP_INFO(this->get_logger(), "🔄 Recovery plan found, executing...");
          normalizeTrajectory(recovery_plan.trajectory_.joint_trajectory);
          
          auto exec_result = move_group_->execute(recovery_plan);
          
          if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_INFO(this->get_logger(), "✅ Recovery successful - arm in safe position");
            move_group_->clearPoseTargets();
            move_group_->setPlanningTime(planning_timeout_);
            move_group_->setNumPlanningAttempts(num_planning_attempts_);
            consecutive_failures_ = 0;
            return true;
          }
        }
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Recovery attempt %zu failed: %s", i + 1, e.what());
      }
    }
    
    RCLCPP_ERROR(this->get_logger(), "❌ All recovery attempts failed!");
    move_group_->stop();
    move_group_->clearPoseTargets();
    move_group_->setPlanningTime(planning_timeout_);
    move_group_->setNumPlanningAttempts(num_planning_attempts_);
    
    return false;
  }

  void singleTargetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), 
      "📨 Received target: [%.3f, %.3f, %.3f] frame='%s'",
      msg->pose.position.x, msg->pose.position.y, msg->pose.position.z,
      msg->header.frame_id.c_str());

    if (!moveit_ready_) {
      RCLCPP_WARN(this->get_logger(), "⏳ MoveIt not ready, queuing target...");
      std::lock_guard<std::mutex> lock(targets_mutex_);
      pending_targets_.push_back(*msg);
      return;
    }

    if (enable_target_queue_ && !pending_targets_.empty()) {
      std::lock_guard<std::mutex> lock(targets_mutex_);
      if ((int)pending_targets_.size() < max_queue_size_) {
        pending_targets_.push_back(*msg);
        RCLCPP_DEBUG(this->get_logger(), "Added to queue (size: %zu)", pending_targets_.size());
      }
      return;
    }

    geometry_msgs::msg::PoseStamped spot_world;
    
    // === FIXED: Use ground truth for transformation ===
    if (msg->header.frame_id == "base_link") {
      // Transform from base_link to world using ground truth
      tf2::Vector3 point_base(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
      tf2::Vector3 point_world = transformToWorld(point_base);
      
      spot_world.header.frame_id = WORLD_FRAME;
      spot_world.header.stamp = msg->header.stamp;
      spot_world.pose.position.x = point_world.x();
      spot_world.pose.position.y = point_world.y();
      spot_world.pose.position.z = point_world.z();
      spot_world.pose.orientation = msg->pose.orientation;
      
      RCLCPP_INFO(this->get_logger(), 
        "🔄 Transformed base_link [%.3f, %.3f, %.3f] -> world [%.3f, %.3f, %.3f]",
        msg->pose.position.x, msg->pose.position.y, msg->pose.position.z,
        point_world.x(), point_world.y(), point_world.z());
        
    } else if (msg->header.frame_id.empty() || msg->header.frame_id == WORLD_FRAME) {
      spot_world = *msg;
      spot_world.header.frame_id = WORLD_FRAME;
    } else {
      // Try TF for other frames
      try {
        spot_world = tf_buffer_->transform(*msg, WORLD_FRAME, tf2::durationFromSec(0.5));
      } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "TF error: %s", ex.what());
        
        std_msgs::msg::Bool fail_msg;
        fail_msg.data = true;
        beam_failed_pub_->publish(fail_msg);
        return;
      }
    }

    executeBeamTarget(spot_world);
  }

  void multiTargetCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "📦 Received %zu targets", msg->poses.size());
    
    for (const auto &pose : msg->poses) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = msg->header;
      ps.pose = pose;
      
      singleTargetCallback(std::make_shared<geometry_msgs::msg::PoseStamped>(ps));
    }
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OptimizedBeamPointingNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}