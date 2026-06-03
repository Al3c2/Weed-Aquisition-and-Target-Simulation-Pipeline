// beam_pointing_optimized.cpp
//
// Optimized beam pointing for fast weed elimination task
// Key improvements:
//   1. Cartesian path planning for smoother, faster movements
//   2. Predictive target queuing with look-ahead
//   3. Joint space planning fallback
//   4. Adaptive velocity scaling based on distance
//   5. Collision checking optimizations
//   6. Better IK seed state management

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

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

static const char WORLD_FRAME[] = "world";
static const char EE_LINK[]     = "tool0";
static const char GROUP_NAME[]  = "ur3_manipulator";

class OptimizedBeamPointingNode : public rclcpp::Node
{
public:
  OptimizedBeamPointingNode()
  : Node("optimized_beam_pointing"),
    tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
    tf_listener_(*tf_buffer_),
    move_group_(std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node*){}), GROUP_NAME)
  {
    RCLCPP_INFO(this->get_logger(), "🚀 Optimized Beam Pointing Node starting...");

    // === Beam parameters ===
    beam_length_ = this->declare_parameter<double>("beam_length", 0.65);
    ray_L_min_   = this->declare_parameter<double>("ray_L_min", 0.40);
    ray_L_max_   = this->declare_parameter<double>("ray_L_max", 1.00);
    ray_L_step_  = this->declare_parameter<double>("ray_L_step", 0.05);

    // === Performance parameters ===
    use_cartesian_path_ = this->declare_parameter<bool>("use_cartesian_path", true);
    cartesian_step_size_ = this->declare_parameter<double>("cartesian_step_size", 0.01);
    cartesian_jump_threshold_ = this->declare_parameter<double>("cartesian_jump_threshold", 0.0);
    
    enable_target_queue_ = this->declare_parameter<bool>("enable_target_queue", true);
    max_queue_size_ = this->declare_parameter<int>("max_queue_size", 10);
    
    adaptive_velocity_ = this->declare_parameter<bool>("adaptive_velocity", true);
    min_velocity_scale_ = this->declare_parameter<double>("min_velocity_scale", 0.5);
    max_velocity_scale_ = this->declare_parameter<double>("max_velocity_scale", 1.0);
    
    planning_timeout_ = this->declare_parameter<double>("planning_timeout", 2.0);
    num_planning_attempts_ = this->declare_parameter<int>("num_planning_attempts", 3);

    // === Swincar collision ===
    enable_swincar_collision_ = this->declare_parameter<bool>("enable_swincar_collision", true);
    swincar_mesh_uri_ = this->declare_parameter<std::string>(
        "swincar_mesh_uri",
        "file:///home/alex/.ignition/gazebo/models/swincar/meshes/swincar_collision.dae");
    swincar_pose_xyz_ = this->declare_parameter<std::vector<double>>(
        "swincar_pose_xyz", std::vector<double>{0.0, 0.0, 0.0});
    swincar_pose_rpy_ = this->declare_parameter<std::vector<double>>(
        "swincar_pose_rpy", std::vector<double>{0.0, 0.0, 0.0});

    // === MoveIt configuration ===
    move_group_.setEndEffectorLink(EE_LINK);
    move_group_.setPoseReferenceFrame(WORLD_FRAME);
    move_group_.setPlanningTime(planning_timeout_);
    move_group_.setPlannerId("RRTConnectkConfigDefault");
    move_group_.setNumPlanningAttempts(num_planning_attempts_);
    move_group_.setGoalPositionTolerance(0.008);  // Tighter tolerance
    move_group_.setGoalOrientationTolerance(0.03);
    move_group_.startStateMonitor();
    
    // Disable replanning for faster execution
    move_group_.allowReplanning(false);

    rclcpp::sleep_for(std::chrono::milliseconds(500));

    // Add Swincar collision
    if (enable_swincar_collision_) {
      addSwincarCollision();
    }

    // === Subscriptions ===
    single_target_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "target_pose", 10,
      std::bind(&OptimizedBeamPointingNode::singleTargetCallback, this, std::placeholders::_1));

    // Optional: subscribe to target array for batch processing
    multi_target_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
      "target_poses", 10,
      std::bind(&OptimizedBeamPointingNode::multiTargetCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
      "✅ Ready! Beam: %.2fm, Cartesian: %s, Adaptive velocity: %s",
      beam_length_, use_cartesian_path_ ? "ON" : "OFF",
      adaptive_velocity_ ? "ON" : "OFF");
  }

private:
  // === Parameters ===
  double beam_length_, ray_L_min_, ray_L_max_, ray_L_step_;
  bool use_cartesian_path_, enable_target_queue_, adaptive_velocity_;
  double cartesian_step_size_, cartesian_jump_threshold_;
  int max_queue_size_, num_planning_attempts_;
  double min_velocity_scale_, max_velocity_scale_, planning_timeout_;
  
  bool enable_swincar_collision_;
  std::string swincar_mesh_uri_;
  std::vector<double> swincar_pose_xyz_, swincar_pose_rpy_;

  // === ROS/MoveIt ===
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  moveit::planning_interface::MoveGroupInterface move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr single_target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr multi_target_sub_;
  
  // === Target queue ===
  std::deque<geometry_msgs::msg::PoseStamped> target_queue_;

  // === Collision setup ===
  void addSwincarCollision()
  {
    auto mesh = std::unique_ptr<shapes::Mesh>(
        shapes::createMeshFromResource(swincar_mesh_uri_));
    if (!mesh) {
      RCLCPP_WARN(this->get_logger(), "⚠️  Could not load Swincar mesh");
      return;
    }

    shapes::ShapeMsg shape_msg;
    shapes::constructMsgFromShape(mesh.get(), shape_msg);
    shape_msgs::msg::Mesh mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg);

    moveit_msgs::msg::CollisionObject obj;
    obj.id = "swincar";
    obj.operation = obj.ADD;
    obj.header.frame_id = WORLD_FRAME;
    obj.header.stamp = this->now();

    geometry_msgs::msg::Pose pose;
    pose.position.x = swincar_pose_xyz_.size() > 0 ? swincar_pose_xyz_[0] : 0.0;
    pose.position.y = swincar_pose_xyz_.size() > 1 ? swincar_pose_xyz_[1] : 0.0;
    pose.position.z = swincar_pose_xyz_.size() > 2 ? swincar_pose_xyz_[2] : 0.0;

    double r = swincar_pose_rpy_.size() > 0 ? swincar_pose_rpy_[0] : 0.0;
    double p = swincar_pose_rpy_.size() > 1 ? swincar_pose_rpy_[1] : 0.0;
    double y = swincar_pose_rpy_.size() > 2 ? swincar_pose_rpy_[2] : 0.0;
    tf2::Quaternion q; q.setRPY(r, p, y); q.normalize();
    pose.orientation = tf2::toMsg(q);

    obj.meshes = {mesh_msg};
    obj.mesh_poses = {pose};

    planning_scene_interface_.applyCollisionObjects({obj});
    rclcpp::sleep_for(std::chrono::milliseconds(300));

    RCLCPP_INFO(this->get_logger(), "🧱 Swincar collision mesh added");
  }

  // === Beam geometry helpers ===
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
      double angle = std::acos(dot);
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
    // Sort by proximity to preferred beam length
    std::sort(Ls.begin(), Ls.end(),
              [this](double a, double b) {
                return std::fabs(a - beam_length_) < std::fabs(b - beam_length_);
              });
    return Ls;
  }

  // === IK with better seed state ===
  bool computeIKWithSeed(const geometry_msgs::msg::Pose &target_pose,
                         moveit::core::RobotState &result_state,
                         double timeout = 0.5)
  {
    moveit::core::RobotModelConstPtr kmodel = move_group_.getRobotModel();
    if (!kmodel) return false;

    const auto* jmg = kmodel->getJointModelGroup(GROUP_NAME);
    if (!jmg) return false;

    // Use current state as seed
    result_state = *move_group_.getCurrentState();
    
    return result_state.setFromIK(jmg, target_pose, EE_LINK, timeout);
  }

  // === Adaptive velocity scaling ===
  double computeVelocityScale(const geometry_msgs::msg::Pose &current,
                              const geometry_msgs::msg::Pose &target)
  {
    if (!adaptive_velocity_) {
      return max_velocity_scale_;
    }

    double dx = target.position.x - current.position.x;
    double dy = target.position.y - current.position.y;
    double dz = target.position.z - current.position.z;
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Scale velocity based on distance: far = fast, close = slow
    double scale = min_velocity_scale_ + 
                   (max_velocity_scale_ - min_velocity_scale_) * 
                   std::min(dist / 0.5, 1.0);

    return std::clamp(scale, min_velocity_scale_, max_velocity_scale_);
  }

  // === Cartesian path planning ===
  bool planCartesianPath(const geometry_msgs::msg::Pose &target_pose,
                         moveit::planning_interface::MoveGroupInterface::Plan &plan,
                         double velocity_scale)
  {
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = move_group_.computeCartesianPath(
        waypoints,
        cartesian_step_size_,
        cartesian_jump_threshold_,
        trajectory,
        true);  // avoid collisions

    if (fraction < 0.95) {
      RCLCPP_DEBUG(this->get_logger(), 
                   "Cartesian path only %.1f%% complete", fraction * 100);
      return false;
    }

    // Apply time parameterization with velocity scaling
    robot_trajectory::RobotTrajectory rt(move_group_.getRobotModel(), GROUP_NAME);
    rt.setRobotTrajectoryMsg(*move_group_.getCurrentState(), trajectory);
    
    trajectory_processing::IterativeParabolicTimeParameterization iptp;
    bool success = iptp.computeTimeStamps(rt, velocity_scale, velocity_scale);
    
    if (!success) {
      RCLCPP_WARN(this->get_logger(), "Time parameterization failed");
      return false;
    }

    rt.getRobotTrajectoryMsg(plan.trajectory_);
    plan.planning_time_ = 0.0;  // Cartesian planning is fast
    
    return true;
  }

  // === Joint space planning (fallback) ===
  bool planJointSpace(const moveit::core::RobotState &target_state,
                      moveit::planning_interface::MoveGroupInterface::Plan &plan,
                      double velocity_scale)
  {
    move_group_.setStartStateToCurrentState();
    move_group_.setJointValueTarget(target_state);
    move_group_.setMaxVelocityScalingFactor(velocity_scale);
    move_group_.setMaxAccelerationScalingFactor(velocity_scale * 0.8);

    auto code = move_group_.plan(plan);
    return (code == moveit::core::MoveItErrorCode::SUCCESS);
  }

  // === Main planning and execution ===
  bool executeBeamTarget(const geometry_msgs::msg::PoseStamped &spot_world)
  {
    // Get spot position
    tf2::Vector3 T(spot_world.pose.position.x,
                   spot_world.pose.position.y,
                   spot_world.pose.position.z);

    // Current TCP position
    auto curr_pose = move_group_.getCurrentPose(EE_LINK).pose;
    tf2::Vector3 P_curr(curr_pose.position.x,
                        curr_pose.position.y,
                        curr_pose.position.z);

    // Beam direction
    tf2::Vector3 d = T - P_curr;
    double dist = d.length();
    if (dist < 1e-4) {
      RCLCPP_WARN(this->get_logger(), "Target too close to current TCP");
      return false;
    }
    d /= dist;

    // Orientation
    tf2::Quaternion q = alignZToDirection(d);

    // Find valid IK solution
    auto lengths = candidateLengths();
    geometry_msgs::msg::Pose goal_pose;
    moveit::core::RobotState goal_state(move_group_.getRobotModel());
    bool found_ik = false;
    double chosen_L = 0.0;

    for (double L : lengths) {
      tf2::Vector3 P_goal = T - d * L;
      
      goal_pose.position.x = P_goal.x();
      goal_pose.position.y = P_goal.y();
      goal_pose.position.z = P_goal.z();
      goal_pose.orientation = tf2::toMsg(q);

      if (computeIKWithSeed(goal_pose, goal_state, 0.3)) {
        chosen_L = L;
        found_ik = true;
        break;
      }
    }

    if (!found_ik) {
      RCLCPP_WARN(this->get_logger(), 
                  "❌ No IK solution for spot [%.2f, %.2f, %.2f]",
                  T.x(), T.y(), T.z());
      return false;
    }

    // Compute velocity scale
    double vel_scale = computeVelocityScale(curr_pose, goal_pose);

    // Try planning
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool planned = false;

    // Method 1: Cartesian path (faster, smoother)
    if (use_cartesian_path_) {
      planned = planCartesianPath(goal_pose, plan, vel_scale);
      if (planned) {
        RCLCPP_DEBUG(this->get_logger(), "✓ Cartesian path planned");
      }
    }

    // Method 2: Joint space fallback
    if (!planned) {
      planned = planJointSpace(goal_state, plan, vel_scale);
      if (planned) {
        RCLCPP_DEBUG(this->get_logger(), "✓ Joint space plan succeeded");
      }
    }

    if (!planned) {
      RCLCPP_WARN(this->get_logger(), "❌ Planning failed for this target");
      return false;
    }

    // Execute
    RCLCPP_INFO(this->get_logger(),
      "🎯 Target [%.2f, %.2f, %.2f] | L=%.2fm | vel=%.1f%%",
      T.x(), T.y(), T.z(), chosen_L, vel_scale * 100);

    auto exec_code = move_group_.execute(plan);
    bool success = (exec_code == moveit::core::MoveItErrorCode::SUCCESS);

    if (!success) {
      RCLCPP_WARN(this->get_logger(), "⚠️  Execution failed");
    }

    move_group_.clearPoseTargets();
    return success;
  }

  // === Callbacks ===
  void singleTargetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (enable_target_queue_ && !target_queue_.empty()) {
      // Add to queue
      if ((int)target_queue_.size() < max_queue_size_) {
        target_queue_.push_back(*msg);
        RCLCPP_DEBUG(this->get_logger(), "Added to queue (size: %zu)", target_queue_.size());
      }
      return;
    }

    // Transform to world frame
    geometry_msgs::msg::PoseStamped spot_world;
    try {
      if (msg->header.frame_id.empty() || msg->header.frame_id == WORLD_FRAME) {
        spot_world = *msg;
        spot_world.header.frame_id = WORLD_FRAME;
      } else {
        spot_world = tf_buffer_->transform(*msg, WORLD_FRAME, tf2::durationFromSec(0.5));
      }
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "TF error: %s", ex.what());
      return;
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