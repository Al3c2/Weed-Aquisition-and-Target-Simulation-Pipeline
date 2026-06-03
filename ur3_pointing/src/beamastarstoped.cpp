// beamastar_stop.cpp
//
// RRT* beam-pointing planner tuned for the STOP-START flow, where the
// swincar brakes to a full stop before a target is sent. Paired with
// color_detector.py (v4.19+), which:
//   1. detects a target,
//   2. publishes /blue_target_primary to stop the car,
//   3. waits for /swincar_stopped,
//   4. re-acquires the target (car still),
//   5. publishes /target_pose → this node,
//   6. waits for /beam_task_done or /beam_task_failed,
//   7. (on failure) retries the whole thing via the detector's
//      short-circuit retry path — no swincar_stop_timeout stalls.

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/msg/collision_object.hpp>

#include <geometric_shapes/shapes.h>
#include <geometric_shapes/shape_operations.h>
#include <shape_msgs/msg/mesh.hpp>

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
#include <mutex>
#include <atomic>

static const char WORLD_FRAME[] = "world";
static const char EE_LINK[]     = "tool0";
static const char GROUP_NAME[]  = "ur3_manipulator";

inline double normalizeAngle(double a) {
  while (a >  M_PI) a -= 2*M_PI;
  while (a < -M_PI) a += 2*M_PI;
  return a;
}
inline double closestAngle(double a, double ref) {
  double n = normalizeAngle(a);
  double opts[3] = {n, n+2*M_PI, n-2*M_PI};
  double best = n, bd = std::fabs(n-ref);
  for (int i=1;i<3;i++){ double d=std::fabs(opts[i]-ref); if(d<bd){bd=d;best=opts[i];} }
  return best;
}

// ════════════════════════════════════════════════════════════════════════════
class BeamPointingStopNode : public rclcpp::Node
{
public:
  BeamPointingStopNode()
  : Node("beam_pointing_rrtstar_stop"),   // distinct from moving variant
    tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
    tf_listener_(*tf_buffer_)
  {
    RCLCPP_INFO(get_logger(), "🚀 Beam Pointing RRT* STOP variant starting...");

    // Exact beam length — no range needed, RRT* works in joint space
    beam_length_ = declare_parameter<double>("beam_length", 0.65);

    // Aggressive timeouts for the stopped flow — we've got a stationary
    // world and the detector is waiting, so push hard.
    rrtstar_timeout_ = declare_parameter<double>("rrtstar_timeout",   3.0);
    rrtcon_timeout_  = declare_parameter<double>("rrtconnect_timeout", 1.5);

    // Adaptive velocity: scales linearly with TCP-to-goal distance.
    // Short moves use vel_min (precise, controlled); long moves ramp toward
    // vel_max (faster throughput). dist_min/max define the scaling range.
    vel_min_       = declare_parameter<double>("vel_min",       0.75);
    vel_max_       = declare_parameter<double>("vel_max",       0.80);
    acc_scale_     = declare_parameter<double>("acc_scale",     0.65);
    dist_min_      = declare_parameter<double>("dist_min",      0.05); // below → vel_min
    dist_max_      = declare_parameter<double>("dist_max",      0.50); // above → vel_max
    RCLCPP_INFO(get_logger(), "⚡ Adaptive vel: %.0f%%–%.0f%% over %.2f–%.2fm",
      vel_min_*100, vel_max_*100, dist_min_, dist_max_);

    dwell_time_ = declare_parameter<double>("dwell_time", 0.05);

    enable_swincar_collision_ = declare_parameter<bool>("enable_swincar_collision", true);
    swincar_mesh_uri_ = declare_parameter<std::string>("swincar_mesh_uri",
        "file:///home/alex/.ignition/gazebo/models/swincar/meshes/swincar_collision.dae");
    swincar_pose_xyz_ = declare_parameter<std::vector<double>>("swincar_pose_xyz", {0.,0.,0.});
    swincar_pose_rpy_ = declare_parameter<std::vector<double>>("swincar_pose_rpy", {0.,0.,0.});

    enable_ground_collision_ = declare_parameter<bool>("enable_ground_collision", true);
    ground_clearance_        = declare_parameter<double>("ground_clearance", 0.05);

    ground_truth_topic_    = declare_parameter<std::string>("ground_truth_topic",    "/model/swincar_ur3/pose");
    ground_truth_frame_id_ = declare_parameter<std::string>("ground_truth_frame_id", "empty");

    beam_error_threshold_ = declare_parameter<double>("beam_error_threshold", 0.04);

    std::string done_t   = declare_parameter<std::string>("arm_done_topic",   "/beam_task_done");
    std::string failed_t = declare_parameter<std::string>("arm_failed_topic", "/beam_task_failed");
    beam_done_pub_     = create_publisher<std_msgs::msg::Bool>(done_t,   10);
    beam_failed_pub_   = create_publisher<std_msgs::msg::Bool>(failed_t, 10);
    traj_duration_pub_ = create_publisher<std_msgs::msg::Float32>("/arm_trajectory_duration", 10);

    target_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "target_pose", 10,
      std::bind(&BeamPointingStopNode::targetCallback, this, std::placeholders::_1));
    multi_target_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "target_poses", 10,
      std::bind(&BeamPointingStopNode::multiTargetCallback, this, std::placeholders::_1));
    ground_truth_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      ground_truth_topic_, 10,
      std::bind(&BeamPointingStopNode::groundTruthCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "⏱️  Dwell:%.2fs | RRT*(%.1fs) → RRTConnect(%.1fs)",
      dwell_time_, rrtstar_timeout_, rrtcon_timeout_);

    init_timer_ = create_wall_timer(std::chrono::seconds(2),
      std::bind(&BeamPointingStopNode::initializeMoveIt, this));
  }

private:
  double beam_length_;
  double rrtstar_timeout_, rrtcon_timeout_;
  double vel_min_, vel_max_, acc_scale_, dist_min_, dist_max_;
  double dwell_time_, beam_error_threshold_;
  bool   enable_swincar_collision_, enable_ground_collision_;
  std::string swincar_mesh_uri_;
  std::vector<double> swincar_pose_xyz_, swincar_pose_rpy_;
  double ground_clearance_;
  std::string ground_truth_topic_, ground_truth_frame_id_;

  std::shared_ptr<tf2_ros::Buffer>         tf_buffer_;
  tf2_ros::TransformListener               tf_listener_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface>     move_group_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr   multi_target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ground_truth_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr    beam_done_pub_, beam_failed_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr traj_duration_pub_;

  rclcpp::TimerBase::SharedPtr init_timer_, collision_timer_;
  bool moveit_ready_ = false;

  geometry_msgs::msg::Pose ground_truth_pose_, last_collision_pose_;
  bool ground_truth_received_ = false;
  std::mutex pose_mutex_;
  std::atomic<bool> planning_frozen_{false};

  int total_targets_ = 0, successful_targets_ = 0;
  // Guard against infinite retry in executeTarget
  bool in_retry_ = false;

  // ── Ground truth ────────────────────────────────────────────────────────────
  void groundTruthCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (msg->header.frame_id != ground_truth_frame_id_) return;
    std::lock_guard<std::mutex> lk(pose_mutex_);
    ground_truth_pose_ = msg->pose;
    ground_truth_received_ = true;
  }

  // ── MoveIt init ─────────────────────────────────────────────────────────────
  void initializeMoveIt()
  {
    init_timer_->cancel();
    RCLCPP_INFO(get_logger(), "🔧 Initializing MoveIt...");
    try {
      move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
                      shared_from_this(), GROUP_NAME);
      planning_scene_interface_ =
        std::make_shared<moveit::planning_interface::PlanningSceneInterface>();

      move_group_->setEndEffectorLink(EE_LINK);
      move_group_->setPoseReferenceFrame(WORLD_FRAME);
      move_group_->setPlannerId("RRTstarkConfigDefault");
      move_group_->setPlanningTime(rrtstar_timeout_);
      move_group_->setNumPlanningAttempts(2);   // RRT* anytime: one long run
      move_group_->setGoalPositionTolerance(0.0001);
      move_group_->setGoalOrientationTolerance(0.001);
      // Velocity is set per-target by adaptive scaling — no fixed default needed
      move_group_->startStateMonitor();

      if (!move_group_->getCurrentState(5.0))
        RCLCPP_WARN(get_logger(), "Could not get initial robot state");

      // Seed ground truth from TF so collision objects land correctly
      // before the first ground truth callback fires
      try {
        auto tf = tf_buffer_->lookupTransform(WORLD_FRAME, "base_link",
                    tf2::TimePointZero, tf2::durationFromSec(2.0));
        ground_truth_pose_.position.x  = tf.transform.translation.x;
        ground_truth_pose_.position.y  = tf.transform.translation.y;
        ground_truth_pose_.position.z  = tf.transform.translation.z;
        ground_truth_pose_.orientation = tf.transform.rotation;
      } catch (...) { ground_truth_pose_.orientation.w = 1.0; }
      last_collision_pose_ = ground_truth_pose_;

      if (enable_ground_collision_)  addGroundCollision();
      if (enable_swincar_collision_) addSwincarCollision();

      moveit_ready_ = true;
      RCLCPP_INFO(get_logger(),
        "✅ MoveIt ready!  Beam:%.2fm  Vel:%.0f%%–%.0f%%  RRT*(%.1fs)→RRTConnect(%.1fs)",
        beam_length_, vel_min_*100, vel_max_*100, rrtstar_timeout_, rrtcon_timeout_);

      collision_timer_ = create_wall_timer(std::chrono::milliseconds(400),
        std::bind(&BeamPointingStopNode::updateCollisions, this));

    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "❌ MoveIt init failed: %s", e.what());
      init_timer_ = create_wall_timer(std::chrono::seconds(2),
        std::bind(&BeamPointingStopNode::initializeMoveIt, this));
    }
  }

  // ── Collision helpers ───────────────────────────────────────────────────────
  geometry_msgs::msg::Pose swincarPose(const geometry_msgs::msg::Pose& robot)
  {
    geometry_msgs::msg::Pose p;
    p.position.x = robot.position.x + (swincar_pose_xyz_.size()>0 ? swincar_pose_xyz_[0] : 0.);
    p.position.y = robot.position.y + (swincar_pose_xyz_.size()>1 ? swincar_pose_xyz_[1] : 0.);
    p.position.z = robot.position.z + (swincar_pose_xyz_.size()>2 ? swincar_pose_xyz_[2] : 0.);
    tf2::Quaternion rq; tf2::fromMsg(robot.orientation, rq);
    tf2::Quaternion oq; oq.setRPY(
      swincar_pose_rpy_.size()>0 ? swincar_pose_rpy_[0] : 0.,
      swincar_pose_rpy_.size()>1 ? swincar_pose_rpy_[1] : 0.,
      swincar_pose_rpy_.size()>2 ? swincar_pose_rpy_[2] : 0.);
    tf2::Quaternion cq = rq*oq; cq.normalize();
    p.orientation = tf2::toMsg(cq);
    return p;
  }

  void addGroundCollision()
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.id="ground_plane"; obj.operation=obj.ADD;
    obj.header.frame_id=WORLD_FRAME; obj.header.stamp=now();
    shape_msgs::msg::SolidPrimitive prim;
    prim.type=prim.BOX; prim.dimensions={20.,20.,0.1};
    geometry_msgs::msg::Pose p;
    p.position.x=ground_truth_pose_.position.x;
    p.position.y=ground_truth_pose_.position.y;
    p.position.z=ground_clearance_-0.05; p.orientation.w=1.;
    obj.primitives.push_back(prim); obj.primitive_poses.push_back(p);
    planning_scene_interface_->applyCollisionObjects({obj});
    RCLCPP_INFO(get_logger(), "🛡️ Ground collision added");
  }

  void addSwincarCollision()
  {
    RCLCPP_INFO(get_logger(), "🔧 Loading swincar mesh: %s", swincar_mesh_uri_.c_str());
    auto mesh = std::unique_ptr<shapes::Mesh>(shapes::createMeshFromResource(swincar_mesh_uri_));
    if (!mesh) { RCLCPP_ERROR(get_logger(), "❌ Could not load swincar mesh"); return; }
    RCLCPP_INFO(get_logger(), "   Mesh: %u vertices, %u triangles",
      mesh->vertex_count, mesh->triangle_count);
    shapes::ShapeMsg sm; shapes::constructMsgFromShape(mesh.get(), sm);
    auto mm = boost::get<shape_msgs::msg::Mesh>(sm);
    moveit_msgs::msg::CollisionObject obj;
    obj.id="swincar"; obj.operation=obj.ADD;
    obj.header.frame_id=WORLD_FRAME; obj.header.stamp=now();
    obj.meshes={mm}; obj.mesh_poses={swincarPose(ground_truth_pose_)};
    planning_scene_interface_->applyCollisionObjects({obj});
    rclcpp::sleep_for(std::chrono::milliseconds(500));
    RCLCPP_INFO(get_logger(), "🧱 Swincar collision mesh added");
  }

  void pushCollisionPoses(const geometry_msgs::msg::Pose& robot)
  {
    {
      moveit_msgs::msg::CollisionObject obj;
      obj.id="ground_plane"; obj.operation=obj.MOVE;
      obj.header.frame_id=WORLD_FRAME; obj.header.stamp=now();
      geometry_msgs::msg::Pose p;
      p.position.x=robot.position.x; p.position.y=robot.position.y;
      p.position.z=ground_clearance_-0.05; p.orientation.w=1.;
      obj.primitive_poses.push_back(p);
      planning_scene_interface_->applyCollisionObjects({obj});
    }
    if (enable_swincar_collision_) {
      moveit_msgs::msg::CollisionObject obj;
      obj.id="swincar"; obj.operation=obj.MOVE;
      obj.header.frame_id=WORLD_FRAME; obj.header.stamp=now();
      obj.mesh_poses.push_back(swincarPose(robot));
      planning_scene_interface_->applyCollisionObjects({obj});
    }
  }

  // ── Dynamic collision update (400ms timer) ──────────────────────────────────
  void updateCollisions()
  {
    if (!moveit_ready_ || !ground_truth_received_) return;
    if (planning_frozen_.load()) return;   // mesh locked during plan/execute

    geometry_msgs::msg::Pose cur;
    { std::lock_guard<std::mutex> lk(pose_mutex_); cur=ground_truth_pose_; }

    double dx=cur.position.x-last_collision_pose_.position.x;
    double dy=cur.position.y-last_collision_pose_.position.y;
    if (std::sqrt(dx*dx+dy*dy) < 0.03) return;

    last_collision_pose_=cur;
    pushCollisionPoses(cur);
  }

  // ── Freeze / unfreeze ───────────────────────────────────────────────────────
  void snapshotAndFreeze()
  {
    planning_frozen_.store(true);
    geometry_msgs::msg::Pose snap;
    { std::lock_guard<std::mutex> lk(pose_mutex_); snap=ground_truth_pose_; }
    // Push latest pose so IK, planning, AND post-processing all see the same world
    pushCollisionPoses(snap);
    rclcpp::sleep_for(std::chrono::milliseconds(50));
    RCLCPP_DEBUG(get_logger(),"🔒 Mesh frozen");
  }

  void unfreeze()
  {
    { std::lock_guard<std::mutex> lk(pose_mutex_);
      last_collision_pose_=ground_truth_pose_; }
    planning_frozen_.store(false);
    RCLCPP_DEBUG(get_logger(),"🔓 Mesh unfrozen");
  }

  // ── Geometry ────────────────────────────────────────────────────────────────
  tf2::Quaternion alignZToDirection(const tf2::Vector3& dir)
  {
    tf2::Vector3 v=dir.normalized(), z(0,0,1);
    double dot=z.dot(v);
    tf2::Quaternion q;
    if      (dot> 0.9999) q.setValue(0,0,0,1);
    else if (dot<-0.9999) q.setRotation({1,0,0},M_PI);
    else { tf2::Vector3 ax=z.cross(v); ax.normalize();
           q.setRotation(ax,std::acos(std::clamp(dot,-1.,1.))); }
    q.normalize(); return q;
  }

  // Adaptive velocity: linear ramp between vel_min and vel_max based on
  // how far the TCP needs to travel to reach the goal. Close targets
  // move slowly (precision), far ones move faster (throughput).
  double computeVelocityScale(const geometry_msgs::msg::Pose& goal_pose)
  {
    auto cur = move_group_->getCurrentPose(EE_LINK).pose;
    double dx = goal_pose.position.x - cur.position.x;
    double dy = goal_pose.position.y - cur.position.y;
    double dz = goal_pose.position.z - cur.position.z;
    double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    double t = std::clamp((dist - dist_min_) / (dist_max_ - dist_min_), 0.0, 1.0);
    double vel = vel_min_ + t * (vel_max_ - vel_min_);

    RCLCPP_INFO(get_logger(), "   TCP dist: %.3fm → vel scale: %.0f%%", dist, vel*100);
    return vel;
  }

  // Compute goal TCP pose: place TCP at (T - dir*beam_length_),
  // Z-axis of tool0 pointing at T.
  bool computeGoalPose(const tf2::Vector3& T, geometry_msgs::msg::Pose& goal_pose)
  {
    auto cur=move_group_->getCurrentPose(EE_LINK).pose;
    tf2::Vector3 P(cur.position.x,cur.position.y,cur.position.z);
    tf2::Vector3 dir=T-P;
    if (dir.length()<0.01){ RCLCPP_WARN(get_logger(),"Target too close"); return false; }
    dir.normalize();

    // Iterative refinement: recompute direction from the goal point itself
    // (not from TCP) for accurate beam_length_ placement
    tf2::Vector3 G;
    for (int i=0;i<3;i++){
      G=T-dir*beam_length_;
      tf2::Vector3 nd=(T-G); if(nd.length()<0.001) break;
      nd.normalize(); if((nd-dir).length()<0.0001) break;
      dir=nd;
    }

    goal_pose.position.x=G.x(); goal_pose.position.y=G.y(); goal_pose.position.z=G.z();
    goal_pose.orientation=tf2::toMsg(alignZToDirection(dir));
    return true;
  }

  // ── Trajectory utils ────────────────────────────────────────────────────────
  void normalizeTrajectory(trajectory_msgs::msg::JointTrajectory& traj)
  {
    if (traj.points.empty()) return;
    auto cs=move_group_->getCurrentState(); if(!cs) return;
    const auto* jmg=cs->getJointModelGroup(GROUP_NAME); if(!jmg) return;
    std::vector<double> ref; cs->copyJointGroupPositions(jmg,ref);
    for (auto& pt:traj.points){
      for (size_t i=0;i<pt.positions.size()&&i<ref.size();i++)
        if (std::fabs(pt.positions[i]-ref[i])>M_PI)
          pt.positions[i]=closestAngle(pt.positions[i],ref[i]);
      ref=pt.positions;
    }
  }

  void publishDuration(const moveit::planning_interface::MoveGroupInterface::Plan& plan,
                       const char* label)
  {
    const auto& pts=plan.trajectory_.joint_trajectory.points;
    if (pts.empty()) return;
    double d=pts.back().time_from_start.sec+pts.back().time_from_start.nanosec*1e-9;
    RCLCPP_INFO(get_logger(),"   [%s] duration: %.2fs",label,d);
    std_msgs::msg::Float32 msg; msg.data=float(d); traj_duration_pub_->publish(msg);
  }

  // ── Planners (both use setPoseTarget — OMPL samples IK internally) ──────────
  bool tryRRTStar(const geometry_msgs::msg::Pose& goal_pose, double vel,
                  moveit::planning_interface::MoveGroupInterface::Plan& plan)
  {
    move_group_->setPlannerId("RRTstarkConfigDefault");
    move_group_->setPlanningTime(rrtstar_timeout_);
    move_group_->setNumPlanningAttempts(2);
    move_group_->setMaxVelocityScalingFactor(vel);
    move_group_->setMaxAccelerationScalingFactor(acc_scale_);
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(goal_pose);
    auto res=move_group_->plan(plan);
    move_group_->clearPoseTargets();
    if (res!=moveit::core::MoveItErrorCode::SUCCESS)
      { RCLCPP_DEBUG(get_logger(),"   RRT* failed: %d",res.val); return false; }
    normalizeTrajectory(plan.trajectory_.joint_trajectory);
    publishDuration(plan,"RRT*");
    RCLCPP_INFO(get_logger(),"   ✓ RRT* plan (vel=%.0f%%)",vel*100); return true;
  }

  bool tryRRTConnect(const geometry_msgs::msg::Pose& goal_pose, double vel,
                     moveit::planning_interface::MoveGroupInterface::Plan& plan)
  {
    move_group_->setPlannerId("RRTConnectkConfigDefault");
    move_group_->setPlanningTime(rrtcon_timeout_);
    move_group_->setNumPlanningAttempts(3);
    move_group_->setMaxVelocityScalingFactor(vel);
    move_group_->setMaxAccelerationScalingFactor(acc_scale_);
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(goal_pose);
    auto res=move_group_->plan(plan);
    move_group_->clearPoseTargets();
    // Restore primary defaults
    move_group_->setPlannerId("RRTstarkConfigDefault");
    move_group_->setPlanningTime(rrtstar_timeout_);
    move_group_->setNumPlanningAttempts(1);
    if (res!=moveit::core::MoveItErrorCode::SUCCESS)
      { RCLCPP_DEBUG(get_logger(),"   RRTConnect failed: %d",res.val); return false; }
    normalizeTrajectory(plan.trajectory_.joint_trajectory);
    publishDuration(plan,"RRTConnect");
    RCLCPP_INFO(get_logger(),"   ✓ RRTConnect fallback (vel=%.0f%%)",vel*100); return true;
  }

  // ── Beam tip verification ───────────────────────────────────────────────────
  double verifyBeamTip(const tf2::Vector3& T)
  {
    try {
      auto tf=tf_buffer_->lookupTransform(WORLD_FRAME,EE_LINK,
                tf2::TimePointZero,tf2::durationFromSec(0.5));
      tf2::Vector3 tcp(tf.transform.translation.x,
                       tf.transform.translation.y,
                       tf.transform.translation.z);
      tf2::Quaternion q; tf2::fromMsg(tf.transform.rotation,q);
      tf2::Matrix3x3 R(q);
      tf2::Vector3 bd(R[0][2],R[1][2],R[2][2]); bd.normalize();
      tf2::Vector3 tip=tcp+bd*beam_length_;
      double err=(tip-T).length();
      RCLCPP_INFO(get_logger(),
        "   📍 tip[%.3f,%.3f,%.3f] tgt[%.3f,%.3f,%.3f] err=%.1fmm",
        tip.x(),tip.y(),tip.z(),T.x(),T.y(),T.z(),err*1000);
      return err;
    } catch(...){ return 1.0; }
  }

  // ── Recovery ────────────────────────────────────────────────────────────────
  bool recoverToSafePosition()
  {
    RCLCPP_WARN(get_logger(),"🔄 Recovering to safe position...");
    move_group_->setPlannerId("RRTConnectkConfigDefault");
    move_group_->setPlanningTime(5.0);
    move_group_->setNumPlanningAttempts(5);
    move_group_->setMaxVelocityScalingFactor(1);
    move_group_->setMaxAccelerationScalingFactor(1);

    std::vector<std::vector<double>> configs={
      { 1.57,-1.57, 0.0,-1.57,-1.57,0.0},  // Initial position
      { 0.0,-1.57, 0.0,-1.57, 0.0,0.0},
      { 0.0,-2.0,  1.5,-1.07,-1.57,0.0},
      { 0.5,-1.8,  1.2,-0.97,-1.57,0.0},
      {-0.5,-1.8,  1.2,-0.97,-1.57,0.0},
    };
    bool ok=false;
    for (size_t i=0;i<configs.size();i++){
      move_group_->setStartStateToCurrentState();
      move_group_->setJointValueTarget(configs[i]);
      moveit::planning_interface::MoveGroupInterface::Plan p;
      if (move_group_->plan(p)==moveit::core::MoveItErrorCode::SUCCESS){
        normalizeTrajectory(p.trajectory_.joint_trajectory);
        if (move_group_->execute(p)==moveit::core::MoveItErrorCode::SUCCESS){
          RCLCPP_INFO(get_logger(),"✅ Recovery ok (config %zu)",i+1);
          ok=true; break;
        }
      }
    }
    if (!ok){ RCLCPP_ERROR(get_logger(),"❌ All recovery configs failed"); move_group_->stop(); }

    move_group_->setPlannerId("RRTstarkConfigDefault");
    move_group_->setPlanningTime(rrtstar_timeout_);
    move_group_->setNumPlanningAttempts(1);
    // Velocity for next real target will be set adaptively per-target
    rclcpp::sleep_for(std::chrono::milliseconds(300));
    return ok;
  }

  // ── Main execution ──────────────────────────────────────────────────────────
  bool executeTarget(const tf2::Vector3& T)
  {
    total_targets_++;
    RCLCPP_INFO(get_logger(),"🎯 Target %d: [%.3f,%.3f,%.3f]",
      total_targets_,T.x(),T.y(),T.z());

    // Freeze mesh so IK, planning, and post-processing all see the same world
    snapshotAndFreeze();

    // Compute goal pose with the fixed beam_length_
    geometry_msgs::msg::Pose goal_pose;
    if (!computeGoalPose(T, goal_pose)){
      RCLCPP_WARN(get_logger(),"❌ Could not compute goal pose");
      unfreeze(); publishFailed(); return false;
    }
    RCLCPP_INFO(get_logger(),"   Goal TCP:[%.3f,%.3f,%.3f]",
      goal_pose.position.x, goal_pose.position.y, goal_pose.position.z);

    // Adaptive velocity: slow for close targets, fast for far ones
    double vel = computeVelocityScale(goal_pose);

    // Plan: RRT* primary, RRTConnect fallback — both use setPoseTarget so OMPL
    // samples across all IK solutions and routes around the swincar mesh
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool planned=false;
    if (tryRRTStar(goal_pose, vel, plan)){
      RCLCPP_INFO(get_logger(),"   [PRIMARY] RRT* succeeded"); planned=true;
    } else {
      RCLCPP_WARN(get_logger(),"   [PRIMARY] RRT* failed — trying RRTConnect...");
      if (tryRRTConnect(goal_pose, vel, plan)){
        RCLCPP_INFO(get_logger(),"   [FALLBACK] RRTConnect succeeded"); planned=true;
      }
    }

    if (!planned){
      RCLCPP_ERROR(get_logger(),"❌ Planning failed — attempting recovery...");
      unfreeze();
      if (!in_retry_ && recoverToSafePosition()){
        RCLCPP_INFO(get_logger(),"   Retrying after recovery...");
        in_retry_=true;
        bool r=executeTarget(T);
        in_retry_=false;
        return r;
      }
      publishFailed(); return false;
    }

    // Execute
    RCLCPP_INFO(get_logger(),"   Executing...");
    auto exec=move_group_->execute(plan);
    unfreeze();   // resume mesh updates immediately after execution

    if (exec!=moveit::core::MoveItErrorCode::SUCCESS){
      RCLCPP_ERROR(get_logger(),"❌ Execution failed: %d",exec.val);
      recoverToSafePosition(); publishFailed(); return false;
    }

    // Dwell then verify
    RCLCPP_INFO(get_logger(),"   ⏱️  Dwell %.2fs...",dwell_time_);
    rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(dwell_time_)));

    double err=verifyBeamTip(T);
    if (err<beam_error_threshold_){
      successful_targets_++;
      RCLCPP_INFO(get_logger(),"✅ SUCCESS  err=%.1fmm  rate=%.1f%%",
        err*1000,100.*successful_targets_/total_targets_);
      std_msgs::msg::Bool m; m.data=true; beam_done_pub_->publish(m);
      return true;
    }
    RCLCPP_WARN(get_logger(),"⚠️  Error %.1fmm > %.1fmm threshold",
      err*1000,beam_error_threshold_*1000);
    publishFailed(); return false;
  }

  void publishFailed()
  { std_msgs::msg::Bool m; m.data=true; beam_failed_pub_->publish(m); }

  // ── Callbacks ───────────────────────────────────────────────────────────────
  void targetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (!moveit_ready_){ RCLCPP_WARN(get_logger(),"⏳ Not ready"); return; }
    tf2::Vector3 T;
    if (msg->header.frame_id.empty()||msg->header.frame_id==WORLD_FRAME)
      T.setValue(msg->pose.position.x,msg->pose.position.y,msg->pose.position.z);
    else {
      try {
        auto tf=tf_buffer_->transform(*msg,WORLD_FRAME,tf2::durationFromSec(0.5));
        T.setValue(tf.pose.position.x,tf.pose.position.y,tf.pose.position.z);
      } catch(const tf2::TransformException& e){
        RCLCPP_ERROR(get_logger(),"TF error: %s",e.what()); publishFailed(); return;
      }
    }
    executeTarget(T);
  }

  void multiTargetCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(),"📦 Received %zu targets",msg->poses.size());
    for (const auto& pose:msg->poses){
      geometry_msgs::msg::PoseStamped ps; ps.header=msg->header; ps.pose=pose;
      targetCallback(std::make_shared<geometry_msgs::msg::PoseStamped>(ps));
    }
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node=std::make_shared<BeamPointingStopNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}