// Beamastar.cpp
//
// BASE: Beam_pointing_rrtstar.cpp (doc-13 — planning known-good).
//   All planning code is IDENTICAL to that version:
//   computeGoalPose → RRT* → RRTConnect fallback, snapshotAndFreeze,
//   recovery, joint normalisation, adaptive velocity.
//
// ADDED: Real-time beam-tip X hold phase.
//   After the arm executes and the dwell elapses, we enter a polling loop
//   that compares the beam-tip X coordinate in WORLD frame against the
//   target ball X (received on /target_world_pos, published by the detector
//   simultaneously with /target_pose).
//
//   Beam tip world coords:
//     1. lookupTransform("base_link", "tool0") → TCP in base_link
//        (reliable — purely joint angles, unaffected by robot motion)
//     2. Extend TCP by beam_length along tool0 Z-axis → beam_tip in base_link
//     3. beam_tip_world = ground_truth_pose_.position + R_robot * beam_tip_bl
//        (uses ground_truth_pose_, NOT tf_buffer_, for base_link→world because
//         TF world→base_link is frozen at spawn and never updates as robot drives)
//
//   When beam_tip_x enters [ball_x - tol, ball_x + tol]: SUCCESS → beam_done
//   When beam_tip_x < ball_x - tol: OVERSHOT → beam_failed
//
// THREADING:
//   targetCallback spawns a detached thread so the hold loop does not block
//   the ROS2 executor. groundTruthCallback must keep firing to update
//   ground_truth_pose_ during the loop.
//   MultiThreadedExecutor in main() keeps timers/subs live concurrently.

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
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
#include <thread>

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
class BeamPointingRRTStarNode : public rclcpp::Node
{
public:
  BeamPointingRRTStarNode()
  : Node("beam_pointing_rrtstar"),
    tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
    tf_listener_(*tf_buffer_)
  {
    RCLCPP_INFO(get_logger(), "🚀 Beam Pointing RRT* Node starting...");

    // ── Planning params (doc-13 identical) ───────────────────────────────
    beam_length_     = declare_parameter<double>("beam_length",       0.65);
    rrtstar_timeout_ = declare_parameter<double>("rrtstar_timeout",   3.5);
    rrtcon_timeout_  = declare_parameter<double>("rrtconnect_timeout", 0.5);
    vel_min_         = declare_parameter<double>("vel_min",            0.91);
    vel_max_         = declare_parameter<double>("vel_max",            0.98);
    acc_scale_       = declare_parameter<double>("acc_scale",          0.95);
    dist_min_        = declare_parameter<double>("dist_min",           0.05);
    dist_max_        = declare_parameter<double>("dist_max",           0.50);
    dwell_time_      = declare_parameter<double>("dwell_time",         0.05);
    beam_error_threshold_ = declare_parameter<double>("beam_error_threshold", 0.02);

    enable_swincar_collision_ = declare_parameter<bool>("enable_swincar_collision", true);
    swincar_mesh_uri_ = declare_parameter<std::string>("swincar_mesh_uri",
        "file:///home/alex/.ignition/gazebo/models/swincar/meshes/swincar_collision.dae");
    swincar_pose_xyz_ = declare_parameter<std::vector<double>>("swincar_pose_xyz", {0.,0.,0.});
    swincar_pose_rpy_ = declare_parameter<std::vector<double>>("swincar_pose_rpy", {0.,0.,0.});
    enable_ground_collision_ = declare_parameter<bool>("enable_ground_collision", true);
    ground_clearance_        = declare_parameter<double>("ground_clearance", 0.05);
    ground_truth_topic_    = declare_parameter<std::string>("ground_truth_topic",    "/model/swincar_ur3/pose");
    ground_truth_frame_id_ = declare_parameter<std::string>("ground_truth_frame_id", "empty");

    // ── Hold-phase params (NEW) ──────────────────────────────────────────
    beam_x_tolerance_ = declare_parameter<double>("beam_x_tolerance", 0.159); // ±30 mm
    beam_poll_ms_     = declare_parameter<int>   ("beam_poll_ms",     20);    // 50 Hz
    hold_timeout_     = declare_parameter<double>("hold_timeout",     60.0);  // seconds

    std::string done_t   = declare_parameter<std::string>("arm_done_topic",   "/beam_task_done");
    std::string failed_t = declare_parameter<std::string>("arm_failed_topic", "/beam_task_failed");
    beam_done_pub_     = create_publisher<std_msgs::msg::Bool>(done_t,   10);
    beam_failed_pub_   = create_publisher<std_msgs::msg::Bool>(failed_t, 10);
    traj_duration_pub_ = create_publisher<std_msgs::msg::Float32>("/arm_trajectory_duration", 10);
    beam_tip_pub_      = create_publisher<geometry_msgs::msg::PointStamped>("/beam_tip_world", 10);
    beam_yz_pub_       = create_publisher<geometry_msgs::msg::PointStamped>("/beam_tip_yz", 10);

    target_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "target_pose", 10,
      std::bind(&BeamPointingRRTStarNode::targetCallback, this, std::placeholders::_1));
    multi_target_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      "target_poses", 10,
      std::bind(&BeamPointingRRTStarNode::multiTargetCallback, this, std::placeholders::_1));
    ground_truth_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      ground_truth_topic_, 10,
      std::bind(&BeamPointingRRTStarNode::groundTruthCallback, this, std::placeholders::_1));
    target_world_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/target_world_pos", 10,
      std::bind(&BeamPointingRRTStarNode::targetWorldCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "⏱️  Dwell:%.2fs | RRT*(%.1fs)→RRTConnect(%.1fs) | x_tol:±%.0fmm poll:%dms",
      dwell_time_, rrtstar_timeout_, rrtcon_timeout_,
      beam_x_tolerance_*1000, beam_poll_ms_);

    init_timer_ = create_wall_timer(std::chrono::seconds(2),
      std::bind(&BeamPointingRRTStarNode::initializeMoveIt, this));
  }

private:
  // Planning params
  double beam_length_;
  double rrtstar_timeout_, rrtcon_timeout_;
  double vel_min_, vel_max_, acc_scale_, dist_min_, dist_max_;
  double dwell_time_, beam_error_threshold_;
  bool   enable_swincar_collision_, enable_ground_collision_;
  std::string swincar_mesh_uri_;
  std::vector<double> swincar_pose_xyz_, swincar_pose_rpy_;
  double ground_clearance_;
  std::string ground_truth_topic_, ground_truth_frame_id_;

  // Hold-phase params
  double beam_x_tolerance_;
  int    beam_poll_ms_;
  double hold_timeout_;

  std::shared_ptr<tf2_ros::Buffer>         tf_buffer_;
  tf2_ros::TransformListener               tf_listener_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface>     move_group_;
  std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr  target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr    multi_target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr  ground_truth_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr target_world_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr    beam_done_pub_, beam_failed_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr traj_duration_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr beam_tip_pub_;
  // Beam tip YZ snapshot in base_link frame, published once after each
  // trajectory completes. X is set to 0 (meaningless during sweep).
  // Evaluator uses Y and Z to compute true arm pointing error vs GT.
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr beam_yz_pub_;

  rclcpp::TimerBase::SharedPtr init_timer_, collision_timer_, beam_tip_timer_;
  bool moveit_ready_ = false;

  geometry_msgs::msg::Pose ground_truth_pose_, last_collision_pose_;
  bool ground_truth_received_ = false;
  std::mutex pose_mutex_;
  std::atomic<bool> planning_frozen_{false};

  // Ball world position received from detector
  tf2::Vector3 latest_ball_world_{0, 0, 0};
  bool         ball_world_received_ = false;
  std::mutex   ball_mutex_;

  // Prevent re-entry from a second /target_pose while executing
  std::atomic<bool> execution_active_{false};

  int total_targets_ = 0, successful_targets_ = 0;
  bool in_retry_ = false;

  // Pending collision pose from detached thread → drained on collision_timer_ (main thread).
  // applyCollisionObjects() internally creates nodes and adds them to executors.
  // Calling it from a detached thread while rclcpp::spin() is active causes
  // "Node already added to executor" crash. Posting the pose here and applying
  // it on the timer callback (which runs on the main thread) avoids the crash.
  std::mutex           pending_mutex_;
  bool                 pending_pose_valid_{false};
  geometry_msgs::msg::Pose pending_pose_;

  // ── Callbacks ────────────────────────────────────────────────────────────

  void groundTruthCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (msg->header.frame_id != ground_truth_frame_id_) return;
    std::lock_guard<std::mutex> lk(pose_mutex_);
    ground_truth_pose_ = msg->pose;
    ground_truth_received_ = true;
  }

  void targetWorldCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(ball_mutex_);
    latest_ball_world_.setValue(msg->point.x, msg->point.y, msg->point.z);
    ball_world_received_ = true;
  }

  // ── Beam tip in world frame (NEW) ────────────────────────────────────────
  // Step 1: base_link→tool0 is reliable (joint angles only, not affected by
  //         robot motion). TF is always correct for this transform.
  bool getBeamTipBaseLink(tf2::Vector3& tip_out)
  {
    try {
      auto tf = tf_buffer_->lookupTransform(
        "base_link", EE_LINK, tf2::TimePointZero, tf2::durationFromSec(0.1));
      tf2::Vector3 tcp(tf.transform.translation.x,
                       tf.transform.translation.y,
                       tf.transform.translation.z);
      tf2::Quaternion q; tf2::fromMsg(tf.transform.rotation, q);
      tf2::Matrix3x3 R(q);
      tf2::Vector3 beam_dir(R[0][2], R[1][2], R[2][2]);
      beam_dir.normalize();
      tip_out = tcp + beam_dir * beam_length_;
      return true;
    } catch (...) { return false; }
  }

  // Step 2: base_link→world using ground_truth_pose_ (NOT tf_buffer_).
  //   world→base_link TF is frozen at spawn and never updates as robot drives.
  //   ground_truth_pose_ is updated on every GT callback — always current.
  bool getBeamTipWorld(tf2::Vector3& tip_out)
  {
    tf2::Vector3 tip_bl;
    if (!getBeamTipBaseLink(tip_bl)) return false;
    geometry_msgs::msg::Pose robot;
    { std::lock_guard<std::mutex> lk(pose_mutex_); robot = ground_truth_pose_; }
    tf2::Quaternion rq; tf2::fromMsg(robot.orientation, rq);
    tf2::Matrix3x3 R(rq);
    tf2::Vector3 rp(robot.position.x, robot.position.y, robot.position.z);
    tip_out = rp + R * tip_bl;
    return true;
  }

  // ── MoveIt init (doc-13 identical) ───────────────────────────────────────

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
      move_group_->setNumPlanningAttempts(3);
      move_group_->setGoalPositionTolerance(0.002);
      move_group_->setGoalOrientationTolerance(0.01);
      move_group_->startStateMonitor();

      if (!move_group_->getCurrentState(5.0))
        RCLCPP_WARN(get_logger(), "Could not get initial robot state");

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
        "✅ MoveIt ready!  Beam:%.2fm  Vel:%.0f%%–%.0f%%  XTol:±%.0fmm",
        beam_length_, vel_min_*100, vel_max_*100, beam_x_tolerance_*1000);

      collision_timer_ = create_wall_timer(std::chrono::milliseconds(10),
        std::bind(&BeamPointingRRTStarNode::updateCollisions, this));

      beam_tip_timer_ = create_wall_timer(std::chrono::milliseconds(50),
        std::bind(&BeamPointingRRTStarNode::publishBeamTip, this));

    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "❌ MoveIt init failed: %s", e.what());
      init_timer_ = create_wall_timer(std::chrono::seconds(2),
        std::bind(&BeamPointingRRTStarNode::initializeMoveIt, this));
    }
  }

  void publishBeamTip()
  {
    tf2::Vector3 tip;
    if (!getBeamTipWorld(tip)) return;
    geometry_msgs::msg::PointStamped m;
    m.header.stamp = now(); m.header.frame_id = WORLD_FRAME;
    m.point.x=tip.x(); m.point.y=tip.y(); m.point.z=tip.z();
    beam_tip_pub_->publish(m);
  }

  // ── Collision helpers (doc-13 identical) ─────────────────────────────────

  geometry_msgs::msg::Pose swincarPose(const geometry_msgs::msg::Pose& robot)
  {
    geometry_msgs::msg::Pose p;
    p.position.x=robot.position.x+(swincar_pose_xyz_.size()>0?swincar_pose_xyz_[0]:0.);
    p.position.y=robot.position.y+(swincar_pose_xyz_.size()>1?swincar_pose_xyz_[1]:0.);
    p.position.z=robot.position.z+(swincar_pose_xyz_.size()>2?swincar_pose_xyz_[2]:0.);
    tf2::Quaternion rq; tf2::fromMsg(robot.orientation,rq);
    tf2::Quaternion oq; oq.setRPY(
      swincar_pose_rpy_.size()>0?swincar_pose_rpy_[0]:0.,
      swincar_pose_rpy_.size()>1?swincar_pose_rpy_[1]:0.,
      swincar_pose_rpy_.size()>2?swincar_pose_rpy_[2]:0.);
    tf2::Quaternion cq=rq*oq; cq.normalize();
    p.orientation=tf2::toMsg(cq);
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
    auto mesh=std::unique_ptr<shapes::Mesh>(shapes::createMeshFromResource(swincar_mesh_uri_));
    if (!mesh){ RCLCPP_ERROR(get_logger(),"❌ Could not load swincar mesh"); return; }
    RCLCPP_INFO(get_logger(),"   Mesh: %u vertices, %u triangles",
      mesh->vertex_count,mesh->triangle_count);
    shapes::ShapeMsg sm; shapes::constructMsgFromShape(mesh.get(),sm);
    auto mm=boost::get<shape_msgs::msg::Mesh>(sm);
    moveit_msgs::msg::CollisionObject obj;
    obj.id="swincar"; obj.operation=obj.ADD;
    obj.header.frame_id=WORLD_FRAME; obj.header.stamp=now();
    obj.meshes={mm}; obj.mesh_poses={swincarPose(ground_truth_pose_)};
    planning_scene_interface_->applyCollisionObjects({obj});
    rclcpp::sleep_for(std::chrono::milliseconds(500));
    RCLCPP_INFO(get_logger(),"🧱 Swincar collision mesh added");
  }

  void pushCollisionPoses(const geometry_msgs::msg::Pose& robot)
  {
    // Only post — never call applyCollisionObjects here.
    // This is called from both the main thread (updateCollisions) AND the
    // detached execution thread (snapshotAndFreeze). applyCollisionObjects
    // internally creates nodes and touches executors — doing so from a
    // non-main thread causes "Node already added to executor" crash.
    // The actual apply happens in updateCollisions() which always runs on
    // the main thread via collision_timer_.
    std::lock_guard<std::mutex> lk(pending_mutex_);
    pending_pose_       = robot;
    pending_pose_valid_ = true;
  }

  void updateCollisions()
  {
    if (!moveit_ready_||!ground_truth_received_) return;

    // Drain any pending pose posted by the detached thread (snapshotAndFreeze).
    {
      std::lock_guard<std::mutex> lk(pending_mutex_);
      if (pending_pose_valid_) {
        last_collision_pose_ = pending_pose_;
        pending_pose_valid_  = false;
        // Apply on main thread — safe.
        applyCollisionPosesNow(last_collision_pose_);
        return;   // applied frozen pose; skip the normal drift check this cycle
      }
    }

    if (planning_frozen_.load()) return;
    geometry_msgs::msg::Pose cur;
    { std::lock_guard<std::mutex> lk(pose_mutex_); cur=ground_truth_pose_; }
    double dx=cur.position.x-last_collision_pose_.position.x;
    double dy=cur.position.y-last_collision_pose_.position.y;
    if (std::sqrt(dx*dx+dy*dy)<0.03) return;
    last_collision_pose_=cur;
    applyCollisionPosesNow(cur);
  }

  void applyCollisionPosesNow(const geometry_msgs::msg::Pose& robot)
  {
    // Always called from main thread (updateCollisions timer callback).
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
    if (enable_swincar_collision_){
      moveit_msgs::msg::CollisionObject obj;
      obj.id="swincar"; obj.operation=obj.MOVE;
      obj.header.frame_id=WORLD_FRAME; obj.header.stamp=now();
      obj.mesh_poses.push_back(swincarPose(robot));
      planning_scene_interface_->applyCollisionObjects({obj});
    }
  }

  void snapshotAndFreeze()
  {
    planning_frozen_.store(true);
    geometry_msgs::msg::Pose snap;
    { std::lock_guard<std::mutex> lk(pose_mutex_); snap=ground_truth_pose_; }
    pushCollisionPoses(snap);
    // Do NOT wait here. The collision_timer_ (400ms) will apply the pose on the
    // main thread. RRT* planning takes 1.5s minimum, so the collision will be
    // applied long before planning finishes. Waiting up to 400ms caused
    // extra robot movement that shifted the beam past close targets.
  }

  void unfreeze()
  {
    { std::lock_guard<std::mutex> lk(pose_mutex_);
      last_collision_pose_=ground_truth_pose_; }
    planning_frozen_.store(false);
  }

  // ── Geometry (doc-13 identical) ───────────────────────────────────────────

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

  double computeVelocityScale(const geometry_msgs::msg::Pose& goal_pose)
  {
    auto cur=move_group_->getCurrentPose(EE_LINK).pose;
    double dx=goal_pose.position.x-cur.position.x;
    double dy=goal_pose.position.y-cur.position.y;
    double dz=goal_pose.position.z-cur.position.z;
    double dist=std::sqrt(dx*dx+dy*dy+dz*dz);
    double t=std::clamp((dist-dist_min_)/(dist_max_-dist_min_),0.0,1.0);
    double vel=vel_min_+t*(vel_max_-vel_min_);
    RCLCPP_INFO(get_logger(),"   TCP dist: %.3fm → vel scale: %.0f%%",dist,vel*100);
    return vel;
  }

  bool computeGoalPose(const tf2::Vector3& T, geometry_msgs::msg::Pose& goal_pose)
  {
    auto cur=move_group_->getCurrentPose(EE_LINK).pose;
    tf2::Vector3 P(cur.position.x,cur.position.y,cur.position.z);
    tf2::Vector3 dir=T-P;
    if (dir.length()<0.01){ RCLCPP_WARN(get_logger(),"Target too close"); return false; }
    dir.normalize();
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

  bool tryRRTStar(const geometry_msgs::msg::Pose& goal_pose, double vel,
                  moveit::planning_interface::MoveGroupInterface::Plan& plan)
  {
    move_group_->setPlannerId("RRTstarkConfigDefault");
    move_group_->setPlanningTime(rrtstar_timeout_);
    move_group_->setNumPlanningAttempts(3);
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
    move_group_->setNumPlanningAttempts(1);
    move_group_->setMaxVelocityScalingFactor(vel);
    move_group_->setMaxAccelerationScalingFactor(acc_scale_);
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(goal_pose);
    auto res=move_group_->plan(plan);
    move_group_->clearPoseTargets();
    move_group_->setPlannerId("RRTstarkConfigDefault");
    move_group_->setPlanningTime(rrtstar_timeout_);
    move_group_->setNumPlanningAttempts(1);
    if (res!=moveit::core::MoveItErrorCode::SUCCESS)
      { RCLCPP_DEBUG(get_logger(),"   RRTConnect failed: %d",res.val); return false; }
    normalizeTrajectory(plan.trajectory_.joint_trajectory);
    publishDuration(plan,"RRTConnect");
    RCLCPP_INFO(get_logger(),"   ✓ RRTConnect fallback (vel=%.0f%%)",vel*100); return true;
  }

  bool recoverToSafePosition()
  {
    RCLCPP_WARN(get_logger(),"🔄 Recovering to safe position...");
    move_group_->setPlannerId("RRTConnectkConfigDefault");
    move_group_->setPlanningTime(1.0);
    move_group_->setNumPlanningAttempts(5);
    move_group_->setMaxVelocityScalingFactor(1);
    move_group_->setMaxAccelerationScalingFactor(1);
    std::vector<std::vector<double>> configs={
      { 1.57,-1.57, 0.0,-1.57,-1.57, 0.0},
      { 0.0, -1.57, 0.0,-1.57, 0.0,  0.0},
      { 0.0, -2.0,  1.5,-1.07,-1.57, 0.0},
      { 0.5, -1.8,  1.2,-0.97,-1.57, 0.0},
      {-0.5, -1.8,  1.2,-0.97,-1.57, 0.0},
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
    rclcpp::sleep_for(std::chrono::milliseconds(300));
    return ok;
  }

  // ════════════════════════════════════════════════════════════════════════
  // Main execution
  // ════════════════════════════════════════════════════════════════════════
  bool executeTarget(const tf2::Vector3& T, const tf2::Vector3& ball, bool ball_valid)
  {
    total_targets_++;
    RCLCPP_INFO(get_logger(),"🎯 Target %d: T=[%.3f,%.3f,%.3f]  ball=[%.3f,%.3f,%.3f]%s",
      total_targets_,T.x(),T.y(),T.z(),ball.x(),ball.y(),ball.z(),
      ball_valid?"":" ⚠️ no ball_world");

    snapshotAndFreeze();

    geometry_msgs::msg::Pose goal_pose;
    if (!computeGoalPose(T,goal_pose)){
      unfreeze(); publishFailed(); return false;
    }
    RCLCPP_INFO(get_logger(),"   Goal TCP:[%.3f,%.3f,%.3f]",
      goal_pose.position.x,goal_pose.position.y,goal_pose.position.z);

    double vel=computeVelocityScale(goal_pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool planned=false;
    if (tryRRTStar(goal_pose,vel,plan)){
      RCLCPP_INFO(get_logger(),"   [PRIMARY] RRT* succeeded"); planned=true;
    } else {
      RCLCPP_WARN(get_logger(),"   [PRIMARY] RRT* failed — trying RRTConnect...");
      if (tryRRTConnect(goal_pose,vel,plan)){
        RCLCPP_INFO(get_logger(),"   [FALLBACK] RRTConnect succeeded"); planned=true;
      }
    }

    if (!planned){
      RCLCPP_ERROR(get_logger(),"❌ Planning failed — attempting recovery...");
      unfreeze();
      if (!in_retry_&&recoverToSafePosition()){
        in_retry_=true;
        bool r=executeTarget(T,ball,ball_valid);
        in_retry_=false;
        return r;
      }
      publishFailed(); return false;
    }

    RCLCPP_INFO(get_logger(),"   Executing...");
    auto exec=move_group_->execute(plan);
    unfreeze();

    if (exec!=moveit::core::MoveItErrorCode::SUCCESS){
      RCLCPP_ERROR(get_logger(),"❌ Execution failed: %d",exec.val);
      recoverToSafePosition(); publishFailed(); return false;
    }

    rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(dwell_time_)));

    // Publish beam tip YZ snapshot in base_link frame for evaluator.
    // Done right after execution+dwell so arm is settled at goal pose.
    // X is set to 0 — only Y and Z are meaningful (arm pointing accuracy).
    // base_link frame keeps this purely in arm kinematics space, unaffected
    // by robot world position (which changes continuously during sweeping).
    {
      tf2::Vector3 tip_bl;
      if (getBeamTipBaseLink(tip_bl)){
        geometry_msgs::msg::PointStamped yz_msg;
        yz_msg.header.stamp    = now();
        yz_msg.header.frame_id = "base_link";
        yz_msg.point.x = 0.0;          // excluded — swept by robot motion
        yz_msg.point.y = tip_bl.y();
        yz_msg.point.z = tip_bl.z();
        beam_yz_pub_->publish(yz_msg);
        RCLCPP_INFO(get_logger(),
          "   📍 beam_tip_yz (base_link): y=%.3f  z=%.3f",
          tip_bl.y(), tip_bl.z());
      }
    }

    // ── Hold phase: poll beam_tip_x vs ball_x ─────────────────────────────
    // If no ball_world was received: declare success immediately (fallback).
    if (!ball_valid){
      successful_targets_++;
      RCLCPP_INFO(get_logger(),
        "   ✅ SUCCESS (no ball_world — skipping X sweep)  rate=%.1f%%",
        100.*successful_targets_/total_targets_);
      std_msgs::msg::Bool m; m.data=true; beam_done_pub_->publish(m);
      return true;
    }

    double ball_x=ball.x();
    RCLCPP_INFO(get_logger(),
      "   ⏳ Hold: sweeping beam_tip_x → ball_x=%.3f  (±%.0fmm, timeout=%.0fs)",
      ball_x,beam_x_tolerance_*1000,hold_timeout_);

    auto hold_start=std::chrono::steady_clock::now();
    int polls=0;

    // Runs on the detached thread — groundTruthCallback keeps firing on the
    // executor so ground_truth_pose_ (and therefore beam tip world) is live.
    while (true){
      std::this_thread::sleep_for(std::chrono::milliseconds(beam_poll_ms_));
      polls++;

      double elapsed=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-hold_start).count();
      if (elapsed>hold_timeout_){
        RCLCPP_WARN(get_logger(),
          "   ⏰ Timeout (%.0fs) — beam never reached ball_x=%.3f",
          hold_timeout_,ball_x);
        publishFailed(); return false;
      }

      tf2::Vector3 tip;
      if (!getBeamTipWorld(tip)) continue;
      double tip_x=tip.x();

      if (polls%25==0)
        RCLCPP_INFO(get_logger(),
          "   ⏳ poll %d (%.1fs): tip_x=%.3f  ball_x=%.3f  Δ=%.1fmm",
          polls,elapsed,tip_x,ball_x,(tip_x-ball_x)*1000);

      if (tip_x > ball_x + beam_x_tolerance_){
        continue;   // still approaching
      } else if (tip_x >= ball_x - beam_x_tolerance_){
        // HIT
        successful_targets_++;
        RCLCPP_INFO(get_logger(),
          "   ✅ BEAM HIT (poll %d, %.2fs): tip_x=%.3f  ball_x=%.3f  dx=%.1fmm  rate=%.1f%%",
          polls,elapsed,tip_x,ball_x,
          std::fabs(tip_x-ball_x)*1000,
          100.*successful_targets_/total_targets_);
        std_msgs::msg::Bool m; m.data=true; beam_done_pub_->publish(m);
        return true;
      } else {
        // Overshot
        RCLCPP_WARN(get_logger(),
          "   ❌ OVERSHOT (poll %d, %.2fs): tip_x=%.3f  ball_x=%.3f  dx=%.1fmm",
          polls,elapsed,tip_x,ball_x,std::fabs(tip_x-ball_x)*1000);
        publishFailed(); return false;
      }
    }
  }

  void publishFailed()
  { std_msgs::msg::Bool m; m.data=true; beam_failed_pub_->publish(m); }

  // ════════════════════════════════════════════════════════════════════════
  // targetCallback — threads out executeTarget so hold loop doesn't block
  // ════════════════════════════════════════════════════════════════════════
  void targetCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (!moveit_ready_){ RCLCPP_WARN(get_logger(),"⏳ Not ready"); return; }

    if (execution_active_.exchange(true)){
      RCLCPP_WARN(get_logger(),"⚠️  Still executing — dropping incoming target");
      return;
    }

    // Resolve T exactly as doc-13 (TF for Y/Z, works for planning)
    tf2::Vector3 T_snap;
    if (msg->header.frame_id.empty()||msg->header.frame_id==WORLD_FRAME){
      T_snap.setValue(msg->pose.position.x,msg->pose.position.y,msg->pose.position.z);
    } else {
      try {
        auto tf=tf_buffer_->transform(*msg,WORLD_FRAME,tf2::durationFromSec(0.5));
        T_snap.setValue(tf.pose.position.x,tf.pose.position.y,tf.pose.position.z);
      } catch(const tf2::TransformException& e){
        RCLCPP_ERROR(get_logger(),"TF error: %s",e.what());
        publishFailed(); execution_active_.store(false); return;
      }
    }

    RCLCPP_INFO(get_logger(),
      "📥 T_plan=[%.3f,%.3f,%.3f]  (ball snapped in 50ms)",
      T_snap.x(),T_snap.y(),T_snap.z());

    // Detach: 50 ms sleep lets /target_world_pos (published simultaneously
    // with /target_pose by detector) be processed before we snap ball_world.
    std::thread([this,T_snap](){
      std::this_thread::sleep_for(std::chrono::milliseconds(50));

      tf2::Vector3 ball_snap;
      bool ball_valid;
      { std::lock_guard<std::mutex> lk(ball_mutex_);
        ball_snap=latest_ball_world_; ball_valid=ball_world_received_; }

      RCLCPP_INFO(get_logger(),
        "🔵 ball_snap=[%.3f,%.3f,%.3f]%s",
        ball_snap.x(),ball_snap.y(),ball_snap.z(),
        ball_valid?"":" ⚠️ no ball_world");

      executeTarget(T_snap,ball_snap,ball_valid);
      execution_active_.store(false);
    }).detach();
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
  auto node=std::make_shared<BeamPointingRRTStarNode>();
  // SingleThreadedExecutor avoids the "already added to executor" crash that
  // MultiThreadedExecutor triggers with PlanningSceneInterface's internal nodes.
  // Thread safety for the hold-phase loop is handled by the detached std::thread
  // + mutexes, not by the executor thread pool.
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}