#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("incremental_cli");

  // Params
  std::string group = node->declare_parameter<std::string>("group", "ur3_manipulator");
  std::string pipeline = node->declare_parameter<std::string>("pipeline", "incremental");
  std::string planner_id = node->declare_parameter<std::string>("planner_id", "incremental");
  std::vector<double> joints = node->declare_parameter<std::vector<double>>("joints", {});
  std::string named = node->declare_parameter<std::string>("named", "");  // optional MoveIt named target
  bool execute = node->declare_parameter<bool>("execute", true);

  RCLCPP_INFO(node->get_logger(), "Using pipeline=%s planner_id=%s group=%s",
              pipeline.c_str(), planner_id.c_str(), group.c_str());

  // MoveGroupInterface
  moveit::planning_interface::MoveGroupInterface mgi(node, group);
  mgi.setPlanningPipelineId(pipeline);
  mgi.setPlannerId(planner_id);
  mgi.setPlanningTime(5.0);

  // Target
  if (!named.empty()) {
    RCLCPP_INFO(node->get_logger(), "Setting named target: %s", named.c_str());
    mgi.setNamedTarget(named);
  } else if (!joints.empty()) {
    // Expect exactly the group's variable count (UR3: 6)
    const auto var_names = mgi.getJointNames();
    if (joints.size() != var_names.size()) {
      RCLCPP_ERROR(node->get_logger(), "Expected %zu joints for group %s, got %zu",
                   var_names.size(), group.c_str(), joints.size());
      rclcpp::shutdown(); return 2;
    }
    mgi.setJointValueTarget(joints);
  } else {
    RCLCPP_ERROR(node->get_logger(), "No goal provided. Pass -p joints:='[...6 values...]' or -p named:='ready'");
    rclcpp::shutdown(); return 2;
  }

  // Plan
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto ok = (mgi.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!ok) {
    RCLCPP_ERROR(node->get_logger(), "Planning failed.");
    rclcpp::shutdown(); return 3;
  }
  RCLCPP_INFO(node->get_logger(), "Planning succeeded. Trajectory has %zu points.",
              plan.trajectory_.joint_trajectory.points.size());

  if (execute) {
    auto ex = (mgi.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(node->get_logger(), ex ? "Executed." : "Execution failed.");
    rclcpp::shutdown(); return ex ? 0 : 4;
  } else {
    rclcpp::shutdown(); return 0;
  }
}
