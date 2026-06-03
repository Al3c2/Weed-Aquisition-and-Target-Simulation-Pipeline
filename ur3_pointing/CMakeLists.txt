cmake_minimum_required(VERSION 3.8)
project(ur3_pointing)

# Release flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# MoveIt 2
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(pointing_servo
  src/pointing_servo.cpp
)
target_include_directories(pointing_servo PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(pointing_servo
  rclcpp
  geometry_msgs
  trajectory_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  geometric_shapes
)

# Install the node binary to lib/${PROJECT_NAME}
install(TARGETS pointing_servo
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install launch/config if you have them
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)
add_executable(beam_pointing_ompl src/beam_pointing_ompl.cpp)
install(TARGETS
  beam_pointing_ompl
  # other executables...
  DESTINATION lib/${PROJECT_NAME}
)
add_executable(Beam_pointing_precise src/Beam_pointing_precise.cpp)
ament_target_dependencies(Beam_pointing_precise
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  tf2
  tf2_ros
  tf2_geometry_msgs
)


install(TARGETS
  Beam_pointing_precise

  DESTINATION lib/${PROJECT_NAME}
)

ament_target_dependencies(beam_pointing_ompl
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  tf2
  tf2_ros
  tf2_geometry_msgs
)
add_executable(Beam_pointing_adaptive src/Beam_pointing_adaptive.cpp)
ament_target_dependencies(Beam_pointing_adaptive
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  tf2
  tf2_ros
  tf2_geometry_msgs
)
add_executable(Beamastar src/Beamastar.cpp)
ament_target_dependencies(Beamastar
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  tf2
  tf2_ros
  tf2_geometry_msgs
)
add_executable(beamastarstoped src/beamastarstoped.cpp)
ament_target_dependencies(beamastarstoped
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  tf2
  tf2_ros
  tf2_geometry_msgs
)
install(TARGETS
  Beam_pointing_adaptive
  Beamastar
  beamastarstoped
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
