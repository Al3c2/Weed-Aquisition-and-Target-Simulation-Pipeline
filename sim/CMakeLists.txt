cmake_minimum_required(VERSION 3.8)
project(sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(ament_cmake_python REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY launch urdf config meshes worlds rviz
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*/log" EXCLUDE
  PATTERN "*/build" EXCLUDE
  PATTERN "*/install" EXCLUDE
)
ament_python_install_package(color_detector)

install(PROGRAMS
  color_detector/color_detector.py
  color_detector/blue_target_to_pose.py
  color_detector/swincar_row_driver.py
  color_detector/detection_evaluator.py  
  color_detector/detection_evaluator_predictive.py
  color_detector/color_detector_predictive.py
  color_detector/swincar_line_follower_adaptive.py
  continuous_beam_pointing/continuous_beam_pointing.py


  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
