Swincar-UR3 Autonomous simulation Pipeline

Autonomous mobile manipulation pipeline for target detection, navigation, beam pointing, and evaluation using a Swincar mobile robot + UR3 manipulator in ROS 2 + Gazebo Harmonic + MoveIt 2.

This repository contains the full software stack developed for my MSc thesis, including:

Mobile robot simulation
RGB-D target detection
Autonomous navigation / line following
Precise beam pointing with MoveIt 2
Ground-truth based evaluation framework
Performance plotting and metrics generation
System Overview

The pipeline integrates multiple ROS 2 nodes into a complete autonomous inspection workflow:

Gazebo Simulation
        ↓
RGB-D Camera Streams
        ↓
Color / Ball Detector
        ↓
Target Localization
        ↓
Swincar Navigation Controller
        ↓
Robot Stop Confirmation
        ↓
UR3 Beam Pointing (MoveIt 2)
        ↓
Success / Failure Feedback
        ↓
Evaluation + Plot Generation
Main Components
1. Gazebo + MoveIt Launch System

Launches:

Gazebo Harmonic simulation
Swincar + UR3 combined model
ROS ↔ Gazebo bridges
RGB-D camera bridges
TF publishers
MoveIt 2 configuration
File
moveit_planner_camera_moving.launch.py
Features
Combined nested robot model (swincar_ur3)
RGB-D camera integration
Ground-truth pose bridging
TF synchronization
ROS-Gazebo communication bridges
2. RGB-D Ball / Target Detector
File
color_detector.py
Features
RGB + depth fusion
3D target localization
Target tracking
Re-acquisition logic
Retry handling
World-frame transformation using robot orientation
Duplicate target filtering
Publication of:
/target_pose
/blue_target_primary
/all_tracked_targets
Detection Pipeline
Detect colored target in RGB image
Estimate depth
Convert image coordinates → 3D camera coordinates
Transform to robot/world frame
Track and filter targets
Publish target pose
3. Swincar Autonomous Navigation
File
swincar_line_follower.py
Features
Ground-truth pose control
Smooth acceleration/deceleration
Target-triggered stopping
Beam synchronization
Resume after beam completion
Goal tracking
Responsibilities
Drive the mobile robot
Stop when detector publishes a target
Wait for UR3 beam completion
Resume mission automatically
4. Precise UR3 Beam Pointing
File
Beam_pointing_precise.cpp
Features
MoveIt 2 motion planning
Cartesian beam approach
Collision-aware planning
Ground-truth collision tracking
Full quaternion-based coordinate transforms
Adaptive planning parameters
Retry logic
Precision verification
Important Improvements
Ground-truth pose integration
Accurate moving-platform transforms
Dynamic collision object updates
Beam error verification
5. Detection Evaluation Framework
File
detection_evaluator.py
Features
Ground-truth comparison
Position error analysis
Success/failure tracking
Timing metrics
CSV export
Full orientation-aware transformations
Metrics
True positives
False positives
False negatives
Detection accuracy
Position error
Time-to-success
Retry statistics
6. Evaluation Plot Generator
File
plot_eval_results.py
Generates
Detection accuracy histograms
Cumulative detection curves
Success/failure summaries
Timing statistics
Retry statistics
Output Formats
PNG
PDF
Technologies Used
Robotics
ROS 2 Humble
Gazebo Harmonic / Ignition Gazebo
MoveIt 2
TF2
Programming
Python
C++
Libraries
OpenCV
NumPy
SciPy
Matplotlib
Repository Structure
.
├── launch/
│   └── moveit_planner_camera_moving.launch.py
│
├── src/
│   ├── color_detector.py
│   ├── swincar_line_follower.py
│   ├── detection_evaluator.py
│   ├── plot_eval_results.py
│   └── Beam_pointing_precise.cpp
│
├── models/
│   └── swincar_ur3/
│
├── worlds/
│   └── sensors.world.sdf
│
├── data/
│   └── gt_world.csv
│
└── plots/
Running the System
1. Launch Simulation
ros2 launch <your_package> moveit_planner_camera_moving.launch.py
2. Run Detector
ros2 run <your_package> color_detector.py
3. Run Navigation Controller
ros2 run <your_package> swincar_line_follower.py
4. Run Beam Pointing Node
ros2 run <your_package> Beam_pointing_precise
5. Run Evaluation
ros2 run <your_package> detection_evaluator.py
6. Generate Plots
python3 plot_eval_results.py eval_detailed.csv -o plots/
Coordinate Frames

Main frames used:

world
 └── swincar_base
      └── base_link
           └── camera_optical_link

The system performs full quaternion-based transformations between:

Camera frame
Robot frame
World frame

to ensure accurate target localization on uneven terrain.

Evaluation Methodology

Ground-truth target locations are stored in:

gt_world.csv

Each published target is matched against GT targets using a configurable threshold.

Evaluation includes:

Final success per GT target
Retry-aware statistics
Position error computation
Timing analysis
Detection coverage
Key Contributions

This work focuses on:

Autonomous mobile manipulation
RGB-D target localization
Dynamic transform handling on moving platforms
Precision beam pointing
Robust retry and re-acquisition logic
Evaluation of perception-to-action pipelines
Thesis Context

This repository was developed as part of my MSc thesis on autonomous robotic inspection systems using a mobile manipulator platform.

The project combines:

perception,
navigation,
manipulation,
motion planning,
and evaluation

into a fully integrated ROS 2 pipeline.

Future Improvements

Potential future work includes:

Real robot deployment
SLAM integration
Multi-target prioritization
Deep-learning based detection
Improved terrain-aware navigation
GPU acceleration
Author

Alexandre Baptista

MSc Thesis Project
Autonomous Mobile Manipulation and Inspection System

License

This project is released for academic and research purposes.

If you use this work, please cite the associated thesis/publication.
