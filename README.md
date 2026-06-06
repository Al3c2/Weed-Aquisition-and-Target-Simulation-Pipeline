# Swincar-UR3 Autonomous Inspection Pipeline

Autonomous mobile manipulation pipeline for target detection, navigation, and precise beam pointing, built on a Swincar mobile robot + UR3 manipulator in **ROS 2 Humble**, **Gazebo Harmonic**, and **MoveIt 2**.

Developed as part of an MSc thesis on autonomous robotic inspection systems. The robot drives along a track, an RGB-D camera detects blue targets, the targets are localized in the world frame, and the UR3 arm points a beam at each one with sub-centimetre precision.

---

## Packages

The workspace is split into two main packages:

### `sim`
Everything needed to run the simulation and perception/navigation stack:
- Robot model (URDF/xacro) — the combined `swincar_ur3` nested model
- World files, RGB-D camera, TF, and ROS ↔ Gazebo bridges
- Launch files for the simulation and the two operating modes
- The detector nodes (`color_detector`, `color_detector_predictive`)
- The navigation nodes (`swincar_line_follower`, `swincar_line_follower_adaptive`, `swincar_row_driver`)
- The evaluation nodes and ground-truth data (`gt_world.csv`)

### `ur3_pointing`
The arm-side beam-pointing stack:
- `Beam_pointing_precise` — Cartesian, collision-aware precise pointing
- `Beamastar` — Rttstar aproach

---

## Two Operating Modes

The pipeline runs in one of two modes:

**1. Start-stop mode** — the robot drives, stops at each detected target, points, then resumes. Simpler and more precise per target.

**2. Continuous / predictive mode** — the robot keeps moving while the arm predicts and sweeps to hit targets on the fly. Higher throughput, uses the `*_predictive` and `*_adaptive` nodes.

---

## Setup

Source ROS 2 and the workspace in every terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/tese_ws/install/setup.bash
```

---

## Running — Start-Stop Mode

**Launch the simulation:**
```bash
ros2 launch sim moveit_planner_camera_moving.launch.py
```

**Run the detector:**
```bash
ros2 run sim color_detector.py --ros-args \
  -p rgb_topic:=/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/image \
  -p depth_topic:=/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/depth_image \
  -p camera_info_topic:=/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/camera_info \
  -p camera_frame:=camera_optical_link \
  -p robot_frame:=base_link \
  -p world_frame:=world \
  -p use_ground_truth:=true \
  -p ground_truth_topic:=/model/swincar_ur3/pose \
  -p ground_truth_frame_id:=empty
```

**Run the navigation controller:**
```bash
ros2 run sim swincar_line_follower.py --ros-args \
  -p pose_topic:=/model/swincar_ur3/pose \
  -p target_topic:=/blue_target_primary \
  -p beam_done_topic:=/beam_task_done \
  -p x_goal:=-54.0 \
  -p base_speed:=0.1 \
  -p min_drive_before_stop:=0.0 \
  -p accel_rate:=0.01
```

**Run the beam-pointing node:**
```bash
ros2 run ur3_pointing pointing_planner
ros2 run ur3_pointing Beam_pointing_precise
```

---

## Running — Continuous / Predictive Mode

**Launch the predictive pipeline:**
```bash
ros2 launch sim predictive_pointing_launch.py
```

**Run the predictive detector:**
```bash
ros2 run sim color_detector_predictive.py --ros-args \
  -p rgb_topic:=/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/image \
  -p depth_topic:=/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/depth_image \
  -p camera_info_topic:=/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/camera_info \
  -p camera_frame:=camera_optical_link \
  -p robot_frame:=base_link \
  -p world_frame:=world \
  -p use_ground_truth:=true \
  -p ground_truth_topic:=/model/swincar_ur3/pose \
  -p ground_truth_frame_id:=empty
```

**Run the adaptive line follower:**
```bash
ros2 run sim swincar_line_follower_adaptive.py --ros-args \
  -p pose_topic:=/model/swincar_ur3/pose \
  -p target_topic:=/blue_target_primary \
  -p beam_done_topic:=/beam_task_done \
  -p x_goal:=-54.0
```

---

## System Flow

```
Detector publishes target
       ↓
UR3 attempts to point
       ↓
    ┌──┴──┐
    ↓     ↓
SUCCESS  FAILED
(< 10mm) (≥ 10mm or exec fail)
    ↓     ↓
 beam_done  beam_failed
    ↓     ↓
    └──┬──┘
       ↓
Detector unlocks → Line follower resumes → Next target
```

---

## Evaluation

Ground-truth target positions live in `gt_world.csv`. Each published target is matched against the ground truth within a configurable threshold, producing position-error, success/failure, timing, and retry statistics.

**Start-stop evaluator:**
```bash
ros2 run sim detection_evaluator.py --ros-args \
  -p gt_world_csv:=/home/alex/tese_ws/src/sim/color_detector/gt_world.csv \
  -p target_pose_topic:=/target_pose \
  -p use_ground_truth_pose:=true \
  -p ground_truth_topic:=/model/swincar_ur3/pose \
  -p ground_truth_frame_id:=empty \
  -p summary_csv:=/home/alex/tese_ws/src/sim/color_detector/eval_summary.csv \
  -p detailed_csv:=/home/alex/tese_ws/src/sim/color_detector/eval_detailed.csv
```

**Predictive evaluator:**
```bash
ros2 run sim detection_evaluator_predictive.py --ros-args \
  -p gt_world_csv:=/home/alex/tese_ws/src/sim/color_detector/gt_world.csv \
  -p target_pose_topic:=/target_pose \
  -p use_ground_truth_pose:=true \
  -p ground_truth_topic:=/model/swincar_ur3/pose \
  -p ground_truth_frame_id:=empty \
  -p summary_csv:=/home/alex/tese_ws/src/sim/color_detector/eval_summary.csv \
  -p detailed_csv:=/home/alex/tese_ws/src/sim/color_detector/eval_detailed.csv
```

**Generate plots:**
```bash
python3 plot_eval_results.py eval_detailed.csv -o plots/
```

---

## Coordinate Frames

```
world
└── swincar_base
    └── base_link
        └── camera_optical_link
```

Full quaternion-based transformations are performed between the camera, robot, and world frames so target localization stays accurate on uneven terrain.

### Ball-Center Localization

The detector estimates the **center** of a ball from the depth reading of its **surface**. The correct method extends along the camera ray by one ball radius, rather than adding the radius to the Z component:

```python
def surface_to_center_correct(u, v, depth):
    # 1. Unit ray through the pixel
    ray = pixel_to_ray(u, v)            # normalized direction

    # 2. Surface point (depth gives the Z component)
    t_surface = depth / ray[2]
    surface_point = ray * t_surface

    # 3. Extend ALONG THE RAY by ball_radius to reach the center
    center_point = surface_point + ray * ball_radius
    return center_point
```

```
         Camera
            \
             \  ray direction
              \
               * Surface (depth sensor reading)
                \
                 * Ball center (surface + radius ALONG RAY)

WRONG:   add radius to Z, recompute X,Y  → center shifts sideways
CORRECT: extend along the ray            → center stays on the line of sight
```

---

## Useful Commands

**Manual drive (Twist command):**
```bash
ros2 topic pub /swincar/cmd_vel geometry_msgs/msg/Twist \
  "{ linear: {x: 0.25, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0} }"
```

**View the camera feed:**
```bash
ros2 run rqt_image_view rqt_image_view
```

**Manual pose bridge (if not started by a launch file):**
```bash
ros2 run ros_gz_bridge parameter_bridge \
  '/world/empty/dynamic_pose/info@ros_gz_interfaces/msg/Pose_V@ignition.msgs.Pose_V'
```

---

## Technologies

- **Robotics:** ROS 2 Humble, Gazebo Harmonic / Ignition, MoveIt 2, TF2
- **Languages:** Python, C++
- **Libraries:** OpenCV, NumPy, SciPy, Matplotlib

---

## Future Work

- Real-robot deployment
- SLAM integration
- Multi-target prioritization
- Deep-learning-based detection
- Improved terrain-aware navigation
- GPU acceleration

---

## License

Released for academic and research purposes, check license fiel for more info. 
