from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from pathlib import Path
import os


def generate_launch_description():
    ld = LaunchDescription()

    # --- Args ---
    ld.add_action(DeclareLaunchArgument("use_sim_time", default_value="true"))
    ld.add_action(DeclareLaunchArgument('x', default_value='0.0'))
    ld.add_action(DeclareLaunchArgument('y', default_value='0.0'))
    ld.add_action(DeclareLaunchArgument('z', default_value='0.0'))
    
    # Real-time system parameters
    ld.add_action(DeclareLaunchArgument('normal_speed', default_value='0.8'))
    ld.add_action(DeclareLaunchArgument('tracking_speed', default_value='0.3'))
    ld.add_action(DeclareLaunchArgument('joint_control_rate', default_value='20.0'))
    
    use_sim = {"use_sim_time": True}

    # --- Resource paths ---
    sim_share = get_package_share_directory('sim')
    meshes_root = os.path.join(sim_share, 'meshes')
    model_path = str(Path.home() / ".ignition/gazebo/models")

    ld.add_action(SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=f"{model_path}:{sim_share}:" + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    ))
    ld.add_action(SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=f"{model_path}:{sim_share}:" + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ))

    # --- Default world ---
    default_world = os.path.join(str(Path.home()), "/home/alex/tese_ws/src/sim/worlds/sensors.world.sdf")

    ld.add_action(DeclareLaunchArgument(
        'gz_args',
        default_value=f'-r {default_world}',
        description='Arguments passed to gz_sim'
    ))

    # --- Gazebo ---
    gazebo_launch_file = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
    )
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={'gz_args': LaunchConfiguration('gz_args')}.items()
    ))

    # --- MoveIt config (still needed for robot description) ---
    moveit_config = (
        MoveItConfigsBuilder("ur3", package_name="ur3_moveit_config")
        .robot_description(file_path="config/ur.urdf.xacro")
        .robot_description_semantic(file_path="config/ur.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
            publish_planning_scene=True
        )
        .to_moveit_configs()
    )

    # --- robot_description ---
    urdf_xacro = PathJoinSubstitution(
        [FindPackageShare('ur3_moveit_config'), 'config', 'ur.urdf.xacro']
    )
    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', urdf_xacro, ' mesh_root:=', sim_share]),
            value_type=str
        )
    }

    # --- Spawn model ---
    def read_swincar_ur3_sdf():
        p = Path.home() / ".ignition/gazebo/models/swincar_ur3/model.sdf"
        return p.read_text()

    swincar_ur3_sdf = read_swincar_ur3_sdf()

    ld.add_action(Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'swincar_ur3', '-string', swincar_ur3_sdf],
        output='screen'
    ))

    # --- TF publishers ---
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_world_swincar',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'swincar_base'],
        parameters=[use_sim],
        output='screen'
    ))

    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_swincar_ur3_mount',
        arguments=[
            LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('z'),
            '0', '0', '0', 'swincar_base', 'base_link'
        ],
        parameters=[use_sim],
        output='screen'
    ))

    # --- Bridges ---
    cmd_vel_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/swincar/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/swincar/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
        ],
        output='screen'
    )
    ld.add_action(cmd_vel_bridge)
    
    rgbd_camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/camera_info'
            '@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/image'
            '@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/depth_image'
            '@sensor_msgs/msg/Image@ignition.msgs.Image',
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/points'
            '@sensor_msgs/msg/PointCloud2@ignition.msgs.PointCloudPacked',
        ],
        output='screen'
    )
    ld.add_action(rgbd_camera_bridge)
    
    # --- Ground truth pose bridge ---
    pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/model/swincar_ur3/pose@geometry_msgs/msg/PoseStamped@ignition.msgs.Pose',
        ],
        output='screen'
    )
    ld.add_action(pose_bridge)

    # --- robot_state_publisher ---
    ld.add_action(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description, use_sim],
        output='screen'
    ))

    # --- RViz2 (optional, comment out if not needed) ---
    ld.add_action(Node(
        package='rviz2',
        executable='rviz2',
        arguments=[
            '-d',
            os.path.join(
                get_package_share_directory("ur3_moveit_config"),
                "config",
                "moveit.rviz"
            )
        ],
        parameters=[
            robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            use_sim
        ],
        output='screen'
    ))

    # --- Controllers ---
    ld.add_action(TimerAction(period=2.4, actions=[Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
        parameters=[use_sim]
    )]))
    
    ld.add_action(TimerAction(period=2.8, actions=[Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen',
        parameters=[use_sim]
    )]))

    # ========================================================================
    # REAL-TIME CONTINUOUS BEAM SYSTEM
    # ========================================================================
    # This is the ONLY node you need for:
    # - Detection
    # - Tracking  
    # - Navigation
    # - Direct UR3 joint control (NO separate beam_pointing_precise!)
    
    ld.add_action(TimerAction(period=3.5, actions=[Node(
        package='sim',  # UPDATE: Change to your package name
        executable='continuous_beam_pointing.py',
        name='continuous_beam_pointing',
        output='screen',
        parameters=[{
            # Camera
            'rgb_topic': '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/image',
            'depth_topic': '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/depth_image',
            'camera_info_topic': '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/camera_info',
            
            # Robot state
            'use_ground_truth': True,
            'ground_truth_topic': '/model/swincar_ur3/pose',
            'odom_topic': '/swincar/odom',
            'joint_states_topic': '/joint_states',
            
            # Base link offset
            'base_link_offset_x': 0.0,
            'base_link_offset_y': 0.2,
            'base_link_offset_z': 0.35,
            
            # Output topics
            'cmd_vel_topic': '/swincar/cmd_vel',
            'joint_trajectory_topic': '/joint_trajectory_controller/joint_trajectory',
            
            # Detection
            'camera_hfov': 0.6,  # RealSense D435: 55° horizontal FOV
            'association_radius': 0.08,
            'min_observations': 2,
            'track_timeout': 2.0,
            'min_blob_area': 200.0,
            
            # Trigger zone - when to start tracking
            'trigger_min_y': 0.6,
            'trigger_max_y': 1.5,
            'trigger_min_x': -0.3,
            'trigger_max_x': 0.3,
            
            # Navigation
            'x_start': -1.0,
            'x_goal': -99.0,
            'y_target': 0.0,
            'goal_tolerance': 0.5,
            
            # Speed control
            'normal_speed': LaunchConfiguration('normal_speed'),      # Cruising speed
            'tracking_speed': LaunchConfiguration('tracking_speed'),  # Speed while pointing
            'max_angular_speed': 1.5,
            
            # Pure pursuit
            'lookahead_distance': 3.0,
            'lateral_gain': 2.0,
            'heading_gain': 1.5,
            'direction_sign': -1.0,
            
            # UR3 real-time control
            'joint_control_rate': LaunchConfiguration('joint_control_rate'),  # Hz
            'pointing_p_gain': 2.0,        # Proportional gain for pointing
            'max_joint_velocity': 1.0,     # Max joint velocity (rad/s)
            
        }, use_sim]
    )]))

    # ========================================================================
    # NOTE: We do NOT launch beam_pointing_precise or move_group!
    # The real-time system controls joints directly.
    # ========================================================================

    return ld