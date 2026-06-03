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
    ld.add_action(DeclareLaunchArgument('x', default_value='0.0'))  # still used for TF offset if you want
    ld.add_action(DeclareLaunchArgument('y', default_value='0.0'))
    ld.add_action(DeclareLaunchArgument('z', default_value='0.0'))
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

    # --- Default world (with Sensors system) ---
    # Adjust this path if you move sensor_world.sdf into your sim package.
    default_world = os.path.join(str(Path.home()), "/home/alex/tese_ws/src/sim/worlds/sensor_world.sdf")

    ld.add_action(DeclareLaunchArgument(
        'gz_args',
        default_value=f'-r {default_world}',
        description='Arguments passed to gz_sim (e.g. "-r /path/to/world.sdf")'
    ))

    # --- Gazebo ---
    gazebo_launch_file = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
    )
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={
            'gz_args': LaunchConfiguration('gz_args')
        }.items()
    ))

    # --- MoveIt base config ---
    moveit_config = (
        MoveItConfigsBuilder("ur3", package_name="ur3_moveit_config")
        .robot_description(file_path="config/ur.urdf.xacro")  # UR3 + camera
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

    # --- robot_description: xacro + mesh_root ---
    urdf_xacro = PathJoinSubstitution(
        [FindPackageShare('ur3_moveit_config'), 'config', 'ur.urdf.xacro']
    )
    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', urdf_xacro, ' mesh_root:=', sim_share]),
            value_type=str
        )
    }

    # === COMBINED MODEL SPAWN =====================================
    # Now we spawn ONE model: swincar_ur3 (swincar + UR3 as nested models)

    def read_swincar_ur3_sdf():
        p = Path.home() / ".ignition/gazebo/models/swincar_ur3/model.sdf"
        sdf = p.read_text()
        return sdf

    swincar_ur3_sdf = read_swincar_ur3_sdf()

    ld.add_action(Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'swincar_ur3', '-string', swincar_ur3_sdf],
        output='screen'
    ))

    # NOTE: removed separate UR3 spawn from robot_description,
    # because UR3 is now inside swincar_ur3 in Ignition.
    # ===============================================================

    # --- TF: world -> swincar_base (for MoveIt planning frame) ---
    # swincar_base is still the base link inside the nested 'swincar' model.
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_world_swincar',
        arguments=[
            '0', '0', '0',
            '0', '0', '0',
            'world', 'swincar_base'
        ],
        parameters=[use_sim],
        output='screen'
    ))

    # --- OPTIONAL TF: swincar_base -> base_link ---
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_swincar_ur3_mount',
        arguments=[
            LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('z'),
            '0', '0', '0',
            'swincar_base',   # parent frame (vehicle)
            'base_link'       # child frame (UR3 root)
        ],
        parameters=[use_sim],
        output='screen'
    ))

    # --- ROS <-> Gazebo bridges for Swincar cmd_vel & odom ---
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
    
    # --- ROS <-> Gazebo bridges for UR3 camera ---
      # --- ROS <-> Gazebo bridges for UR3 RGBD camera ---
    rgbd_camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Camera intrinsics
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/camera_info'
            '@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',

            # RGB image
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/image'
            '@sensor_msgs/msg/Image@ignition.msgs.Image',

            # Depth image
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/depth_image'
            '@sensor_msgs/msg/Image@ignition.msgs.Image',

            # Point cloud
            '/world/empty/model/swincar_ur3/model/ur3/link/base_link/sensor/rgbd_camera/points'
            '@sensor_msgs/msg/PointCloud2@ignition.msgs.PointCloudPacked',
        ],
        output='screen'
    )
    ld.add_action(rgbd_camera_bridge)


    # --- robot_state_publisher ---
    ld.add_action(Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[robot_description, use_sim],
        output='screen'
    ))

    # --- move_group ---
    mg_params = moveit_config.to_dict()
    mg_params.update(robot_description)  # ensure our robot_description wins
    mg_params.update(use_sim)
    ld.add_action(Node(
        package='moveit_ros_move_group', executable='move_group',
        parameters=[
            mg_params,
            moveit_config.robot_description_kinematics,
            os.path.join(sim_share, 'config', 'incremental_pipeline.yaml')
        ],
        output='screen'
    ))

    # --- RViz2 ---
    ld.add_action(Node(
        package='rviz2', executable='rviz2',
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
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen', parameters=[use_sim]
    )]))
    ld.add_action(TimerAction(period=2.8, actions=[Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen', parameters=[use_sim]
    )]))

    # # --- Pointing servo ---
    # ld.add_action(Node(
    #     package="ur3_pointing", executable="pointing_servo", name="pointing_servo",
    #     output="screen",
    #     parameters=[
    #         robot_description,
    #         moveit_config.robot_description_semantic,
    #         moveit_config.robot_description_kinematics,
    #         os.path.join(sim_share, 'config', 'servo_params.yaml'),
    #         use_sim
    #     ],
    #     remappings=[("target_pose", "/target_pose")],
    # ))
        # --- OMPL beam pointing (one-shot planner) ---
    ld.add_action(Node(
        package="ur3_pointing",
        executable="beam_pointing_ompl",
        name="beam_pointing_ompl",
        output="screen",
        parameters=[
            robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            use_sim
        ],
        remappings=[("target_pose", "/target_pose")],
    ))



    # vineyard_runner = Node(
    #     package='sim',  # e.g., 'sim' or 'vineyard_control'
    #     executable='vineyard_runner',
    #     name='vineyard_runner',
    #     output='screen',
    #     parameters=[{
    #         'cmd_vel_topic': '/swincar/cmd_vel',
    #         'blue_topic': '/blue_target_primary',
    #         'forward_speed': 0.5,
    #         'stop_duration': 4.0,
    #         'min_time_between_stops': 6.0,
    #         'min_target_distance_change': 0.05,
    #     }],
    # )

    # ld.add_action(vineyard_runner)


    return ld
