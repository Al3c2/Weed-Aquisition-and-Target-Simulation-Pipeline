from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
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

    use_sim = {"use_sim_time": True}

    # --- Gazebo ---
    gazebo_launch_file = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={'gz_args': '-r'}.items()
    )
    ld.add_action(gazebo)

    # --- MoveIt config ---
    moveit_config = (
        MoveItConfigsBuilder("ur3", package_name="ur3_moveit_config")  # your package
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

    # --- Swincar as STATIC (never moves) ---
    def read_swincar_sdf_as_static():
        p = Path.home() / ".ignition/gazebo/models/swincar/model.sdf"
        sdf = p.read_text()
        if "<static>" in sdf:
            sdf = sdf.replace("<static>false</static>", "<static>true</static>")
            sdf = sdf.replace("<static>0</static>", "<static>true</static>")
        else:
            sdf = sdf.replace("</model>", "  <static>true</static>\n</model>", 1)
        return sdf

    swincar_sdf_static = read_swincar_sdf_as_static()

    spawn_swincar = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-name', 'swincar', '-string', swincar_sdf_static],
        output='screen'
    )

    # --- UR3 spawn from robot_description ---
    spawn_ur3 = Node(
        package='ros_gz_sim', executable='create',
        arguments=[
            '-name', 'ur3',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-topic', 'robot_description'
        ],
        output='screen'
    )

    # --- Static TF: world -> base_link (RViz aligns with Gazebo) ---
    static_tf_ur3 = Node(
        package='tf2_ros', executable='static_transform_publisher', name='static_tf_ur3',
        arguments=[LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('z'),
                   '0', '0', '0', 'world', 'base_link'],
        output='screen',
        parameters=[use_sim]
    )
    ld.add_action(static_tf_ur3)

    # Spawn order (give Gazebo a moment to bring up /create and controller manager)
    ld.add_action(TimerAction(period=1.0, actions=[spawn_swincar]))
    ld.add_action(TimerAction(period=1.6, actions=[spawn_ur3]))

    # --- robot_state_publisher (publishes robot_description) ---
    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[moveit_config.robot_description, use_sim],
        output='screen'
    )
    ld.add_action(robot_state_publisher)

    # --- MoveIt move_group ---
    move_group_node = Node(
        package='moveit_ros_move_group', executable='move_group',
        parameters=[
            moveit_config.to_dict(),
            moveit_config.robot_description_kinematics,
            os.path.join(get_package_share_directory('sim'), 'config', 'incremental_pipeline.yaml'),
            use_sim,
        ],
        output='screen'
    )
    ld.add_action(move_group_node)

    # --- RViz2 ---
    rviz_node = Node(
        package='rviz2', executable='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory("ur3_moveit_config"), "config", "moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            use_sim,
        ],
        output='screen'
    )
    ld.add_action(rviz_node)

    # --- Spawners (talk to /controller_manager owned by gz_ros2_control) ---
    joint_state_broadcaster = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
        parameters=[use_sim]
    )
    joint_trajectory_controller = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen',
        parameters=[use_sim]
    )

    # Start spawners a bit AFTER both models spawn so gz_ros2_control is up
    ld.add_action(TimerAction(period=2.4, actions=[joint_state_broadcaster]))
    ld.add_action(TimerAction(period=2.8, actions=[joint_trajectory_controller]))

    # # --- Pointing SERVO (publishes tiny JTC steps) ---
    cfg = os.path.join(get_package_share_directory('sim'), 'config', 'servo_params.yaml')

    pointing_servo = Node(
        package="ur3_pointing",
        executable="pointing_servo",
        name="pointing_servo",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            cfg,
            use_sim,
        ],
        remappings=[("target_pose", "/target_pose")],
    )
    ld.add_action(pointing_servo)

  

    # EITHER run the planner (adds Swincar + consumes /target_pose)
    # pointing_planner = Node(
    #     package='ur3_pointing', executable='pointing_planner', name='pointing_planner',
    #     parameters=[
    #         moveit_config.robot_description,
    #         moveit_config.robot_description_semantic,
    #         moveit_config.robot_description_kinematics,
    #         # this pulls in the incremental pipeline into move_group
    #         os.path.join(get_package_share_directory('sim'), 'config', 'incremental_pipeline.yaml'),
    #         use_sim,
    #     ],
    #     output='screen'
    # )
    # ld.add_action(pointing_planner)



    return ld
