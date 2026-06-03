from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

import os

def generate_launch_description():
    ld = LaunchDescription()

    # Joint controller config
    joint_controllers_file = os.path.join(
        get_package_share_directory('sim'), 'config', 'ur3_controllers.yaml'
    )

    # Gazebo (Ignition) world
    gazebo_launch_file = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
    )

    # MoveIt config builder for ur3
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

    # Position args
    x_arg = DeclareLaunchArgument('x', default_value='0')
    y_arg = DeclareLaunchArgument('y', default_value='0')
    z_arg = DeclareLaunchArgument('z', default_value='0')

    # Gazebo sim
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={'gz_args': '-r'}.items()
    )

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'ur3',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-topic', 'robot_description'
        ],
        output='screen'
    )

    # Controller manager
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[moveit_config.robot_description, joint_controllers_file],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[moveit_config.robot_description],
        output='screen'
    )

    # MoveGroup node
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), {"use_sim_time": True}],
    )

    # RViz2
    rviz_config_path = os.path.join(
        get_package_share_directory("ur3_moveit_config"),
        "config",
        "moveit.rviz"
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
    )

    # Spawners
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    joint_trajectory_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    # Delay actions
    delay_joint_state = RegisterEventHandler(
        OnProcessStart(
            target_action=controller_manager_node,
            on_start=[joint_state_broadcaster],
        )
    )
    delay_arm_controller = RegisterEventHandler(
        OnProcessStart(
            target_action=joint_state_broadcaster,
            on_start=[joint_trajectory_controller],
        )
    )
    delay_rviz = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_state_publisher,
            on_start=[rviz_node],
        )
    )

    # Launch order
    ld.add_action(x_arg)
    ld.add_action(y_arg)
    ld.add_action(z_arg)
    ld.add_action(gazebo)
    ld.add_action(controller_manager_node)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_robot)
    ld.add_action(move_group_node)
    ld.add_action(delay_joint_state)
    ld.add_action(delay_arm_controller)
    ld.add_action(delay_rviz)

    return ld
