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
    use_sim = {"use_sim_time": True}

    # --- Resource paths (optional if you use file://) ---
    sim_share = get_package_share_directory('sim')
    meshes_root = os.path.join(sim_share, 'meshes')
    ld.add_action(SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=f"{sim_share}:" + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    ))
    ld.add_action(SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=f"{sim_share}:" + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ))

    # --- Gazebo ---
    gazebo_launch_file = os.path.join(
        get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
    )
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={'gz_args': '-r'}.items()
    ))

    # --- MoveIt base config ---
    moveit_config = (
        MoveItConfigsBuilder("ur3", package_name="ur3_moveit_config")
        .robot_description(file_path="config/ur.urdf.xacro")         # this xacro must include your camera.xacro
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

    # --- Override robot_description to inject absolute mesh root for camera ---
    # Pass mesh_root to your xacro so it emits file:///... paths that gz understands.
    urdf_xacro = PathJoinSubstitution([FindPackageShare('ur3_moveit_config'), 'config', 'ur.urdf.xacro'])
    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', urdf_xacro, ' mesh_root:=', sim_share]),
            value_type=str
        )
    }

    # --- Swincar (static) ---
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

    ld.add_action(Node(
        package='ros_gz_sim', executable='create',
        arguments=['-name', 'swincar', '-string', swincar_sdf_static],
        output='screen'
    ))

    # --- UR3 spawn from robot_description ---
    ld.add_action(Node(
        package='ros_gz_sim', executable='create',
        arguments=['-name','ur3',
                   '-x', LaunchConfiguration('x'),
                   '-y', LaunchConfiguration('y'),
                   '-z', LaunchConfiguration('z'),
                   '-topic','robot_description'],
        output='screen'
    ))

    # --- Static TF ---
    ld.add_action(Node(
        package='tf2_ros', executable='static_transform_publisher', name='static_tf_ur3',
        arguments=[LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('z'),
                   '0', '0', '0', 'world', 'base_link'],
        output='screen',
        parameters=[use_sim]
    ))

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
        parameters=[mg_params, moveit_config.robot_description_kinematics,
                    os.path.join(sim_share, 'config', 'incremental_pipeline.yaml')],
        output='screen'
    ))

    # --- RViz2 ---
    ld.add_action(Node(
        package='rviz2', executable='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory("ur3_moveit_config"), "config", "moveit.rviz")],
        parameters=[robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    use_sim],
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

    # --- Pointing servo ---
    ld.add_action(Node(
        package="ur3_pointing", executable="pointing_servo", name="pointing_servo",
        output="screen",
        parameters=[robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    os.path.join(sim_share, 'config', 'servo_params.yaml'),
                    use_sim],
        remappings=[("target_pose", "/target_pose")],
    ))

    return ld
