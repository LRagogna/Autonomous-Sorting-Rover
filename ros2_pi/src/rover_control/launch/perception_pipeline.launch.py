"""Bring up the full autonomous perception pipeline ON THE RASPBERRY PI.

    camera_node -> perception_node -> action_node -> arduino_bridge_node
                                                  \\-> odometry_node -> /odom

This is the Pi flavour of the pipeline. Same nodes as the dev workspace, but the
defaults suit the real rover:

    * camera source defaults to the Pi CSI camera (picamera2)
    * motors default to the REAL Arduino serial bridge (motor_backend:=arduino)
    * the OpenCV overlay window (viz) is OFF by default — the Pi is headless

Launch it with:

    ros2 launch rover_control perception_pipeline.launch.py

Common overrides (all optional):

    # Bench-test with the fake motor node (just prints values, no Arduino):
    ros2 launch rover_control perception_pipeline.launch.py motor_backend:=fake

    # Point at a different serial port:
    ros2 launch rover_control perception_pipeline.launch.py port:=/dev/ttyUSB0

    # Chase only one class:
    ros2 launch rover_control perception_pipeline.launch.py target_class:=bit

    # If you have a display / VNC and want the overlay window:
    ros2 launch rover_control perception_pipeline.launch.py viz:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    source = LaunchConfiguration('source')
    video_path = LaunchConfiguration('video_path')
    model = LaunchConfiguration('model')
    conf = LaunchConfiguration('conf')
    target_class = LaunchConfiguration('target_class')
    forward_speed = LaunchConfiguration('forward_speed')
    search = LaunchConfiguration('search')
    action_enable = LaunchConfiguration('action')
    motor_backend = LaunchConfiguration('motor_backend')
    port = LaunchConfiguration('port')
    baud = LaunchConfiguration('baud')
    viz = LaunchConfiguration('viz')
    odom = LaunchConfiguration('odom')

    args = [
        DeclareLaunchArgument('source', default_value='picamera2',
                              description='camera source: picamera2 | usb | video'),
        DeclareLaunchArgument('video_path', default_value='',
                              description="video file when source:=video"),
        DeclareLaunchArgument('model', default_value='active_model.pt',
                              description='YOLO weights path or name'),
        DeclareLaunchArgument('conf', default_value='0.35',
                              description='detection confidence cutoff'),
        DeclareLaunchArgument('target_class', default_value='',
                              description="object to chase; '' = any"),
        DeclareLaunchArgument('forward_speed', default_value='0.15',
                              description='linear speed toward the target [m/s]'),
        DeclareLaunchArgument('search', default_value='false',
                              description='rotate to search when nothing is seen'),
        DeclareLaunchArgument('action', default_value='true',
                              description='autonomous steering: publish /cmd_vel from '
                                          'detections. Set false to drive by keyboard only.'),
        DeclareLaunchArgument('motor_backend', default_value='arduino',
                              description='motor driver: arduino (real serial) | fake (print)'),
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0',
                              description='Arduino serial port'),
        DeclareLaunchArgument('baud', default_value='115200',
                              description='Arduino serial baud rate'),
        DeclareLaunchArgument('viz', default_value='false',
                              description='open the OpenCV overlay window (needs a display)'),
        DeclareLaunchArgument('odom', default_value='true',
                              description='dead-reckon pose from /cmd_vel onto /odom'),
    ]

    camera = Node(
        package='rover_control', executable='camera_node', name='camera_node',
        output='screen',
        parameters=[{'source': source, 'video_path': video_path}],
    )

    perception = Node(
        package='rover_control', executable='perception_node', name='perception_node',
        output='screen',
        parameters=[{'model': model, 'conf': conf, 'target_class': target_class}],
    )

    action = Node(
        package='rover_control', executable='action_node', name='action_node',
        output='screen',
        parameters=[{'target_class': target_class,
                     'forward_speed': forward_speed,
                     'search': search}],
        condition=IfCondition(action_enable),
    )

    arduino_motor = Node(
        package='rover_control', executable='arduino_bridge_node', name='arduino_bridge_node',
        output='screen',
        parameters=[{'port': port, 'baud': baud}],
        condition=LaunchConfigurationEquals('motor_backend', 'arduino'),
    )

    fake_motor = Node(
        package='rover_control', executable='fake_motor_node', name='fake_motor_node',
        output='screen',
        condition=LaunchConfigurationEquals('motor_backend', 'fake'),
    )

    odometry = Node(
        package='rover_control', executable='odometry_node', name='odometry_node',
        output='screen',
        condition=IfCondition(odom),
    )

    viz_node = Node(
        package='rover_control', executable='viz_node', name='viz_node',
        output='screen',
        condition=IfCondition(viz),
    )

    return LaunchDescription(
        args + [camera, perception, action, arduino_motor, fake_motor, odometry, viz_node])
