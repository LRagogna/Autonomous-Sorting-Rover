"""Bring up the full autonomous perception pipeline.

    camera_node -> perception_node -> action_node -> fake_motor_node

Launch it with:

    ros2 launch rover_control perception_pipeline.launch.py

Common overrides (all optional):

    # Develop on a laptop with a USB webcam instead of the Pi camera:
    ros2 launch rover_control perception_pipeline.launch.py source:=usb

    # Or replay a recorded clip, fully offline:
    ros2 launch rover_control perception_pipeline.launch.py \
        source:=video video_path:=/path/to/clip.mp4

    # Chase only one class, and actually drive the (simulated) motors:
    ros2 launch rover_control perception_pipeline.launch.py target_class:=bit

    # Swap the simulated motors for the real Arduino bridge later:
    ros2 launch rover_control perception_pipeline.launch.py motors:=false
    # (then run your arduino bridge node separately on /cmd_vel)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
    motors = LaunchConfiguration('motors')
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
        DeclareLaunchArgument('motors', default_value='true',
                              description='also start fake_motor_node on /cmd_vel'),
        DeclareLaunchArgument('viz', default_value='true',
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

    fake_motor = Node(
        package='rover_control', executable='fake_motor_node', name='fake_motor_node',
        output='screen',
        condition=IfCondition(motors),
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
        args + [camera, perception, action, fake_motor, odometry, viz_node])
