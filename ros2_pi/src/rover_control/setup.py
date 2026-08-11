from glob import glob

from setuptools import find_packages, setup

package_name = 'rover_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leonardo Ragogna',
    maintainer_email='alexiragogna@gmail.com',
    description='Raspberry Pi rover control: perception pipeline + real Arduino motor bridge.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'keyboard_node = rover_control.keyboard_node:main',
            'fake_motor_node = rover_control.fake_motor_node:main',
            'camera_node = rover_control.camera_node:main',
            'perception_node = rover_control.perception_node:main',
            'action_node = rover_control.action_node:main',
            'odometry_node = rover_control.odometry_node:main',
            'viz_node = rover_control.viz_node:main',
            'arduino_bridge_node = rover_control.arduino_bridge_node:main',
        ],
    },
)
