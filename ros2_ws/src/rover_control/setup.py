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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leonardo Ragogna',
    maintainer_email='alexiragogna@gmail.com',
    description='Modular keyboard teleop and fake motor driver for the autonomous rover.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'keyboard_node = rover_control.keyboard_node:main',
            'fake_motor_node = rover_control.fake_motor_node:main',
        ],
    },
)
