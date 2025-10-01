from setuptools import find_packages, setup

package_name = 'nunbot_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='labtech',
    maintainer_email='akhilmjohn@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_nav_node = nunbot_navigation.simple_nav_node:main',
            'simple_waypoint_navigator = nunbot_navigation.simple_waypoint_navigator:main',
            'simplified_navigator = nunbot_navigation.simplified_navigator:main',
            'test_navigation = nunbot_navigation.test_navigation:main',
        ],
    },
)
