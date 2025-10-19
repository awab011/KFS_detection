from setuptools import find_packages, setup

package_name = 'KFS_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (package_name + '/weights', ['KFS_detection/weights/Initial_detection.pt',
                                     'KFS_detection/weights/Authenticity.pt']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parallels',
    maintainer_email='awab011@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = KFS_detection.yolo_node:main'
        ],
    },
)
