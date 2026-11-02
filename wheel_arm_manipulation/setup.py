from setuptools import find_packages, setup
from glob import glob

setup(
    name="wheel_arm_manipulation",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/wheel_arm_manipulation"]),
        ("share/wheel_arm_manipulation", ["package.xml"]),
        ("share/wheel_arm_manipulation/launch", glob("launch/*.launch.py")),
        ("share/wheel_arm_manipulation/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="renyimen",
    maintainer_email="renyimen2026@foxmail.com",
    description="Position-only MoveIt arm commands with preview and confirmation.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": ["arm_goal = wheel_arm_manipulation.arm_goal:main",
                                    "cup_grasp = wheel_arm_manipulation.cup_grasp:main",
                                    "camera_check = wheel_arm_manipulation.camera_check:main",
                                    "living_room_fixture = wheel_arm_manipulation.living_room_fixture:main",
                                    "grasp_fixture = wheel_arm_manipulation.grasp_fixture:main"]},
)
