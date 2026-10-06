from setuptools import setup

package_name = "wheel_arm_navigation"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/patrol.example.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="renyimen",
    maintainer_email="renyimen2026@foxmail.com",
    description="Named map goals for wheel arm robot navigation.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "semantic_goal = wheel_arm_navigation.semantic_goal:main",
        "mission_runner = wheel_arm_navigation.mission_runner:main",
    ]},
)
