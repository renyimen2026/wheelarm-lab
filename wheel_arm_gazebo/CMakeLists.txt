cmake_minimum_required(VERSION 3.8)
project(wheel_arm_gazebo)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch rviz worlds models
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/imu_covariance.py
    scripts/localization_debug.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
