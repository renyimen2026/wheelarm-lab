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

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(initial_pose test/test_initial_pose.py)
  ament_add_pytest_test(initial_pose_ros test/test_initial_pose_ros.py)
endif()

ament_package()
