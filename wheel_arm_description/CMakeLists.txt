cmake_minimum_required(VERSION 3.8)
project(wheel_arm_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
