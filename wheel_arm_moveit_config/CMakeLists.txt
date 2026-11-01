cmake_minimum_required(VERSION 3.8)
project(wheel_arm_moveit_config)

find_package(ament_cmake REQUIRED)

add_library(wheel_arm_moveit_plugin_lifetime SHARED src/plugin_lifetime.cpp)
target_compile_features(wheel_arm_moveit_plugin_lifetime PRIVATE cxx_std_17)
target_compile_options(wheel_arm_moveit_plugin_lifetime PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(wheel_arm_moveit_plugin_lifetime PRIVATE ${CMAKE_DL_LIBS})
install(TARGETS wheel_arm_moveit_plugin_lifetime LIBRARY DESTINATION lib)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(configuration test/test_configuration.py)
  ament_add_pytest_test(plugin_lifetime test/test_plugin_lifetime.py)
  ament_add_pytest_test(moveit_ros test/test_moveit_ros.py TIMEOUT 120)
endif()

ament_package()
