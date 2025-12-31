cmake_minimum_required(VERSION 3.8)
project(lidar_vo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(PCL REQUIRED)

include_directories(/usr/include/g2o)
set(G2O_LIBS
    g2o_core
    g2o_stuff
    g2o_types_slam3d
    g2o_types_slam2d
    g2o_solver_csparse
    cxsparse
)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${g2o_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_executable(icp_node
    src/icp_node.cpp
)

target_link_libraries(icp_node
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Sophus::Sophus
    ${G2O_LIBS}
    ${PCL_LIBRARIES}
)

ament_target_dependencies(icp_node
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    tf2
    tf2_ros
    visualization_msgs
)

install(TARGETS icp_node DESTINATION lib/${PROJECT_NAME})


add_executable(slam_node
    src/slam_node.cpp
)

target_link_libraries(slam_node
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Sophus::Sophus
    ${G2O_LIBS}
    ${PCL_LIBRARIES}
)

ament_target_dependencies(slam_node
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    tf2
    tf2_ros
    visualization_msgs
)

install(TARGETS slam_node DESTINATION lib/${PROJECT_NAME})



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
