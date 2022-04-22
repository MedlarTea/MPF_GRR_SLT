cmake_minimum_required(VERSION 3.0.2)
project(mono_tracking)


## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

# Debug with GDB
# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  eigen_conversions
  tf_conversions
  image_transport
  roscpp
  rospy
  std_msgs
  geometry_msgs
  tf
  message_generation
  cv_bridge
  # pcl_ros
  # pcl_conversions
  # velodyne_pcl
  # velodyne_pointcloud
  message_filters
)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(PCL 1.8 REQUIRED)

add_message_files(
  FILES
  Box.msg
  BoxArray.msg
  Track.msg
  TrackArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mono_tracking
#  CATKIN_DEPENDS eigen_conversions image_transport roscpp rospy std_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

# add_executable(mono_tracking_node
#   src/track_system.cpp
#   src/person_tracker.cpp
#   src/people_tracker.cpp
#   src/mono_tracking_node.cpp
# )
add_executable(mono_tracking_node
  src/track_system.cpp
  src/person_tracker.cpp
  src/people_tracker.cpp
  src/mono_tracking_node.cpp
)
target_link_libraries(mono_tracking_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
add_dependencies(mono_tracking_node mono_tracking_gencpp)


# add_executable(evaluate_distance_by_mc src/debug/evaluate_distance_by_mc.cpp)
# target_link_libraries(evaluate_distance_by_mc
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBS}
# )
# add_dependencies(evaluate_distance_by_mc mono_tracking_gencpp)