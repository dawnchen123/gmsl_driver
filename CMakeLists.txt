cmake_minimum_required(VERSION 2.8.3)
project(gmsl_camera)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# [Critical Fix] Force finding the custom OpenCV with CUDA
# Assuming you installed to /usr/local. 
# If you installed elsewhere, change this path.
set(OpenCV_DIR "/usr/local/share/OpenCV") 

find_package(OpenCV 3 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Verify CUDA is enabled in the found OpenCV
if(NOT OpenCV_CUDA_VERSION)
    message(WARNING "The found OpenCV does NOT have CUDA modules! Check OpenCV_DIR.")
else()
    message(STATUS "OpenCV CUDA version: ${OpenCV_CUDA_VERSION}")
endif()

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# ... [Standard Settings] ...

find_package(CUDA REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS  ${CUDA_NVCC_FLAGS} --maxrregcount=32; -G;-g;-std=c++11) 

# Jetson Multimedia API paths
set(CUDIR /usr/src/jetson_multimedia_api/samples/common/algorithm/cuda)
set(COMMONCLS /usr/src/jetson_multimedia_api/samples/common/classes)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_compile_options(-std=c++11 -g)
add_definitions(-DCAM_IMX390)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} )

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  image_transport 
  cv_bridge
  camera_info_manager
  nodelet
)

find_package(YAML-CPP REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gmsl_camera
  CATKIN_DEPENDS image_transport roscpp cv_bridge sensor_msgs camera_info_manager
  DEPENDS system_lib
)

set(COMMONCLSFILES
    ${COMMONCLS}/NvBuffer.cpp
    ${COMMONCLS}/NvElement.cpp
    ${COMMONCLS}/NvElementProfiler.cpp
    ${COMMONCLS}/NvLogging.cpp)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    include
    include/nvencoder 
    /usr/src/jetson_multimedia_api/include
    /usr/src/jetson_multimedia_api/samples/common/algorithm/cuda/
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
)

#-------------------------------------------------------------------------------
# Build Node 
#-------------------------------------------------------------------------------

add_executable(${PROJECT_NAME}_node src/master.cpp src/stitcherglobal.cpp  ${COMMONCLSFILES})

# [Critical Fix] Link OpenCV libs *before* catkin libraries (which contain cv_bridge)
# to ensure our symbols take precedence.
target_link_libraries(${PROJECT_NAME}_node
    ${OpenCV_LIBS} 
    yaml-cpp
    -L/usr/lib/aarch64-linux-gnu/tegra/ -lnvbuf_utils  
    -L/usr/lib/aarch64-linux-gnu/ -lv4l2 -lEGL -lGLESv2 -lX11 
    ${CUDA_LIBRARIES} ${GLUT_LIBRARY} 
    ${catkin_LIBRARIES}
    -L${CMAKE_CURRENT_SOURCE_DIR}/cfg/ -ltkDNN
)