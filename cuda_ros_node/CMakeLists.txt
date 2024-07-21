cmake_minimum_required(VERSION 3.16)
project(cuda_ros_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-std=c++17)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
)

#--------------------------------------------------------------
#                     CudaRosNode Node
#--------------------------------------------------------------

# List sources
set(CudaRosNode_SRC
  src/CudaRosNode_impl.cpp
)

# List headers
set(CudaRosNode_HEADERS
  include/CudaRosNode_impl.hpp
)

# Create a shared (.so) library
ament_auto_add_library(CudaRosNode_lib
    ${CudaRosNode_SRC}
    ${CudaRosNode_HEADERS}
)

ament_target_dependencies(CudaRosNode_lib
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  CUDA
)
# CUDA KERNELS MUST BE IN A STATIC LIBRARY
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#libraries
set(LIB_TYPE STATIC)
cuda_add_library(CudaLib ${LIB_TYPE}
                  include/CPUBitmap.h
                  src/Add.cu
                  src/Dot.cu
                  src/Ray.cu
)

target_include_directories(CudaLib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
)

cuda_add_executable(CudaRosNode_exe
  src/CudaRosNode_main.cpp
)

target_link_libraries(CudaRosNode_exe
  CudaRosNode_lib  
  CudaLib
)

set_property(TARGET CudaRosNode_exe
  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(CudaRosNode_exe PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
)

include_directories(
  ${CUDA_INCLUDE_DIRS}
)

install(TARGETS CudaRosNode_exe
  DESTINATION lib/${PROJECT_NAME}
)

# Set strict compile options for the library
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(CudaRosNode_exe PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

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
