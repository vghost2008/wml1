cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)
#SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_PREFIX_PATH /home/wj/ai/software/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ")
aux_source_directory(. SRC_LIST)
project(hrnetop)
include_directories(/usr/include /usr/include/eigen3)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Define our library target
add_library(hrnet_op SHARED ${SRC_LIST})
# Enable C++14
target_compile_features(hrnet_op PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(hrnet_op "${TORCH_LIBRARIES}")
# Link against OpenCV
target_link_libraries(hrnet_op opencv_core opencv_imgproc)
