cmake_minimum_required(VERSION 2.8)
project(main)

add_definitions(-std=c++11)
#set(QMAKE_CXXFLAGS "-std=c++11")
include_directories(/tensorRT/include)
link_directories(/data/ai_solib_group/tensorRT/lib)
link_directories(/usr/local/cftlib/ai_solib_group/cuda)
include_directories(/opencv/include)
link_directories(/opencv/lib)


AUX_SOURCE_DIRECTORY(./ DIR_SRCS)
add_executable( main ${DIR_SRCS} main.cpp)

set(CMAKE_CXX_FLAGS  "-lpthread")
set(CMAKE_CXX_FLAGS  "-O3")
target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( main openblas)
target_link_libraries( main cudart)
target_link_libraries( main nvinfer)
target_link_libraries( main nvparsers)
target_link_libraries( main nvonnxparser)
target_link_libraries( main opencv_world)

