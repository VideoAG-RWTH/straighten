cmake_minimum_required(VERSION 2.8)

enable_language(CXX)

SET(CMAKE_CXX_FLAGS "-std=c++0x -O2 -funroll-loops")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

project( straighten )
find_package( OpenCV REQUIRED )

add_executable( straighten main.cpp )
target_link_libraries( straighten ${OpenCV_LIBS} )
