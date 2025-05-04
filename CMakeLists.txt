cmake_minimum_required(VERSION 3.28)
project(handwritten_check_reader)

find_package(OPENCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 23)

add_executable(geometric_properties main.cpp
        main.cpp)
target_link_libraries(geometric_properties ${OpenCV_LIBS})
