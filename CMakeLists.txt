cmake_minimum_required(VERSION 3.22)
project(VariationalDisparity)

set(CMAKE_CXX_STANDARD 17)

message(STATUS ${CMAKE_BUILD_TYPE})
if(NOT ${CMAKE_BUILD_TYPE})
    message(STATUS "Specified build type: ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type not specified. Set to release.")
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(BUILD_GMOCK 1)
add_subdirectory(thirdparty/googletest
        ${PROJECT_BINARY_DIR}/gtest)
include(CTest)

find_package(OpenCV REQUIRED)
MESSAGE(STATUS ${OpenCV_INCLUDE_DIRS})

find_package(TBB REQUIRED)

find_package(OpenMP REQUIRED)

add_definitions(-DWITH_OPENCV)
find_package(Python3 COMPONENTS Development NumPy)
include_directories(${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)
message(STATUS "Install path: ${CMAKE_INSTALL_PREFIX}")
install(TARGETS main DESTINATION bin/debug CONFIGURATIONS Debug)
install(TARGETS main DESTINATION bin/release CONFIGURATIONS Release)

file(GLOB files "data/*.png")
foreach(file ${files})
    get_filename_component(name ${file} NAME)
    MESSAGE(STATUS "Data file to be installed: ${name}")
    install(FILES ${file} DESTINATION data)
endforeach(file ${files})
