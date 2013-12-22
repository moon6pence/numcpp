cmake_minimum_required(VERSION 2.8.11)
project(numcpp)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}")

# Compiler option for gcc: C++11
if (${CMAKE_COMPILER_IS_GNUCXX})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
endif (${CMAKE_COMPILER_IS_GNUCXX})

# numcpp library
add_library(
	numcpp
	numcpp.cpp

	numcpp/array.h
	numcpp/array_function.h
)

# Unit tests with Google Test
link_directories("thirdparty/gtest-1.7.0/lib/linux_x86_64")

add_executable(test test_main.cpp)
target_include_directories(test PUBLIC "thirdparty/gtest-1.7.0/include")
target_link_libraries(test numcpp gtest pthread)
