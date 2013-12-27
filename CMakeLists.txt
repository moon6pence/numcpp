cmake_minimum_required(VERSION 2.8.11)
project(numcpp)

# Conditional build with external dependencies
set(USE_OPENCV ON CACHE BOOL "Use OpenCV")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)

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
	numcpp/stl.h
	numcpp/functions.h

	# TODO: conditional include?
	numcpp/opencv.h
)

# External library: OpenCV
if (USE_OPENCV)
	find_package(OpenCV REQUIRED)
	target_link_libraries(numcpp ${OpenCV_LIBS})

	file(COPY images/Lena.bmp DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
endif (USE_OPENCV)

# Unit tests with Google Test
link_directories("thirdparty/gtest-1.7.0/lib/linux_x86_64")

# test executable
add_executable(
	test 
	test_main.cpp

	numcpp/array.test.inl
	numcpp/stl.test.inl
	numcpp/functions.test.inl
	numcpp/opencv.test.inl
)

target_include_directories(test PUBLIC "thirdparty/gtest-1.7.0/include")
target_link_libraries(test numcpp gtest pthread)

# Example: circularize
if (USE_OPENCV)
	add_executable(circularize example/circularize.h example/circularize.cpp)
	target_link_libraries(circularize numcpp)
endif (USE_OPENCV)
