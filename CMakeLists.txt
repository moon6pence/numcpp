cmake_minimum_required(VERSION 2.8.8)
project(numcpp2)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}")

# CXX compiler option
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

# Executable: test
add_executable(
	test
	test_main.cpp
	numcpp2/array.h
)
