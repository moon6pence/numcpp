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
	numcpp/array.h
	numcpp/array_allocate.h
	numcpp/array_function.h
	numcpp/tmp.h
)

# Executable: magick
include_directories("/usr/local/include/GraphicsMagick")

add_executable(
	magick
	test_magick.cpp
	numcpp/image.h
)

target_link_libraries(
	magick
	GraphicsMagick++ GraphicsMagick gomp pthread
)
