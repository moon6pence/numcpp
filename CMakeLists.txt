cmake_minimum_required(VERSION 2.8.8)
project(numcpp2)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}" "/usr/local/include/GraphicsMagick")

# CXX compiler option
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )

# Executable: test
add_executable(
	test
	test_main.cpp
	numcpp/array.h
)

# Executable: magick
add_executable(
	magick
	test_magick.cpp
)

find_package(ImageMagick COMPONENTS Magick++)

target_link_libraries(
	magick
	GraphicsMagick++ GraphicsMagick gomp pthread
)

