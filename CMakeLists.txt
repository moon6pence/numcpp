cmake_minimum_required(VERSION 2.8.8)
project(numcpp2)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}")

# Compiler option for C++11
if (${CMAKE_COMPILER_IS_GNUCXX})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
endif (${CMAKE_COMPILER_IS_GNUCXX})

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
if (${UNIX})
	include_directories("/usr/local/include/GraphicsMagick")
endif (${UNIX})

add_executable(
	magick
	test_magick.cpp
	numcpp/image.h
)

if (${UNIX})
	target_link_libraries(
		magick
		GraphicsMagick++ GraphicsMagick gomp pthread
	)
endif (${UNIX})
