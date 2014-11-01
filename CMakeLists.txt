cmake_minimum_required(VERSION 2.8.7)
cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0043 NEW)

project(numcpp)

# Conditional build with external dependencies
set(USE_OPENCV ON CACHE BOOL "Use OpenCV")
set(USE_CUDA ON CACHE BOOL "Use CUDA")
set(BUILD_TEST ON CACHE BOOL "Build unit tests")
set(BUILD_EXAMPLE OFF CACHE BOOL "Build examples")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}")

# Compiler option for gcc
if (${CMAKE_COMPILER_IS_GNUCXX})
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11" )
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
endif (${CMAKE_COMPILER_IS_GNUCXX})

# Compiler options for VC++
if (${MSVC})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS /wd4819")
endif (${MSVC})

# numcpp library
add_library(
	numcpp
	numcpp/numcpp.h
	numcpp.cpp

	numcpp/allocator.h
	numcpp/base_array.h
	numcpp/array.h
	numcpp/tarray.h
	numcpp/array_functions.h
	numcpp/stl.h
	numcpp/functions.h
	numcpp/lazy_array.h

	# TODO: conditional include?
	numcpp/opencv.h
	numcpp/cuda.h
	numcpp/cufft.h
)

# External library: OpenCV
if (USE_OPENCV)
	find_package(OpenCV REQUIRED)
	target_link_libraries(numcpp ${OpenCV_LIBS})
endif (USE_OPENCV)

# External library: NVIDIA CUDA
if (USE_CUDA)
	find_package(CUDA REQUIRED)
	include_directories(${CUDA_INCLUDE_DIRS})
	target_link_libraries(numcpp ${CUDA_LIBRARIES})

	# Generate 2.0 code
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20;)

	if (${MSVC})
		# VS2012 + CUDA 5.0
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use-local-env --cl-version 2010")
	endif (${MSVC})

	if (${UNIX})
		# Use C++03 compiler, not C++11
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -std=c++03)
	endif (${UNIX})
endif (USE_CUDA)

if (BUILD_TEST)
	# Unit tests with Google Test
	if (${CMAKE_COMPILER_IS_GNUCXX})
		link_directories("thirdparty/gtest-1.7.0/lib/linux_x86_64")
	elseif (${MSVC})
		if (${CMAKE_SIZEOF_VOID_P} MATCHES "8")
			# Link 64 bit library
			link_directories("thirdparty/gtest-1.7.0/lib/msvc2012_64")
		else ()
			# Link 32 bit library
			link_directories("thirdparty/gtest-1.7.0/lib/msvc2012")
		endif ()

		add_definitions(-D_VARIADIC_MAX=10)
	endif()

	if (USE_CUDA)
		cuda_compile(TEST_CUDA_KERNEL_O numcpp/test_cuda_kernel.cu)
	endif (USE_CUDA)

	# test executable
	add_executable(
		numcpp.test 
		test.cpp

		numcpp/base_array.test.inl
		numcpp/array.test.inl
		numcpp/tarray.test.inl
		numcpp/stl.test.inl
		numcpp/functions.test.inl
		numcpp/lazy_array.test.inl
		
		numcpp/opencv.test.inl
		numcpp/cuda.test.inl
		${TEST_CUDA_KERNEL_O}

		numcpp/test_arrayfixture.h
	)

	find_package(Threads)
	include_directories("thirdparty/gtest-1.7.0/include")
	target_link_libraries(numcpp.test numcpp gtest ${CMAKE_THREAD_LIBS_INIT})

	# Copy files for test
	file(COPY images/Lena.bmp DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
	file(COPY images/fromfile_int.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
endif (BUILD_TEST)

if (BUILD_EXAMPLE)
	# Example: circularize
	if (USE_OPENCV)
		add_executable(circularize example/circularize.h example/circularize.cpp)
		target_link_libraries(circularize numcpp)
	endif (USE_OPENCV)
endif (BUILD_EXAMPLE)
