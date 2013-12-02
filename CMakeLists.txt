cmake_minimum_required(VERSION 2.8.8)
project(numcpp2)

# Conditional build with external dependencies
set(BUILD_TESTS ON CACHE BOOL "Build tests")
set(BUILD_EXAMPLES ON CACHE BOOL "Build examples")
set(USE_OPENCV ON CACHE BOOL "Use OpenCV")
set(USE_IPP ON CACHE BOOL "Use Intel IPP library")
set(USE_CUDA OFF CACHE BOOL "Use NVIDIA CUDA")
set(VARIADIC_TEMPLATE ON CACHE BOOL "Supports C++11 variadic template")
set(INITIALIZER_LIST ON CACHE BOOL "Supports C++11 initializer_list")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

# Compiler option for gcc: C++11
if (${CMAKE_COMPILER_IS_GNUCXX})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
endif (${CMAKE_COMPILER_IS_GNUCXX})

# Compiler options for VC++
if (${MSVC})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS /wd4819")
endif (${MSVC})

# Library: numcpp
add_library(
	numcpp
	numcpp.cpp
	numcpp.h
	numcpp/array.h
	numcpp/array_function.h
	numcpp/tmp.h

	# TODO: conditional add?
	numcpp/opencv.h
	numcpp/ipp.h
	numcpp/device_array.h
)

# External library: OpenCV
if (USE_OPENCV)
	find_package(OpenCV REQUIRED)
	target_link_libraries(numcpp ${OpenCV_LIBS})
endif (USE_OPENCV)

# External library: Intel IPP
if (USE_IPP)
	if (${UNIX})
		include_directories("/opt/intel/ipp/include")
		link_directories("/opt/intel/ipp/lib/intel64")
		target_link_libraries(numcpp ippcore ipps ippi)
	elseif (${WIN32})
		include_directories("$ENV{ProgramFiles}/Intel/Composer XE/ipp/include")
		link_directories("$ENV{ProgramFiles}/Intel/Composer XE/ipp/lib/ia32")
		target_link_libraries(numcpp ippcore ipps ippi)
	endif (${UNIX})
endif (USE_IPP)

# External library: NVIDIA CUDA
if (USE_CUDA)
	find_package(CUDA REQUIRED)

	# Generate 2.0 code
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20;)

	if (${UNIX})
		# Use C++03 compiler, not C++11
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -std=c++03)
	endif (${UNIX})
endif (USE_CUDA)

## Tests ##
if (BUILD_TESTS)
	# Executable: test
	add_executable(test test_main.cpp)
	target_link_libraries(test numcpp)

	# Testcase: OpenCV
	if (USE_OPENCV)
		add_executable(opencv test_opencv.cpp)
		target_link_libraries(opencv numcpp)
	endif (USE_OPENCV)

	# Testcase: ipp
	if (USE_IPP)
		add_executable(ipp test_ipp.cpp)
		target_link_libraries(ipp numcpp)
	endif (USE_IPP)

	# Testcase: cuda
	if (USE_CUDA)
		cuda_compile(TEST_CUDA_KERNEL_O test_cuda_kernel.cu)
		cuda_add_executable(cuda test_cuda.cpp ${TEST_CUDA_KERNEL_O})
		target_link_libraries(cuda numcpp)	
	endif (USE_CUDA)
endif (BUILD_TESTS)

## Examples ##
if (BUILD_EXAMPLES)
	# Example: circularize
	if (USE_OPENCV)
		add_executable(circularize example/circularize.h example/circularize.cpp)
		target_link_libraries(circularize numcpp)
	endif (USE_OPENCV)
endif (BUILD_EXAMPLES)