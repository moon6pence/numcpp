cmake_minimum_required(VERSION 2.8.8)
project(numcpp2)

# Conditional build with external dependencies
set(USE_MAGICK ON CACHE BOOL "Use GraphicsMagick library")
set(USE_IPP ON CACHE BOOL "Use Intel IPP library")
set(USE_CUDA OFF CACHE BOOL "Use NVIDIA CUDA")
set(VARIADIC_TEMPLATE ON CACHE BOOL "Supports C++11 variadic template")
set(INITIALIZER_LIST ON CACHE BOOL "Supports C++11 initializer_list")
configure_file(config.h.in config.h)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

# Compiler option for gcc: C++11
if (${CMAKE_COMPILER_IS_GNUCXX})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
endif (${CMAKE_COMPILER_IS_GNUCXX})

# Compiler options for VC++
if (${MSVC})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS" )
endif (${MSVC})

# Library: numcpp
add_library(
	numcpp
	numcpp.cpp
	numcpp.h
	numcpp/array.h
	numcpp/array_allocate.h
	numcpp/array_function.h
	numcpp/tmp.h

	# TODO: conditional add?
	numcpp/image.h
	numcpp/device_array.h
)

# External library: GraphicsMagick
if (USE_MAGICK)
	if (${UNIX})
		include_directories("/usr/local/include/GraphicsMagick")
		target_link_libraries(numcpp GraphicsMagick++ GraphicsMagick gomp pthread)
	elseif (${WIN32})
		include_directories("$ENV{USERPROFILE}/GraphicsMagick-1.3.18")
		include_directories("$ENV{USERPROFILE}/GraphicsMagick-1.3.18/Magick++/lib")
		link_directories("$ENV{USERPROFILE}/GraphicsMagick-1.3.18/VisualMagick/lib")
		target_link_libraries(numcpp CORE_RL_magick_ CORE_RL_Magick++_)
	endif (${UNIX})
endif (USE_MAGICK)

# External library: Intel IPP
if (USE_IPP)
	if (${UNIX})
		include_directories("/opt/intel/ipp/include")
		link_directories("/opt/intel/ipp/lib/intel64")
		target_link_libraries(numcpp ippi)
	elseif (${WIN32})
		include_directories("$ENV{ProgramFiles}/Intel/Composer XE/ipp/include")
		link_directories("$ENV{ProgramFiles}/Intel/Composer XE/ipp/lib/ia32")
		target_link_libraries(numcpp ippi)
	endif (${UNIX})
endif (USE_IPP)

# External library: NVIDIA CUDA
if (USE_CUDA)
	find_package(CUDA)

	# Generate 2.0 code
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20;)

	if (${UNIX})
		# Use C++03 compiler, not C++11
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -std=c++03)
	endif (${UNIX})
endif (USE_CUDA)

# Executable: test
add_executable(test test_main.cpp)
target_link_libraries(test numcpp)

# Testcase: magick
if (USE_MAGICK)
	add_executable(magick test_magick.cpp	)
	target_link_libraries(magick numcpp)
endif (USE_MAGICK)

# Testcase: ipp
if (USE_IPP)
	add_executable(ipp test_ipp.cpp)
	target_link_libraries(ipp numcpp)
endif (USE_IPP)

# Testcase: cuda
if (USE_CUDA)
	cuda_compile(TEST_CUDA_KERNEL_O test_cuda_kernel.cu)
	cuda_add_library(cuda_kernel ${TEST_CUDA_KERNEL_O})

	add_executable(cuda test_cuda.cpp)
	target_link_libraries(cuda cuda_kernel numcpp)	
endif (USE_CUDA)

# Example: circularize
if (USE_MAGICK AND USE_IPP)
	add_executable(circularize example/circularize.cpp)
	target_link_libraries(circularize numcpp)
endif(USE_MAGICK AND USE_IPP)