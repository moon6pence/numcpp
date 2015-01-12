cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0020 NEW)
if (POLICY CMP0043)
	cmake_policy(SET CMP0043 NEW)
endif (POLICY CMP0043)

project(nbb)

# Conditional build with external dependencies
set(USE_OPENCV ON CACHE BOOL "Use OpenCV")
set(USE_CUDA ON CACHE BOOL "Use CUDA")
set(BUILD_TEST ON CACHE BOOL "Build unit tests")
set(BUILD_EXAMPLE OFF CACHE BOOL "Build examples")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

# Link library directory
set(CMAKE_LIBRARY_PATH "$ENV{ProgramFiles}/Windows Kits/8.0/Lib/win8/um/x86")

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
add_library(numcpp
	numcpp.cpp

	numcpp/allocator.h
	numcpp/base_array.h
	numcpp/array.h
	numcpp/array_functions.h
	numcpp/gpu_array.h
	numcpp/opencv.h

	# numcpp/stl.h
	# numcpp/functions.h
	# numcpp/lazy_array.h

	# TODO: conditional include?
	# numcpp/cuda.h
	# numcpp/cufft.h
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
			# link_directories("thirdparty/gtest-1.7.0/lib/msvc2012_64")
			link_directories("thirdparty/gtest-1.7.0/lib/msvc2013_64")
		else ()
			# Link 32 bit library
			# link_directories("thirdparty/gtest-1.7.0/lib/msvc2012")
			link_directories("thirdparty/gtest-1.7.0/lib/msvc2013")
		endif ()

		add_definitions(-D_VARIADIC_MAX=10)
	endif()

	if (USE_CUDA)
		cuda_compile(TEST_CUDA_KERNEL_O numcpp/test_cuda_kernel.cu)
	endif (USE_CUDA)

	# test executable
	add_executable(numcpp.test 
		test.cpp

		numcpp/base_array.test.inl
		numcpp/array.test.inl
		numcpp/gpu_array.test.inl
		${TEST_CUDA_KERNEL_O}
		numcpp/opencv.test.inl
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

# ----------

# Qt
find_package(Qt5Widgets REQUIRED)

qt5_wrap_ui(UI_FILES nbb/ContextWindow.ui)
qt5_add_resources(QRC_FILES nbb/ContextWindow.qrc)

# JsonCPP
include_directories(thirdparty/jsoncpp/include)
get_filename_component(jsoncpp_LIB "thirdparty/jsoncpp/lib/jsoncpp.lib" REALPATH) # to absolute path

# Library: nbb
add_library(nbb
	nbb/signal.h
	nbb/property.h
	nbb/Object.h
	nbb/Object.cpp
	nbb/Context.h
	nbb/Context.cpp
	nbb/QuickDialog.h
	nbb/QuickDialog.cpp
	nbb/QuickJSON.h
	nbb/QuickJSON.cpp

	nbb/ContextWindow.h
	nbb/ContextWindow.cpp
	${UI_FILES}
	${QRC_FILES}
)

target_link_libraries(nbb ${jsoncpp_LIB})
qt5_use_modules(nbb Core Widgets)

# Auto moc
set_target_properties(nbb PROPERTIES AUTOMOC TRUE)