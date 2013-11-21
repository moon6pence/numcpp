cmake_minimum_required(VERSION 2.8.8)
project(numcpp2)

# Conditional build with external dependencies
SET(USE_MAGICK ON CACHE BOOL "Use GraphicsMagick library")
SET(USE_IPP ON CACHE BOOL "Use Intel IPP library")

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}")

# Compiler option for gcc: C++11
if (${CMAKE_COMPILER_IS_GNUCXX})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
endif (${CMAKE_COMPILER_IS_GNUCXX})

# Compiler options for VC++
if (${MSVC})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS" )
endif (${MSVC})

# Executable: test
add_executable(
	test
	test_main.cpp
	numcpp.h
	numcpp/array.h
	numcpp/array_allocate.h
	numcpp/array_function.h
	numcpp/tmp.h
)

# External library: GraphicsMagick
if (USE_MAGICK)

	# Dependency : GraphicsMagick
	if (${UNIX})
		include_directories("/usr/local/include/GraphicsMagick")
	elseif (${WIN32})
		include_directories("$ENV{USERPROFILE}/GraphicsMagick-1.3.18")
		include_directories("$ENV{USERPROFILE}/GraphicsMagick-1.3.18/Magick++/lib")
		link_directories("$ENV{USERPROFILE}/GraphicsMagick-1.3.18/VisualMagick/lib")
	endif (${UNIX})

	# Testcase: magick
	add_executable(
		magick
		test_magick.cpp
	)

	if (${UNIX})
		target_link_libraries(
			magick
			GraphicsMagick++ GraphicsMagick gomp pthread
		)
	elseif (${WIN32})
		target_link_libraries(
			magick
			CORE_RL_magick_
			CORE_RL_Magick++_
		)
	endif (${UNIX})

endif(USE_MAGICK)

# External library: Intel IPP
if (USE_IPP)

	# Dependency: Intel IPP
	if (${UNIX})
		include_directories("/opt/intel/ipp/include")
		link_directories("/opt/intel/ipp/lib/intel64")
	elseif (${WIN32})
		include_directories("$ENV{ProgramFiles}/Intel/Composer XE/ipp/include")
		link_directories("$ENV{ProgramFiles}/Intel/Composer XE/ipp/lib/ia32")
	endif (${UNIX})

	# Testcase: ipp
	add_executable(ipp test_ipp.cpp)
	target_link_libraries(ipp ippi)

endif(USE_IPP)

# Example: circularize
if (USE_MAGICK AND USE_IPP)

	# Executable: circularize
	add_executable(
		circularize
		example/circularize.cpp
	)

	if (${UNIX})
		target_link_libraries(
			circularize
			GraphicsMagick++ GraphicsMagick gomp pthread
			ippi
		)
	elseif (${WIN32})
		target_link_libraries(
			circularize
			CORE_RL_magick_
			CORE_RL_Magick++_
			ippi
		)
	endif (${UNIX})

endif(USE_MAGICK AND USE_IPP)