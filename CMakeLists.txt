cmake_minimum_required(VERSION 2.8.9)
cmake_policy(SET CMP0020 NEW)

project(QuickDialog)

# Link library directory
set (CMAKE_LIBRARY_PATH "$ENV{ProgramFiles}/Windows Kits/8.0/Lib/win8/um/x86")

# JsonCPP
include_directories("jsoncpp/include")

# Qt
find_package(Qt5Widgets REQUIRED)

add_library(QuickDialog
	property.h
	Object.h
	Context.h
	Context.cpp
	QuickDialog.h
	QuickDialog.cpp
	QuickJSON.h
	QuickJSON.cpp

	print.cpp
)

qt5_use_modules(QuickDialog Core Widgets)