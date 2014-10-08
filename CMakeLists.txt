cmake_minimum_required(VERSION 2.8.9)
cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0043 NEW)

project(QuickDialog)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

# Link library directory
set(CMAKE_LIBRARY_PATH "$ENV{ProgramFiles}/Windows Kits/8.0/Lib/win8/um/x86")

# JsonCPP
find_package(jsoncpp)
include_directories(${jsoncpp_INCLUDE_DIRS})

# Qt
find_package(Qt5Widgets REQUIRED)

qt5_wrap_ui(UI_FILES ContextWindow.ui)
qt5_add_resources(QRC_FILES ContextWindow.qrc)

# Library: QuickDialog
add_library(QuickDialog
	property.h
	Object.h
	Object.cpp
	Context.h
	Context.cpp
	QuickDialog.h
	QuickDialog.cpp
	QuickJSON.h
	QuickJSON.cpp

	ContextWindow.h
	ContextWindow.cpp
	${UI_FILES}
	${QRC_FILES}
)

target_link_libraries(QuickDialog ${jsoncpp_LIBRARIES})
qt5_use_modules(QuickDialog Core Widgets)

# Auto moc
set_target_properties(QuickDialog PROPERTIES AUTOMOC TRUE)