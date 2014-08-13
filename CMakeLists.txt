cmake_minimum_required(VERSION 2.8.9)
cmake_policy(SET CMP0020 NEW)

project(QuickDialog)

# Project header directory
include_directories("${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

# Link library directory
set(CMAKE_LIBRARY_PATH "$ENV{ProgramFiles}/Windows Kits/8.0/Lib/win8/um/x86")

# JsonCPP
include_directories("jsoncpp/include")

# Qt
find_package(Qt5Widgets REQUIRED)

qt5_wrap_ui(UI_FILES MainWindow.ui)
qt5_add_resources(QRC_FILES MainWindow.qrc)

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

	MainWindow.h
	MainWindow.cpp
	${UI_FILES}
	${QRC_FILES}
)

qt5_use_modules(QuickDialog Core Widgets)

# Auto moc
set_target_properties(QuickDialog PROPERTIES AUTOMOC TRUE)