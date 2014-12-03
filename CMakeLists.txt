cmake_minimum_required(VERSION 2.8.9)
cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0043 NEW)

# JsonCPP
include_directories(jsoncpp/include)
get_filename_component(jsoncpp_LIB "jsoncpp/lib/jsoncpp.lib" REALPATH) # to absolute path

# subdirectory: nbb
add_subdirectory(nbb)