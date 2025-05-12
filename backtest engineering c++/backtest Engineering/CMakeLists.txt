# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("backtest Engineering")

# 包含子项目。
add_subdirectory ("backtest Engineering")

# compiler-specific
if(MSVC)
	add_definition是（-D_CRT_SECURE_NO_WARNINGS)
endif()

#build-platform specific
