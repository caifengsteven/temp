cmake_minimum_required(VERSION 2.8.11)

project(backtest)

message(STATUS "Build kind: " ${CMAKE_BUILD_TYPE})

#Get the current working branch
execute_process(
	COMMAND git rev-parse --abbrev-ref HEAD
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE GIT_BRANCH
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
	COMMAND git log -1 --format=%h
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE GIT_COMMIT_HASH
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
	COMMAND git diff-index --quiet HEAD --
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	RESULT_VARIABLE GIT_UNTRACKED_STATUS
)

add_definitons("-D__GIT_COMMIT_HASH__=${GIT_COMMIT_HASH}")
add_definitons("-D__GIT_BRANCH__=${GIT_BRANCH}")
add_definitons("-D__GIT_UNTRACKED_STATUS__=${GIT_UNTRACKED_STATUS}")
add_definitons("-D__MAX_DATA_HISTORY_DEPTH__=10)

#compiler - specific
if (MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

#build-platform specifi

if (WIN32)
	file(TO_CMAKE_PATH $ENV{USERPROFILE}, USERBASEDIR)
	message(STATUS, "User profile: " ${USERBASEDIR})
	find_path(HDFS_INCLUDE_DIR NAMES hdf5.h REQUIRED PATHS ${USERBASEDIR} "/3rdparty/include")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		find_library(HDF5_LIB NAMES libhdf5_D REQUIRED PATH ${USERBASEDIR} "/3rdparty/lib")
		find_library(HDF5_CPP_LIB NAMES libhdf5_cpp_D REQUIRED PATH ${USERBASEDIR} "/3rdparty/lib")
	else()
		find_library(HDF5_LIB NAMES libhdf5 REQUIRED PATH ${USERBASEDIR} "/3rdparty/lib")
		find_library(HDF5_CPP_LIB NAMES libhdf5_cpp REQUIRED PATH ${USERBASEDIR} "/3rdparty/lib")
	endif()
	find_path(RAPIDJSON_INCLUDE_DIR NAMES rapidjson REQUIRED PATH ${USERBASEDIR} "/3rdparty/include")
	add_definitions(-D__USERNAME__= $ENV{USERNAME})
	set(LINK_LIBRARIES ${HDF5_LIB} ${HDF5_CPP_LIB})
else()
	find_path(HDFS_INCLUDE_DIR hdf5.h REQUIRED)
	find_library(HDF5_LIB NAMES hdf5 REQUIRED)
	find_library(HDF5_CPP_LIB NAMES hdf5_cpp REQUIRED)
	find_path(RAPIDJSON_INCLUDE_DIR NAMES rapidjson REQUIRED)
	add_definitions(-D__USERNAME__=$ENV{USER})
	set(LINK_LIBRARIES ${HDF5_LIB} ${HDF5_CPP_LIB} dl)
endif()

#external headers
include_directories(${HDFS_INCLUDE_DIR} ${RAPIDJSON_INCLUDE_DIR})

set(
	SOURCES
	main.cxx
	NewStrategy.cxx
	AgressiveStrategy.cxx
	MyStrategy.cxx
	Instrument.cxx
	TradingIndicators.cxx
	StrategyBase.cxx
	Trade.cxx
	Order.cxx
	OrderContainer.cxx
	Hdf5Reader.cxx
	Hdf5FixedFormatReader.cxx
	Hdf5DataConsumer.cxx
	TradingIndicatorFileReader.cxx
	HistoricalDataHolder.cxx
	Tester.cxx
	Configurable.cxx
	TickLadder.cxx
	IndicatorLogger.cxx
	MovingObserverStrategy.cxx
	Window.cxx
	LightGBMPlugin.cxx
	PredictionTestStrategy.cxx
	SCStrategy.cpp
	Hdf5TableFormatReader.cxx
	MultiInstrMktDataFileReader.cxx
	SingleInstrMktDataFileReader.cxx
	DebugConsole.cxx
)

# source
add_executable(backtest ${SOURCES})

#link libraries
message(STATUS "HDF5 libraries: " ${HDF5_LIB} ", " ${HDF5_CPP_LIB})
target_link_libraries(backtest ${LINK_LIBRARIES})
