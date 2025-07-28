cmake_minimum_required(VERSION 3.16)
project(DCGeneratorBacktesting)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(SQLite3 REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/dc_generator.cpp
    src/database_reader.cpp
    src/backtesting_engine.cpp
    src/order_manager.cpp
    src/performance_metrics.cpp
    src/market_data.cpp
)

# Header files
set(HEADERS
    include/dc_generator.h
    include/database_reader.h
    include/backtesting_engine.h
    include/order_manager.h
    include/performance_metrics.h
    include/market_data.h
    include/types.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} SQLite::SQLite3)

# Compiler flags for optimization and warnings
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -O3>
)

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
