cmake_minimum_required(VERSION 3.10)
project(HFT_Functional_Control_Cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add matplotlib-cpp as a dependency (you'll need to download this separately)
# You can get it from: https://github.com/lava/matplotlib-cpp
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/matplotlib-cpp)

# Add source files
set(SOURCES
    src/OptimalExecutionSimulator.cpp
    src/TradingNetwork.cpp
    src/NeuralNetworkController.cpp
    src/ClosedFormController.cpp
    src/ModelExplainability.cpp
    src/main.cpp
)

# Create main executable
add_executable(hft_functional_control ${SOURCES})

# Link libraries
target_link_libraries(hft_functional_control ${TORCH_LIBRARIES})
target_link_libraries(hft_functional_control Eigen3::Eigen)
target_link_libraries(hft_functional_control Python::Python)  # For matplotlib-cpp

# Create example executable
add_executable(simple_example examples/simple_example.cpp ${SOURCES})

# Link libraries for example
target_link_libraries(simple_example ${TORCH_LIBRARIES})
target_link_libraries(simple_example Eigen3::Eigen)

# Set C++ flags
set_target_properties(hft_functional_control PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# If using CUDA
if(TORCH_CUDA_AVAILABLE)
    target_compile_definitions(hft_functional_control PRIVATE WITH_CUDA)
endif()

# Installation
install(TARGETS hft_functional_control DESTINATION bin)

# Add a README file
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/README.md "# High-Frequency Trading Functional Control

This project implements a C++ version of the 'Learning a Functional Control for High-Frequency Trading' algorithm.

## Dependencies

- CMake (>= 3.10)
- LibTorch (PyTorch C++ API)
- Eigen3
- Python (for matplotlib-cpp)

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

```bash
./hft_functional_control
```

## Features

- Optimal execution simulator for high-frequency trading
- Neural network controller for trading decisions
- Closed-form controller based on PDE solution
- Model explainability tools
- Support for multi-preference learning
- Seasonality modeling
- Sub-diffusive loss functions
")

# Add a .gitignore file
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/.gitignore "# Build directories
build/
cmake-build-*/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# Compiled files
*.o
*.obj
*.exe
*.out
*.a
*.so
*.dll
*.lib

# Python cache
__pycache__/
*.py[cod]

# Torch model files
*.pt

# Generated plots
*.png
*.jpg
*.pdf

# Logs
*.log
")
