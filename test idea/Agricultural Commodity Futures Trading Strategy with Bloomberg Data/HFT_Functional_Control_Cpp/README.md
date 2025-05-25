# High-Frequency Trading Functional Control in C++

This project is a C++ implementation of the "Learning a Functional Control for High-Frequency Trading" algorithm, originally implemented in Python using PyTorch.

## Overview

The project implements a neural network-based approach to optimal execution in high-frequency trading. It includes:

1. **Optimal Execution Simulator**: Simulates the trading environment with price impact models
2. **Neural Network Controller**: Learns trading strategies from simulated data
3. **Closed-Form Controller**: Implements the analytical solution for comparison
4. **Model Explainability Tools**: Projects learned controls onto interpretable functions

## Dependencies

- **C++17 compiler**: GCC, Clang, or MSVC
- **CMake** (>= 3.10)
- **LibTorch**: PyTorch C++ API
- **Eigen3**: Linear algebra library
- **matplotlib-cpp**: C++ wrapper for matplotlib (for visualization)
- **Python**: Required for matplotlib-cpp

## Installation

### 1. Install LibTorch

Download LibTorch from the [PyTorch website](https://pytorch.org/get-started/locally/). Choose the C++ version that matches your system.

```bash
# Example for Linux
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip
```

### 2. Install Eigen3

```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen

# Windows (with vcpkg)
vcpkg install eigen3
```

### 3. Install matplotlib-cpp

Clone the matplotlib-cpp repository:

```bash
git clone https://github.com/lava/matplotlib-cpp.git external/matplotlib-cpp
```

### 4. Build the project

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

## Usage

### Main Program

Run the main executable:

```bash
./hft_functional_control
```

This will:
1. Train neural network controllers on simulated data
2. Compare the learned controllers with the closed-form solution
3. Generate plots showing the results

### Simple Example

For a quick demonstration, run the simple example:

```bash
./simple_example
```

This example:
1. Creates a simulator and controllers
2. Trains a neural network controller with fewer iterations
3. Compares the neural network controller with the closed-form solution
4. Prints the results to the console

## Project Structure

```
HFT_Functional_Control_Cpp/
├── include/
│   ├── OptimalExecutionSimulator.h
│   ├── TradingNetwork.h
│   ├── NeuralNetworkController.h
│   ├── ClosedFormController.h
│   └── ModelExplainability.h
├── src/
│   ├── OptimalExecutionSimulator.cpp
│   ├── TradingNetwork.cpp
│   ├── NeuralNetworkController.cpp
│   ├── ClosedFormController.cpp
│   ├── ModelExplainability.cpp
│   └── main.cpp
├── examples/
│   └── simple_example.cpp
├── CMakeLists.txt
└── README.md
```

## Features

- **Optimal Execution Simulation**: Realistic market simulation with price impact
- **Neural Network Training**: Learn trading strategies from simulated data
- **Multi-Preference Learning**: Support for different risk preferences
- **Seasonality Modeling**: Account for intraday seasonality patterns
- **Sub-Diffusive Loss**: Support for non-quadratic inventory penalties
- **Model Explainability**: Project learned controls onto interpretable functions

## Extending the Project

### Adding New Controllers

Create a new class that implements the control function:

```cpp
class MyController {
public:
    double control(const std::vector<double>& state) {
        // Your control logic here
        return trading_rate;
    }
};
```

### Modifying the Simulator

The `OptimalExecutionSimulator` class can be extended to include more complex market dynamics:

```cpp
class EnhancedSimulator : public OptimalExecutionSimulator {
public:
    // Override methods to add new features
};
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the paper "Learning a Functional Control for High-Frequency Trading" and its original Python implementation.
