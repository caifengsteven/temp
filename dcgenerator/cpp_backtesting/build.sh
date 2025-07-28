#!/bin/bash

echo "Building DC Generator Backtesting Project..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Build completed successfully!"
    echo "Executable location: build/bin/DCGeneratorBacktesting"
    echo ""
    echo "To run the backtest:"
    echo "./bin/DCGeneratorBacktesting --help"
else
    echo ""
    echo "Build failed!"
    exit 1
fi

cd ..
