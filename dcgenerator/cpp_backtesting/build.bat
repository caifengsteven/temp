@echo off
echo Building DC Generator Backtesting Project...

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
cmake .. -G "Visual Studio 16 2019" -A x64

REM Build the project
cmake --build . --config Release

REM Check if build was successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build completed successfully!
    echo Executable location: build\bin\Release\DCGeneratorBacktesting.exe
    echo.
    echo To run the backtest:
    echo bin\Release\DCGeneratorBacktesting.exe --help
) else (
    echo.
    echo Build failed with error code %ERRORLEVEL%
)

cd ..
pause
