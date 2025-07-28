@echo off
echo =======================================================
echo Simple Database Test
echo =======================================================
echo.

echo Compiling simple database test...
g++ -std=c++17 -o simple_db_test.exe simple_db_test.cpp

if %ERRORLEVEL% EQU 0 (
    echo Compilation successful!
    echo.
    echo Running database test...
    echo.
    simple_db_test.exe
) else (
    echo Compilation failed!
)

echo.
pause
