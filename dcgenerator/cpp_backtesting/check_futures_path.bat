@echo off
echo =======================================================
echo US Futures Database Path Check
echo =======================================================
echo.

set DB_PATH=F:\database\us futures 1mins\us_fut_1min.db

echo Checking database path: %DB_PATH%
echo.

if exist "%DB_PATH%" (
    echo ✅ Database file exists!
    
    echo.
    echo File details:
    dir "%DB_PATH%"
    
    echo.
    echo Attempting to open with SQLite command line...
    echo (This will test if the database is valid)
    
    rem Try to use sqlite3 command if available
    sqlite3 "%DB_PATH%" ".tables" 2>nul
    if %errorlevel% equ 0 (
        echo ✅ Database is valid SQLite format
    ) else (
        echo ❌ Database might be corrupted or not SQLite format
    )
    
) else (
    echo ❌ Database file NOT found!
    echo.
    echo Please check:
    echo 1. Drive F: is accessible
    echo 2. Path exists: F:\database\us futures 1mins\
    echo 3. File name is correct: us_fut_1min.db
    echo.
    echo Checking if drive F: exists...
    if exist F:\ (
        echo ✅ Drive F: is accessible
        echo.
        echo Checking parent directory...
        if exist "F:\database\" (
            echo ✅ F:\database\ exists
            if exist "F:\database\us futures 1mins\" (
                echo ✅ F:\database\us futures 1mins\ exists
                echo.
                echo Files in directory:
                dir "F:\database\us futures 1mins\"
            ) else (
                echo ❌ F:\database\us futures 1mins\ does not exist
            )
        ) else (
            echo ❌ F:\database\ does not exist
        )
    ) else (
        echo ❌ Drive F: is not accessible
    )
)

echo.
pause
