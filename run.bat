@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "ENV_NAME=sth_eb"

cls
echo ========================================
echo   Data Analysis Tool
echo ========================================
echo.
echo [INFO] Script directory: %SCRIPT_DIR%
echo [INFO] Using conda environment: %ENV_NAME%
echo.

REM Check if app.py exists
if not exist "%SCRIPT_DIR%app.py" (
    echo [ERROR] app.py not found: %SCRIPT_DIR%app.py
    echo.
    echo [INFO] Please ensure app.py is in the same directory as this script.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

REM Check if conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda not found in PATH
    echo.
    echo [INFO] Please ensure Anaconda or Miniconda is installed and added to PATH.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

REM Check if st environment exists
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda environment '%ENV_NAME%' not found
    echo.
    echo [INFO] Available environments:
    conda env list
    echo.
    echo [INFO] Please create the '%ENV_NAME%' environment first or modify ENV_NAME in this script.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo [OK] Found conda environment: %ENV_NAME%
echo.

REM Create Streamlit config if needed
set "STREAMLIT_CONFIG=%SCRIPT_DIR%.streamlit\config.toml"
if not exist "%STREAMLIT_CONFIG%" (
    echo [INFO] Creating Streamlit configuration...
    if not exist "%SCRIPT_DIR%.streamlit" mkdir "%SCRIPT_DIR%.streamlit"
    (
        echo [server]
        echo headless = false
        echo port = 8501
        echo enableCORS = false
        echo enableXsrfProtection = false
        echo.
        echo [browser]
        echo gatherUsageStats = false
        echo serverAddress = "localhost"
        echo.
        echo [client]
        echo showErrorDetails = true
    ) > "%STREAMLIT_CONFIG%"
    echo [INFO] Configuration created successfully
    echo.
)

echo ========================================
echo   Starting Application...
echo ========================================
echo.
echo [INFO] Activating conda environment: %ENV_NAME%
echo [INFO] The app will open in your default browser at: http://localhost:8501
echo [INFO] To stop the app, close this window or press Ctrl+C
echo.
echo [TIPS] If the browser shows a blank page:
echo        1. Clear browser cache (Ctrl+Shift+Delete)
echo        2. Try incognito/private mode
echo        3. Check if port 8501 is already in use
echo        4. Check firewall settings
echo.

REM Activate conda environment and run streamlit
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment: %ENV_NAME%
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

streamlit run "%SCRIPT_DIR%app.py" --server.port=8501 --server.enableCORS=false

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with errors
    echo [INFO] Please check the error messages above
    echo.
) else (
    echo.
    echo [INFO] Application closed normally
    echo.
)

echo Press any key to exit...
pause > nul