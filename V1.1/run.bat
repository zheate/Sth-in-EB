@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%app"
set "ENV_NAME=qwe"
set "ARCHIVE=%APP_DIR%\%ENV_NAME%-conda-pack.tar.gz"
set "ENV_DIR=%APP_DIR%\%ENV_NAME%"
set "PYTHON_EXE=%ENV_DIR%\python.exe"
set "PYTHON_FALLBACK=%ENV_DIR%\Scripts\python.exe"
set "CONDUNPACK=%ENV_DIR%\Scripts\conda-unpack.exe"
set "MARKER=%ENV_DIR%\.conda_unpacked"

cls
echo ========================================
echo   Data Analysis Tool - Portable Version
echo ========================================
echo.
echo [INFO] Script directory: %SCRIPT_DIR%
echo [INFO] App directory: %APP_DIR%
echo.

REM Check if app folder exists
if not exist "%APP_DIR%" (
    echo [ERROR] app folder not found: %APP_DIR%
    echo.
    echo [INFO] Please ensure the app folder is in the same directory as this script.
    echo.
    echo Current directory contents:
    dir /b
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

REM Check for Python executable in fallback location
if not exist "%PYTHON_EXE%" (
    if exist "%PYTHON_FALLBACK%" (
        set "PYTHON_EXE=%PYTHON_FALLBACK%"
    )
)

REM Check if environment folder already exists
if exist "%ENV_DIR%" (
    echo [INFO] Found existing environment folder: %ENV_DIR%
    if exist "%PYTHON_EXE%" (
        echo [OK] Python found: %PYTHON_EXE%
        goto :start_app
    )
    if exist "%PYTHON_FALLBACK%" (
        set "PYTHON_EXE=%PYTHON_FALLBACK%"
        echo [OK] Python found: %PYTHON_FALLBACK%
        goto :start_app
    )
    echo [WARN] Environment folder exists but Python not found
    echo [INFO] Will try to extract from archive...
    echo.
)

REM If Python not found, check for archive or try to extract
if not exist "%PYTHON_EXE%" (
    if not exist "%ARCHIVE%" (
        echo [ERROR] Python environment archive not found
        echo.
        echo Expected location: %ARCHIVE%
        echo.
        echo [INFO] This is a PORTABLE version that requires a pre-packaged environment.
        echo.
        echo If you have conda installed on THIS computer:
        echo   - Use START.bat or run_direct.bat instead
        echo.
        echo To create the portable environment package:
        echo   1. Run: create_portable_package.bat
        echo   OR
        echo   2. Manually run these commands:
        echo      conda activate qwe
        echo      conda install conda-pack
        echo      conda pack -n qwe -o app\qwe-conda-pack.tar.gz
        echo.
        echo Press any key to exit...
        pause >nul
        exit /b 1
    )
    
    echo [INFO] First run detected - extracting Python environment...
    echo [INFO] This may take 3-5 minutes, please wait...
    echo.
    
    if not exist "%ENV_DIR%" (
        mkdir "%ENV_DIR%"
        if errorlevel 1 (
            echo [ERROR] Failed to create folder: %ENV_DIR%
            echo.
            echo Press any key to exit...
            pause >nul
            exit /b 1
        )
    )
    
    tar -xf "%ARCHIVE%" -C "%ENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Extraction failed
        echo.
        echo [INFO] Please ensure:
        echo   - Windows 10 or later (tar command required)
        echo   - Sufficient disk space (2GB+)
        echo.
        echo Or manually extract %ARCHIVE% to %ENV_DIR%
        echo.
        echo Press any key to exit...
        pause >nul
        exit /b 1
    )
    
    echo [INFO] Extraction completed successfully
    echo.
)

REM Check again for Python after extraction
if not exist "%PYTHON_EXE%" (
    if exist "%PYTHON_FALLBACK%" (
        set "PYTHON_EXE=%PYTHON_FALLBACK%"
    )
)

if not exist "%PYTHON_EXE%" (
    echo [ERROR] python.exe not found after extraction
    echo.
    echo [INFO] The archive may be corrupted. Please recreate it using:
    echo        conda pack -n qwe -o app\qwe-conda-pack.tar.gz
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

REM Run conda-unpack if needed
if not exist "%MARKER%" (
    if exist "%CONDUNPACK%" (
        echo [INFO] Configuring environment paths...
        "%CONDUNPACK%"
        if errorlevel 1 (
            echo [WARN] Path configuration failed. The app may still work.
        ) else (
            echo done>"%MARKER%"
            echo [INFO] Environment configured successfully
        )
    ) else (
        echo [WARN] conda-unpack not found in the environment
        echo [INFO] The app should still work, but some features may be limited
    )
    echo.
)

:start_app
REM Set environment variables
set "PATH=%ENV_DIR%;%ENV_DIR%\Library\bin;%ENV_DIR%\Scripts;%ENV_DIR%\bin;%PATH%"
set "PYTHONPATH=%ENV_DIR%\Lib;%ENV_DIR%\Lib\site-packages"

REM Create Streamlit config if needed
set "STREAMLIT_CONFIG=%APP_DIR%\.streamlit\config.toml"
if not exist "%STREAMLIT_CONFIG%" (
    echo [INFO] Creating Streamlit configuration...
    if not exist "%APP_DIR%\.streamlit" mkdir "%APP_DIR%\.streamlit"
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
echo [INFO] The app will open in your default browser at: http://localhost:8501
echo [INFO] To stop the app, close this window or press Ctrl+C
echo.
echo [TIPS] If the browser shows a blank page:
echo        1. Clear browser cache (Ctrl+Shift+Delete)
echo        2. Try incognito/private mode
echo        3. Check if port 8501 is already in use
echo        4. Check firewall settings
echo.

"%PYTHON_EXE%" -m streamlit run "%APP_DIR%\app.py" --server.port=8501 --server.enableCORS=false

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
pause >nul
