@echo off
REM This script runs the main bat file in a minimized window

if not "%1"=="min" (
    start /min cmd /c "%~f0" min
    exit /b
)

REM The actual script content starts here
setlocal

set "SCRIPT_DIR=%~dp0"
set "APP_DIR=%SCRIPT_DIR%app"
set "ENV_NAME=st"

REM Check if app folder exists
if not exist "%APP_DIR%" (
    msg * "Error: app folder not found at %APP_DIR%"
    exit /b 1
)

REM Check if conda is available
where conda >nul 2>&1
if errorlevel 1 (
    msg * "Error: Conda not found in PATH. Please install Anaconda or Miniconda."
    exit /b 1
)

REM Check if st environment exists
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    msg * "Error: Conda environment '%ENV_NAME%' not found. Please create it first."
    exit /b 1
)

REM Activate conda environment and run streamlit
call conda activate %ENV_NAME%
if errorlevel 1 (
    msg * "Error: Failed to activate conda environment: %ENV_NAME%"
    exit /b 1
)

streamlit run "%APP_DIR%\app.py" --server.port=8501 --server.enableCORS=false
