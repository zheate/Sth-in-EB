@echo off
chcp 65001 >nul
setlocal

echo ========================================
echo   Sth-in-EB 环境安装与启动脚本
echo ========================================
echo.

set "ENV_DIR=%~dp0sth_eb_env"
set "TAR_FILE=%~dp0sth_eb_env.tar.gz"

:: 检查环境是否已解压（python.exe在根目录）
if exist "%ENV_DIR%\python.exe" (
    echo [√] 检测到已有环境，直接启动应用...
    echo.
    goto :start_app
)

:: 解压环境
echo [1/3] 正在解压环境（首次运行需要1-2分钟）...
if not exist "%TAR_FILE%" (
    echo 错误: 找不到 sth_eb_env.tar.gz
    echo 请确保该文件与此脚本在同一目录
    pause
    exit /b 1
)
mkdir "%ENV_DIR%" 2>nul
tar -xzf "%TAR_FILE%" -C "%ENV_DIR%"
if errorlevel 1 (
    echo 解压失败，请检查 tar 命令是否可用
    pause
    exit /b 1
)
echo 解压完成!
echo.

echo [2/3] 正在修复环境路径...
call "%ENV_DIR%\Scripts\activate.bat"
conda-unpack
echo 路径修复完成!
echo.

:start_app
echo [3/3] 正在启动应用...
echo.
call "%ENV_DIR%\Scripts\activate.bat"
cd /d "%~dp0"
python -m streamlit run app/app.py

pause
