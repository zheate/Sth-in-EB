@echo off
echo ========================================
echo   停止光耦数据分析系统
echo ========================================
echo.

echo [INFO] 正在查找并停止 Streamlit 进程...
echo.

REM 尝试停止streamlit进程
taskkill /F /IM streamlit.exe 2>nul
if %errorlevel%==0 (
    echo [OK] Streamlit 进程已停止
) else (
    echo [INFO] 未找到 streamlit.exe 进程
)

REM 尝试停止包含streamlit的python进程
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr /C:"PID:"') do (
    wmic process where "ProcessId=%%i" get CommandLine 2>nul | findstr /C:"streamlit" >nul
    if not errorlevel 1 (
        taskkill /F /PID %%i >nul 2>&1
        echo [OK] 已停止 Python Streamlit 进程 (PID: %%i^)
    )
)

REM 尝试停止占用8501端口的进程
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    if not errorlevel 1 (
        echo [OK] 已停止占用端口8501的进程 (PID: %%a^)
    )
)

echo.
echo [INFO] 清理完成
echo.
echo 按任意键退出...
pause >nul
