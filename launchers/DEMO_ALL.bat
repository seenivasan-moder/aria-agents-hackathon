@echo off
title Airia Sentinel - 16-Agent Pipeline Demo
cd /d "F:\BUREAU\aria-agents-hackathon-private"
echo.
echo  ================================================================
echo   AIRIA SENTINEL - 16-Agent Multi-AI Pipeline
echo   Hackathon Demo: Market Scan to Executive Report in 10 seconds
echo  ================================================================
echo.
echo  [1] Full Pipeline v2 (16 agents - DEFAULT)
echo  [2] Demo All (Sentinel + Organizer + JARVIS + Matrix)
echo  [3] Dashboard Web (http://127.0.0.1:8900/dashboard)
echo  [4] Pipeline v1 (4 agents - legacy)
echo.
set /p CHOICE="Select mode [1-4, default=1]: "
if "%CHOICE%"=="" set CHOICE=1
if "%CHOICE%"=="1" "C:\Users\franc\.local\bin\uv.exe" run python main.py pipeline
if "%CHOICE%"=="2" "C:\Users\franc\.local\bin\uv.exe" run python main.py demo-all
if "%CHOICE%"=="3" "C:\Users\franc\.local\bin\uv.exe" run python main.py dashboard
if "%CHOICE%"=="4" "C:\Users\franc\.local\bin\uv.exe" run python main.py pipeline-v1
pause
