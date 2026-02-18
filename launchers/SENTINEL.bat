@echo off
title Airia Sentinel
cd /d "F:\BUREAU\aria-agents-hackathon-private"
echo.
echo  ============================================
echo   AIRIA SENTINEL - Treasury Risk Management
echo  ============================================
echo.
"C:\Users\franc\.local\bin\uv.exe" run python main.py %*
pause
