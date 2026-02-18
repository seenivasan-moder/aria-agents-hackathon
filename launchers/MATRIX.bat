@echo off
title Airia Sentinel - Matrix Pipeline
cd /d "F:\BUREAU\aria-agents-hackathon-private"
echo.
echo  ============================================
echo   MATRIX - Pipeline Engine Orchestration
echo   Domino + Vectorial + Matrix + Composite
echo  ============================================
echo.
"C:\Users\franc\.local\bin\uv.exe" run python main.py demo-matrix
pause
