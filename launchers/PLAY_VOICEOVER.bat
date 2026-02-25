@echo off
title Airia Sentinel - Voiceover Player
cd /d "F:\BUREAU\aria-agents-hackathon-private"
echo.
echo  ================================================================
echo   AIRIA SENTINEL - Demo Voiceover (en-US-GuyNeural)
echo  ================================================================
echo.
echo  [1] Play voiceover (existing MP3)
echo  [2] Regenerate voiceover (Edge TTS)
echo  [3] Play + Launch Pipeline simultaneously
echo.
set /p CHOICE="Select [1-3, default=1]: "
if "%CHOICE%"=="" set CHOICE=1
if "%CHOICE%"=="1" start "" "data\voiceover\airia_sentinel_demo.mp3"
if "%CHOICE%"=="2" (
    echo Generating...
    "C:\Users\franc\.local\bin\uv.exe" run python scripts/generate_voiceover.py
    echo.
    start "" "data\voiceover\airia_sentinel_demo.mp3"
)
if "%CHOICE%"=="3" (
    start "" "data\voiceover\airia_sentinel_demo.mp3"
    timeout /t 2 >nul
    "C:\Users\franc\.local\bin\uv.exe" run python main.py pipeline
)
pause
