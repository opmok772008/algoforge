@echo off
title ECG Anomaly Detection & Cardiac Monitor
echo ========================================================
echo   ECG Anomaly Detection & Arrhythmia Classification
echo ========================================================
echo Starting Application...
python main_gui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while starting the application.
    pause
)
