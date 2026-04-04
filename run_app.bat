@echo off
title Run Moving Average Dashboard
echo ===================================================
echo   Adaptive Moving Average Dashboard - Docker Run
echo ===================================================
echo.

echo Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker does not seem to be running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b
)

echo Building Docker Image (this might take a few minutes the first time)...
docker build -t movingaverage-app .

if %errorlevel% neq 0 (
    echo [ERROR] Failed to build the Docker image.
    pause
    exit /b
)

echo.
echo ===================================================
echo Starting the application...
echo The dashboard will be available at http://localhost:8501
echo Opening browser...
echo ===================================================

start http://localhost:8501

echo Running Container (Press Ctrl+C to stop)...
docker run -p 8501:8501 --name movingaverage-container --rm movingaverage-app

pause
