@echo off
echo ========================================
echo    KNEECHAT BOT LAUNCHER
echo ========================================
echo.
echo Which version do you want to run?
echo.
echo 1. Groq Version (Fast, Cloud-based)
echo    - Uses GROQ API
echo    - Token: TELEGRAM_TOKEN2 (7672372919...)
echo    - File: app_groq.py + chatbot2.py
echo.
echo 2. Hugging Face Version (Local LLaMA)
echo    - Uses local LLaMA 3.1 8B model
echo    - Token: TELEGRAM_TOKEN (7904367436...)
echo    - File: app_hf.py + chatbot_hf.py
echo.
echo 3. Exit
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting Groq version...
    echo Using TELEGRAM_TOKEN2: 7672372919...
    echo.
    python src\app_groq.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Hugging Face version...
    echo Using TELEGRAM_TOKEN: 7904367436...
    echo WARNING: This requires GPU and will download ~16GB model
    echo.
    python src\app_hf.py
) else if "%choice%"=="3" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    pause
    goto :start
)

pause
