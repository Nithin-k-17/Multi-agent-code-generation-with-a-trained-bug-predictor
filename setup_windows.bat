@echo off
REM One-time setup on Windows: creates a virtual environment and installs the Python libraries.
REM Run from the project folder:  setup_windows.bat
cd /d "%~dp0"

python --version || (echo Python 3.10+ is required: https://www.python.org/downloads/ & exit /b 1)

if not exist .venv (
    echo Creating virtual environment .venv ...
    python -m venv .venv || exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || exit /b 1

echo.
python main.py check
echo.
echo Setup finished. Next time, activate the environment with:  .venv\Scripts\activate
echo Then start the website with:  python main.py serve
