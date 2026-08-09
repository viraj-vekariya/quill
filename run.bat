@echo off
REM Cross-platform convenience runner: venv + deps + full training run.
setlocal
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Running tests...
python -m pytest tests\ -q
if errorlevel 1 exit /b 1

echo Training (this takes ~15-25 minutes)...
python train.py %*
endlocal
