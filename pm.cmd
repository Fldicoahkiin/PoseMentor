@echo off
set SCRIPT_DIR=%~dp0

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
  "%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%scripts\posementor.py" %*
  exit /b %errorlevel%
)

if exist "%SystemRoot%\py.exe" (
  py -3 "%SCRIPT_DIR%scripts\posementor.py" %*
  exit /b %errorlevel%
)

python "%SCRIPT_DIR%scripts\posementor.py" %*
