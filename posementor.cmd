@echo off
set SCRIPT_DIR=%~dp0

if exist "%SCRIPT_DIR%.venv\Scripts\posementor.exe" (
  "%SCRIPT_DIR%.venv\Scripts\posementor.exe" %*
  exit /b %errorlevel%
)

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
  "%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%posementor_cli.py" %*
  exit /b %errorlevel%
)

python "%SCRIPT_DIR%posementor_cli.py" %*
