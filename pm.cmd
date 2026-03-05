@echo off
set SCRIPT_DIR=%~dp0

where uv >nul 2>nul
if %errorlevel%==0 (
  uv run --project "%SCRIPT_DIR%" python "%SCRIPT_DIR%scripts\posementor.py" %*
  exit /b %errorlevel%
)

if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
  "%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%scripts\posementor.py" %*
  exit /b %errorlevel%
)

if exist "%SystemRoot%\py.exe" (
  py -3 "%SCRIPT_DIR%scripts\posementor.py" %*
  exit /b %errorlevel%
)

python "%SCRIPT_DIR%scripts\posementor.py" %*
