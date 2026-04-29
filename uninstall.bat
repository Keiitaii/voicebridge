@echo off
REM Double-click this to remove all voicebridge shortcuts.

cd /d "%~dp0"
python autostart.py uninstall-all
echo.
pause
