@echo off
REM Double-click this to install voicebridge shortcuts (autostart + desktop + Start menu).
REM Requires Python 3.12+ on PATH and `pip install pywin32` already done.

cd /d "%~dp0"
python autostart.py install-all
echo.
pause
