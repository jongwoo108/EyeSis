@echo off
echo Starting EyeSis Server...
cd /d C:\EyeSis
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
pause


