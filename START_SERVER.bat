@echo off
echo Starting FaceWatch Server...
cd /d C:\FaceWatch
uvicorn backend.main:app --reload --host 0.0.0.0 --port 5000
pause


