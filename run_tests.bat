@echo off
echo Running Complete End-to-End Test Suite...
call .\venv\Scripts\activate.bat
python test_pipeline.py
pause

