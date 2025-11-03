@echo off
title Project Starter

echo Starting Backend Server...
cd backend
call venv\Scripts\activate
start cmd /k "python app.py"
cd ..

echo Starting Frontend...
cd frontend
start cmd /k "npm start"
cd ..

echo All services started successfully!
pause
