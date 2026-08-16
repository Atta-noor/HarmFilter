@echo off
echo Starting Hate Speech Detection App in Google Chrome...
echo.
start chrome.exe http://localhost:8501
timeout /t 3 /nobreak >nul
streamlit run app.py --server.headless false





