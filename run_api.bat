@echo off
echo Starting Hate Speech Detection API...
echo.
echo API runs at http://localhost:5000
echo   Health check: http://localhost:5000/health
echo   Predict:      POST http://localhost:5000/predict
echo.
echo To expose it publicly for the Flutter app, open a SECOND terminal and run:
echo   ngrok http 5000
echo.
echo Press Ctrl+C to stop the server.
echo.
python api.py
