@echo off
cd /d "%~dp0"
echo ========================================
echo   RC Car PC Controller
echo ========================================
echo   키보드 조종 + 센서 모니터 동시 실행
echo ========================================

start "Sensor Monitor" python sensor_monitor.py
timeout /t 2 >nul
python keyboard_client.py
pause
