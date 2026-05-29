@echo off
cd /d "%~dp0"
echo ========================================
echo   RC Car Control Panel
echo ========================================
echo   카메라 + 키보드 조종 + 센서 모니터 통합
echo ========================================

python control_panel.py %*
pause
