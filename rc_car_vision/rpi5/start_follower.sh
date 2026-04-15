#!/bin/bash
# Blue Follower 런처 (sudo 실행 필요 — 서보 HW PWM + UART)
# 파란색 물체를 자동으로 따라가는 자율주행 모드
#
# 실행:  sudo ~/project/start_follower.sh
# 종료:  Ctrl+C

if [ -n "$SUDO_USER" ]; then
    USER_HOME="/home/$SUDO_USER"
else
    USER_HOME="$HOME"
fi

SCRIPT="$USER_HOME/project/blue_follower.py"

if [ "$(id -u)" -ne 0 ]; then
    echo "[Error] sudo로 실행하세요: sudo $0"
    exit 1
fi

echo "========================================"
echo "  Blue Follower - Auto Drive"
echo "========================================"
echo "  브라우저: http://192.168.0.23:8000"
echo "  종료:     Ctrl+C"
echo "========================================"

exec python3 -u "$SCRIPT" "$@"
