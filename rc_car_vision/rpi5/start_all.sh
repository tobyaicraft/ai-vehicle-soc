#!/bin/bash
# RC Car 통합 런처 (sudo 실행 필요 — 서보 하드웨어 PWM sysfs 접근)
# - camera_stream.py  : 카메라 + 파란색 검출 (MJPEG :8000)
# - uart_server.py    : TCP→UART 모터 + GPIO18 HW PWM 서보 (:9000)
#
# 실행:  sudo ~/project/start_all.sh
# 종료:  Ctrl+C  (둘 다 정리)

# sudo로 실행됐을 때 $HOME이 /root가 되는 문제 방지
if [ -n "$SUDO_USER" ]; then
    USER_HOME="/home/$SUDO_USER"
else
    USER_HOME="$HOME"
fi

CAM_SCRIPT="$USER_HOME/project/camera_stream.py"
UART_SCRIPT="$USER_HOME/project/rc_car_keyboard/uart_server.py"

if [ "$(id -u)" -ne 0 ]; then
    echo "[Error] 이 스크립트는 sudo로 실행해야 합니다 (서보 HW PWM 접근)."
    echo "        sudo $0"
    exit 1
fi

cleanup() {
    echo ""
    echo "[Exit] 종료 중..."
    kill "$CAM_PID" "$UART_PID" 2>/dev/null
    sleep 0.3
    kill -9 "$CAM_PID" "$UART_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

echo "========================================"
echo "  RC Car All-in-One Launcher"
echo "========================================"

python3 -u "$CAM_SCRIPT"  > >(sed 's/^/[CAM ] /') 2>&1 &
CAM_PID=$!
echo "  [1/2] Camera (PID $CAM_PID)"

sleep 1

python3 -u "$UART_SCRIPT" > >(sed 's/^/[UART] /') 2>&1 &
UART_PID=$!
echo "  [2/2] UART+Servo (PID $UART_PID)"

echo "========================================"
echo "  브라우저: http://192.168.0.23:8000"
echo "  키보드:   run.bat 더블클릭 (노트북)"
echo "  종료:     Ctrl+C"
echo "========================================"

wait -n
cleanup
