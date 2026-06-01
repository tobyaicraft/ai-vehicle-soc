#!/bin/bash
# ================================================================
#  RC Car 통합 런처 — 카메라(파랑 인식) + UART(센서/명령)
# ================================================================
#  기능:
#    1) camera_stream.py  : 카메라 MJPEG + 파랑색 검출   (HTTP :8000)
#    2) uart_server.py    : 키보드 명령 + 센서 브로드캐스트 (TCP :9000/:9001)
#
#  실행:  sudo ~/project/rpi5_start.sh
#  종료:  Ctrl+C
# ================================================================

# sudo 시 HOME 경로 보정
if [ -n "$SUDO_USER" ]; then
    USER_HOME="/home/$SUDO_USER"
else
    USER_HOME="$HOME"
fi

PROJECT_DIR="$USER_HOME/project"
CAM_SCRIPT="$PROJECT_DIR/camera_stream.py"
UART_SCRIPT="$PROJECT_DIR/uart_server.py"

# sudo 체크
if [ "$(id -u)" -ne 0 ]; then
    echo "[Error] sudo로 실행해야 합니다 (서보 PWM/UART 접근)"
    echo "        sudo $0"
    exit 1
fi

# 이전 프로세스 정리 (수동 실행 잔여 프로세스)
pkill -f camera_stream 2>/dev/null
pkill -f uart_server 2>/dev/null
pkill -f libcamera 2>/dev/null
sleep 2

# 파일 존재 확인
for f in "$CAM_SCRIPT" "$UART_SCRIPT"; do
    if [ ! -f "$f" ]; then
        echo "[Error] 파일 없음: $f"
        exit 1
    fi
done

# 종료 시 프로세스 정리
cleanup() {
    echo ""
    echo "[Exit] 종료 중..."
    kill "$CAM_PID" "$UART_PID" 2>/dev/null
    sleep 0.3
    kill -9 "$CAM_PID" "$UART_PID" 2>/dev/null
    echo "[Exit] 완료"
    exit 0
}
trap cleanup INT TERM

# RPi IP 자동 감지
RPI_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "========================================"
echo "  RC Car All-in-One Launcher"
echo "========================================"
echo ""

# 1) 카메라 스트리밍 + 검출
python3 -u "$CAM_SCRIPT" > >(sed 's/^/[CAM ] /') 2>&1 &
CAM_PID=$!
echo "  [1/2] Camera + Detection (PID $CAM_PID)"

sleep 2

# 2) UART 서버 (키보드 명령 + 센서 + GPS)
python3 -u "$UART_SCRIPT" > >(sed 's/^/[UART] /') 2>&1 &
UART_PID=$!
echo "  [2/2] UART + Sensor + GPS (PID $UART_PID)"

echo ""
echo "========================================"
echo "  카메라:  http://${RPI_IP}:8000"
echo "  키보드:  PC에서 run.bat 실행"
echo "  센서:    PC에서 sensor_monitor 실행"
echo "----------------------------------------"
echo "  UART:    /dev/ttyAMA2 (115200)"
echo "  명령:    TCP :9000 (F/B/L/R/S)"
echo "  센서:    TCP :9001 (L:xx,R:xx,U:xx)"
echo "========================================"
echo "  종료: Ctrl+C"
echo "========================================"
echo ""

wait
cleanup
