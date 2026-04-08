"""
RC Car UART Server (RPi5)
TCP 소켓으로 PC 키보드 명령 수신 → UART(ttyAMA2, GPIO4/5)로 TC237에 전달

Usage:
    python3 uart_server.py                          # 기본: 포트 9000, UART /dev/serial0
    python3 uart_server.py 9000 /dev/serial0        # 포트/UART 지정
"""

import socket
import sys
import serial

# --- Configuration ---
DEFAULT_PORT = 9000
DEFAULT_UART = "/dev/ttyAMA2"
BAUD_RATE = 115200

CMD_NAMES = {
    'F': '전진 (Forward)',
    'B': '후진 (Backward)',
    'L': '좌회전 (Left)',
    'R': '우회전 (Right)',
    'S': '정지 (Stop)',
}


def handle_command(cmd_byte, ser):
    """수신 명령 처리 — 터미널 출력 + UART 전송"""
    cmd = cmd_byte.decode('ascii', errors='ignore')
    name = CMD_NAMES.get(cmd, f'Unknown({cmd})')
    print(f"  [RX] {cmd} → {name}")

    ser.write(cmd_byte)


def run_server(port, uart_port):
    # UART 초기화
    ser = serial.Serial(uart_port, BAUD_RATE, timeout=0)
    print(f"  UART: {uart_port} @ {BAUD_RATE} baud")

    # TCP 소켓 서버
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)

    print("=" * 40)
    print("  RC Car UART Server (RPi5)")
    print("=" * 40)
    print(f"  TCP:  0.0.0.0:{port}")
    print(f"  UART: {uart_port} @ {BAUD_RATE}")
    print("  Waiting for PC client...")
    print("=" * 40)

    try:
        while True:
            conn, addr = server.accept()
            print(f"\n[Connected] Client: {addr[0]}:{addr[1]}")

            try:
                while True:
                    data = conn.recv(1)
                    if not data:
                        break
                    handle_command(data, ser)
            except ConnectionResetError:
                pass

            conn.close()
            print(f"[Disconnected] {addr[0]}:{addr[1]}")
            print("  Waiting for reconnection...\n")
    finally:
        ser.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    uart_port = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_UART

    try:
        run_server(port, uart_port)
    except KeyboardInterrupt:
        print("\n[Exit] Server stopped")
    except serial.SerialException as e:
        print(f"[Error] UART: {e}")
        print("  - raspi-config에서 Serial Port 활성화 확인")
        print("  - sudo raspi-config → Interface → Serial Port")


if __name__ == "__main__":
    main()
