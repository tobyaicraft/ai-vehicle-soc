"""
RC Car UART Server (RPi5)
TCP 소켓으로 PC 키보드 명령 수신 + TC237 센서 데이터 WiFi 중계

포트 구성:
 - 9000 : PC → RPi5 명령 수신 (F/B/L/R/S/U/I)
 - 9001 : RPi5 → PC 센서 브로드캐스트 ("L:xxxx,R:xxxx,U:xxx" 라인)

UART (ttyAMA2, 115200 8N1):
 - TX: F/B/L/R/S 명령 → TC237
 - RX: TC237 센서 라인 수신 → 9001 접속된 모든 클라이언트로 전송

서보 제어:
 - U/I : RPi GPIO 18 하드웨어 PWM으로 서보모터 직접 제어
         (pwmchip0/pwm2 = GPIO18 = PWM0_CHAN2)

서보는 하드웨어 PWM을 사용하므로 sudo 실행 필요 (sysfs 접근):
    sudo python3 uart_server.py

사전 조건:
    /boot/firmware/config.txt 에 아래가 [all] 앞에 있어야 함:
        [pi5]
        dtoverlay=pwm-2chan,pin=18,func=2
"""

import socket
import sys
import threading
import time
import serial

# --- Configuration ---
DEFAULT_CMD_PORT = 9000
DEFAULT_SENSOR_PORT = 9001
DEFAULT_UART = "/dev/ttyAMA2"
BAUD_RATE = 115200

# --- 하드웨어 PWM 서보 (pwmchip0, channel 2 = GPIO 18) ---
PWM_PATH = "/sys/class/pwm/pwmchip0/pwm2"
PWM_CHIP = "/sys/class/pwm/pwmchip0"
PERIOD = 20000000          # 20ms = 50Hz
MIN_DUTY = 500000          # 0.5ms = 0도
MAX_DUTY = 2500000         # 2.5ms = 180도
INIT_ANGLE = 90.0          # 초기 각도
ANGLE_STEP = 10.0          # U/I 한 번 누를 때 이동 각도

CMD_NAMES = {
    'F': '전진 (Forward)',
    'B': '후진 (Backward)',
    'L': '좌회전 (Left)',
    'R': '우회전 (Right)',
    'S': '정지 (Stop)',
    'U': '서보 왼쪽',
    'I': '서보 오른쪽',
    'P': '자동 주차 (Auto Parking)',
}

UART_CMDS = {'F', 'B', 'L', 'R', 'S', 'P'}
SERVO_CMDS = {'U', 'I'}


# --- 하드웨어 PWM 헬퍼 ---
def pwm_write(filename, value):
    with open(f"{PWM_PATH}/{filename}", 'w') as f:
        f.write(str(value))


def pwm_setup():
    try:
        with open(f"{PWM_CHIP}/export", 'w') as f:
            f.write("2")
    except OSError:
        pass  # 이미 export된 경우
    time.sleep(0.1)
    pwm_write("period", PERIOD)
    pwm_write("duty_cycle", 1500000)  # 90도로 시작
    pwm_write("enable", 1)


def pwm_teardown():
    try:
        pwm_write("enable", 0)
    except OSError:
        pass


def angle_to_duty(angle):
    angle = max(0.0, min(180.0, angle))
    return int(MIN_DUTY + (MAX_DUTY - MIN_DUTY) * angle / 180.0)


class ServoState:
    def __init__(self):
        self.angle = INIT_ANGLE

    def step(self, direction):
        """direction: -1 = 왼쪽, +1 = 오른쪽"""
        self.angle = max(0.0, min(180.0, self.angle + direction * ANGLE_STEP))
        pwm_write("duty_cycle", angle_to_duty(self.angle))
        return self.angle


# --- 센서 브로드캐스트 클라이언트 집합 (9001 접속) ---
sensor_clients = set()
sensor_lock = threading.Lock()
# UART write는 명령 처리 스레드에서만 호출되지만, 향후 확장 대비 락 하나 둠
uart_write_lock = threading.Lock()


def handle_command(cmd_byte, ser, servo):
    cmd = cmd_byte.decode('ascii', errors='ignore')
    name = CMD_NAMES.get(cmd, f'Unknown({cmd})')

    if cmd in UART_CMDS:
        with uart_write_lock:
            ser.write(cmd_byte)
        print(f"  [RX→UART]  {cmd} → {name}")
    elif cmd in SERVO_CMDS:
        direction = +1 if cmd == 'U' else -1
        angle = servo.step(direction)
        print(f"  [RX→SERVO] {cmd} → {name}  (angle={angle:.1f}°)")
    else:
        print(f"  [RX] {cmd} → {name} (무시)")


def broadcast_sensor_line(line):
    """연결된 모든 9001 클라이언트에 센서 라인 전송. 끊어진 소켓은 제거."""
    payload = (line + "\n").encode("ascii", errors="ignore")
    dead = []
    with sensor_lock:
        clients = list(sensor_clients)
    for sock in clients:
        try:
            sock.sendall(payload)
        except (OSError, BrokenPipeError):
            dead.append(sock)
    if dead:
        with sensor_lock:
            for sock in dead:
                sensor_clients.discard(sock)
                try:
                    sock.close()
                except OSError:
                    pass


def uart_reader_thread(ser, stop_event):
    """TC237 → UART 센서 라인 수신 → 9001 클라이언트에 브로드캐스트."""
    buffer = ""
    while not stop_event.is_set():
        try:
            waiting = ser.in_waiting
            if waiting:
                raw = ser.read(waiting)
                buffer += raw.decode("ascii", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    broadcast_sensor_line(line)
                    print(f"  [UART→9001] {line}")
            else:
                time.sleep(0.01)
        except serial.SerialException as e:
            print(f"[UART reader] Serial error: {e}")
            break


def sensor_server_thread(port, stop_event):
    """9001 accept 루프: 새로 접속한 PC 모니터를 sensor_clients 에 등록."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(4)
    server.settimeout(1.0)

    print(f"  [Sensor] Listening on 0.0.0.0:{port}")

    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        print(f"\n[Sensor Connected] {addr[0]}:{addr[1]}")
        with sensor_lock:
            sensor_clients.add(conn)

    server.close()


def run_server(cmd_port, sensor_port, uart_port):
    # UART 초기화
    ser = serial.Serial(uart_port, BAUD_RATE, timeout=0)

    # 하드웨어 PWM 서보 초기화
    pwm_setup()
    servo = ServoState()

    # 백그라운드 스레드 기동
    stop_event = threading.Event()
    t_reader = threading.Thread(target=uart_reader_thread,
                                args=(ser, stop_event), daemon=True)
    t_sensor = threading.Thread(target=sensor_server_thread,
                                args=(sensor_port, stop_event), daemon=True)
    t_reader.start()
    t_sensor.start()

    # 명령 TCP 서버 (9000) — 메인 스레드
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', cmd_port))
    server.listen(1)

    print("=" * 40)
    print("  RC Car UART Server (RPi5)")
    print("=" * 40)
    print(f"  CMD   : 0.0.0.0:{cmd_port}    (PC → RPi → TC237)")
    print(f"  SENSOR: 0.0.0.0:{sensor_port}    (TC237 → RPi → PC)")
    print(f"  UART  : {uart_port} @ {BAUD_RATE}")
    print(f"  SERVO : GPIO 18 HW PWM (pwmchip0/pwm2, init={INIT_ANGLE:.0f}°)")
    print("  Waiting for PC client...")
    print("=" * 40)

    try:
        while True:
            conn, addr = server.accept()
            print(f"\n[Cmd Connected] Client: {addr[0]}:{addr[1]}")

            try:
                while True:
                    data = conn.recv(1)
                    if not data:
                        break
                    handle_command(data, ser, servo)
            except ConnectionResetError:
                pass

            conn.close()
            print(f"[Cmd Disconnected] {addr[0]}:{addr[1]}")
            print("  Waiting for reconnection...\n")
    finally:
        stop_event.set()
        ser.close()
        pwm_teardown()


def main():
    cmd_port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CMD_PORT
    sensor_port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SENSOR_PORT
    uart_port = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_UART

    try:
        run_server(cmd_port, sensor_port, uart_port)
    except KeyboardInterrupt:
        print("\n[Exit] Server stopped")
    except serial.SerialException as e:
        print(f"[Error] UART: {e}")
        print("  - raspi-config에서 Serial Port 활성화 확인")
    except PermissionError as e:
        print(f"[Error] PWM sysfs 접근 권한 없음: {e}")
        print("  - sudo로 실행하세요: sudo python3 uart_server.py")
    finally:
        pwm_teardown()


if __name__ == "__main__":
    main()
