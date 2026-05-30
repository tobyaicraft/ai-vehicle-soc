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

# --- GPS (NEO-6M on ttyAMA0) ---
GPS_PORT = "/dev/ttyAMA0"
GPS_BAUD = 9600

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
    'X': 'MCU 리셋 (Reset)',
}

UART_CMDS = {'F', 'B', 'L', 'R', 'S', 'P'}
SERVO_CMDS = {'U', 'I'}
RESET_CMD = {'X'}

# --- TC237 패킷 프로토콜 (AA LEN CMD PAYLOAD CHK 55) ---
PROTO_STX = 0xAA
PROTO_ETX = 0x55
CMD_MOVE  = 0x01
CMD_MODE  = 0x02
CMD_RESET = 0x30
DEFAULT_SPEED = 100  # 모터 속도 (0~100%)

# PC 키 → MOVE 방향 매핑
DIR_MAP = {
    'S': 0,  # STOP
    'F': 1,  # FORWARD
    'B': 2,  # REVERSE
    'L': 3,  # LEFT (spin)
    'R': 4,  # RIGHT (spin)
}


def build_move_packet(direction, speed=DEFAULT_SPEED):
    """TC237 MOVE 패킷 생성: AA 03 01 dir speed chk 55"""
    cmd = CMD_MOVE
    chk = cmd ^ direction ^ speed
    return bytes([PROTO_STX, 0x03, cmd, direction, speed, chk, PROTO_ETX])


def build_mode_packet(mode):
    """TC237 MODE 패킷 생성: AA 02 02 mode chk 55"""
    cmd = CMD_MODE
    chk = cmd ^ mode
    return bytes([PROTO_STX, 0x02, cmd, mode, chk, PROTO_ETX])


def build_reset_packet():
    """TC237 RESET 패킷 생성: AA 01 30 30 55"""
    cmd = CMD_RESET
    chk = cmd
    return bytes([PROTO_STX, 0x01, cmd, chk, PROTO_ETX])


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

# --- MOVE 명령 주기적 재전송 (MCU 200ms 타임아웃 대응) ---
MOVE_RESEND_INTERVAL = 0.1  # 100ms마다 재전송
last_move_pkt = None
last_move_lock = threading.Lock()


def move_resend_thread(ser, stop_event):
    """마지막 MOVE 명령을 100ms마다 재전송하여 MCU 타임아웃 방지"""
    while not stop_event.is_set():
        with last_move_lock:
            pkt = last_move_pkt
        if pkt:
            with uart_write_lock:
                ser.write(pkt)
        time.sleep(MOVE_RESEND_INTERVAL)


def handle_command(cmd_byte, ser, servo):
    global last_move_pkt
    cmd = cmd_byte.decode('ascii', errors='ignore')
    name = CMD_NAMES.get(cmd, f'Unknown({cmd})')

    if cmd in DIR_MAP:
        pkt = build_move_packet(DIR_MAP[cmd])
        with uart_write_lock:
            ser.write(pkt)
        with last_move_lock:
            last_move_pkt = pkt if DIR_MAP[cmd] != 0 else None
        print(f"  [RX→UART]  {cmd} → {name}  (pkt={pkt.hex()})")
    elif cmd == 'P':
        # 자동 주차: AUTO 모드 전환
        pkt = build_mode_packet(2)  # VEHICLE_MODE_AUTO
        with uart_write_lock:
            ser.write(pkt)
        print(f"  [RX→UART]  {cmd} → {name}  (pkt={pkt.hex()})")
    elif cmd == 'X':
        # MCU 소프트 리셋
        pkt = build_reset_packet()
        with uart_write_lock:
            ser.write(pkt)
        with last_move_lock:
            last_move_pkt = None
        print(f"  [RX→UART]  {cmd} → {name}  (pkt={pkt.hex()})")
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


# --- GPS NMEA Parser ---
def parse_nmea_coord(raw, direction):
    """NMEA 좌표 (ddmm.mmmm) → 십진수 도(degrees)"""
    if not raw or not direction:
        return 0.0
    try:
        if direction in ('N', 'S'):
            deg = float(raw[:2])
            minutes = float(raw[2:])
        else:
            deg = float(raw[:3])
            minutes = float(raw[3:])
        result = deg + minutes / 60.0
        if direction in ('S', 'W'):
            result = -result
        return result
    except (ValueError, IndexError):
        return 0.0


def gps_reader_thread(stop_event):
    """NEO-6M GPS 리더: NMEA 파싱 → 9001 브로드캐스트"""
    try:
        gps_ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
        print(f"  [GPS] Opened {GPS_PORT} @ {GPS_BAUD}")
    except Exception as e:
        print(f"  [GPS] Failed to open {GPS_PORT}: {e}")
        return

    lat = 0.0
    lon = 0.0
    speed_kmh = 0.0
    sats = 0

    buf = ""
    while not stop_event.is_set():
        try:
            raw = gps_ser.read(gps_ser.in_waiting or 1)
            if not raw:
                continue
            buf += raw.decode("ascii", errors="ignore")

            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()

                # $GPRMC — 위치, 속도
                if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                    parts = line.split(",")
                    if len(parts) >= 8 and parts[2] == "A":  # A=valid
                        lat = parse_nmea_coord(parts[3], parts[4])
                        lon = parse_nmea_coord(parts[5], parts[6])
                        try:
                            speed_kmh = float(parts[7]) * 1.852  # knots→km/h
                        except ValueError:
                            speed_kmh = 0.0

                # $GPGGA — 위성 수
                elif line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                    parts = line.split(",")
                    if len(parts) >= 8:
                        try:
                            sats = int(parts[7])
                        except ValueError:
                            sats = 0

                        # GPS 데이터 브로드캐스트 (GGA 수신 시 1Hz)
                        gps_line = f"G:{lat:.6f},{lon:.6f},{speed_kmh:.1f},{sats}"
                        broadcast_sensor_line(gps_line)

        except Exception:
            time.sleep(0.1)

    gps_ser.close()


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
    t_resend = threading.Thread(target=move_resend_thread,
                                args=(ser, stop_event), daemon=True)
    t_gps = threading.Thread(target=gps_reader_thread,
                              args=(stop_event,), daemon=True)
    t_reader.start()
    t_sensor.start()
    t_resend.start()
    t_gps.start()

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
    print(f"  GPS   : {GPS_PORT} @ {GPS_BAUD}")
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
