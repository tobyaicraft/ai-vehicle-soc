"""
파란색 물체 자동 추적 & 추종 (Blue Follower)
카메라로 파란색을 인식하여 서보(좌우 방향) + 모터(전진/정지/좌회전/우회전)를 자동 제어한다.

서보: GPIO 18 (Pin 12) - 하드웨어 PWM (pwmchip0/pwm2)
모터: UART2 (ttyAMA2) - TC237 MCU 연동 (F/B/L/R/S)
영상: http://<RPi_IP>:8000 으로 MJPEG 스트리밍

실행:
    sudo python3 blue_follower.py
    sudo python3 blue_follower.py --port 8000 --uart /dev/ttyAMA2
"""

import argparse
import io
import time
from threading import Condition, Thread, Lock
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import serial
from picamera2 import Picamera2


# ──────────────────────── 설정값 ────────────────────────
FRAME_W = 640
FRAME_H = 480
FRAME_CENTER_X = FRAME_W // 2   # 320

# 파랑색 HSV 범위
BLUE_LOW = np.array([100, 80, 50])
BLUE_HIGH = np.array([130, 255, 255])
MIN_AREA = 500           # 노이즈 무시 최소 면적

# 서보 (하드웨어 PWM)
PWM_PATH = "/sys/class/pwm/pwmchip0/pwm2"
PWM_CHIP = "/sys/class/pwm/pwmchip0"
PERIOD = 20_000_000      # 20ms = 50Hz
MIN_DUTY = 500_000       # 0.5ms = 0도
MAX_DUTY = 2_500_000     # 2.5ms = 180도
SERVO_CENTER = 85.0       # 카메라 정면 보정 (기계적 오프셋)

# 모터 제어 임계값
STEER_THRESHOLD = 60     # |error| > 60px → L/R 제자리 회전
CLOSE_AREA = 8000        # 면적이 이 이상이면 "너무 가까움" → 정지
LOST_PATIENCE = 15       # 이 프레임 수만큼 못 찾으면 정지

# UART
DEFAULT_UART = "/dev/ttyAMA2"
BAUD_RATE = 115200


# ──────────────────────── 하드웨어 PWM 서보 ────────────────────────
def pwm_write(filename, value):
    with open(f"{PWM_PATH}/{filename}", "w") as f:
        f.write(str(value))


def pwm_setup():
    try:
        with open(f"{PWM_CHIP}/export", "w") as f:
            f.write("2")
    except OSError:
        pass
    time.sleep(0.1)
    pwm_write("period", PERIOD)
    pwm_write("duty_cycle", int(MIN_DUTY + (MAX_DUTY - MIN_DUTY) * SERVO_CENTER / 180))
    pwm_write("enable", 1)


def pwm_teardown():
    try:
        pwm_write("enable", 0)
    except OSError:
        pass


def angle_to_duty(angle):
    angle = max(0.0, min(180.0, angle))
    return int(MIN_DUTY + (MAX_DUTY - MIN_DUTY) * angle / 180.0)


def set_servo_angle(angle):
    angle = max(0.0, min(180.0, angle))
    pwm_write("duty_cycle", angle_to_duty(angle))
    return angle


# ──────────────────────── MJPEG 스트리밍 ────────────────────────
class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def set(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


PAGE = """\
<!DOCTYPE html>
<html>
<head><title>Blue Follower</title></head>
<body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
<h2>Blue Follower - Auto Drive</h2>
<p>파란색 물체를 따라 차량이 자동으로 이동합니다</p>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""


class StreamingHandler(BaseHTTPRequestHandler):
    buffer = None  # 클래스 변수로 바인딩

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                buf = self.__class__.buffer
                while True:
                    with buf.condition:
                        buf.condition.wait()
                        frame = buf.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # HTTP 로그 무시


class StreamingServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ──────────────────────── 메인 추적 루프 ────────────────────────
def tracking_loop(picam, ser, frame_buffer):
    """카메라 정면 고정, 프레임 내 파란색 위치로 차체(모터)만 제어"""
    # 서보를 정면(90도)에 고정
    set_servo_angle(SERVO_CENTER)

    last_motor_cmd = "S"
    lost_count = 0

    while True:
        frame = picam.capture_array()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 파랑색 마스크
        mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motor_cmd = None
        status_text = ""
        status_color = (0, 0, 255)  # 빨강(기본)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > MIN_AREA:
                lost_count = 0
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(largest)

                    # 오버레이 그리기
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

                    # 수평 오차 계산
                    error = cx - FRAME_CENTER_X

                    # ── 모터 제어 (서보 없이 차체로만 조향) ──
                    if area > CLOSE_AREA:
                        # 너무 가까움 → 정지
                        motor_cmd = "S"
                        status_text = "TOO CLOSE - STOP"
                        status_color = (0, 165, 255)  # 주황
                    elif abs(error) > STEER_THRESHOLD:
                        # 컵이 좌/우에 있음 → 제자리 회전
                        if error < 0:
                            motor_cmd = "L"
                            status_text = f"TURN LEFT (err={error})"
                        else:
                            motor_cmd = "R"
                            status_text = f"TURN RIGHT (err={error})"
                        status_color = (0, 255, 255)  # 노랑
                    else:
                        # 컵이 중앙 → 직진
                        motor_cmd = "F"
                        status_text = "FORWARD"
                        status_color = (0, 255, 0)  # 초록

                    # 정보 텍스트
                    cv2.putText(frame, f"area={int(area)}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                lost_count += 1
        else:
            lost_count += 1

        # 파란색 못 찾으면 정지
        if lost_count > 0:
            if lost_count >= LOST_PATIENCE:
                motor_cmd = "S"
                status_text = "LOST - STOP"
            else:
                motor_cmd = "S"
                status_text = f"SEARCHING... ({lost_count}/{LOST_PATIENCE})"
            status_color = (0, 0, 255)

        # 모터 명령 전송 (변경 시에만)
        if motor_cmd and motor_cmd != last_motor_cmd:
            ser.write(motor_cmd.encode("ascii"))
            last_motor_cmd = motor_cmd

        # HUD 오버레이
        cv2.line(frame, (FRAME_CENTER_X, 0), (FRAME_CENTER_X, FRAME_H), (255, 255, 255), 1)
        # 좌/우 회전 기준선 표시
        cv2.line(frame, (FRAME_CENTER_X - STEER_THRESHOLD, 0),
                 (FRAME_CENTER_X - STEER_THRESHOLD, FRAME_H), (0, 255, 255), 1)
        cv2.line(frame, (FRAME_CENTER_X + STEER_THRESHOLD, 0),
                 (FRAME_CENTER_X + STEER_THRESHOLD, FRAME_H), (0, 255, 255), 1)

        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Motor: {last_motor_cmd}", (FRAME_W - 150, FRAME_H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # JPEG 인코딩 → 스트리밍 버퍼
        ret, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            frame_buffer.set(jpg.tobytes())


# ──────────────────────── 엔트리포인트 ────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Blue Follower - 파란색 물체 자동 추종")
    parser.add_argument("--port", type=int, default=8000, help="MJPEG 스트리밍 포트 (기본 8000)")
    parser.add_argument("--uart", default=DEFAULT_UART, help="UART 포트 (기본 /dev/ttyAMA2)")
    args = parser.parse_args()

    # 카메라
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    ))
    picam.start()
    time.sleep(1)

    # UART
    ser = serial.Serial(args.uart, BAUD_RATE, timeout=0)

    # 서보 PWM
    pwm_setup()

    # 스트리밍 버퍼
    buf = FrameBuffer()
    StreamingHandler.buffer = buf

    # 추적 스레드
    tracker = Thread(target=tracking_loop, args=(picam, ser, buf), daemon=True)
    tracker.start()

    # HTTP 서버
    server = StreamingServer(("0.0.0.0", args.port), StreamingHandler)

    print("=" * 48)
    print("  Blue Follower - Auto Drive")
    print("=" * 48)
    print(f"  Stream : http://192.168.0.23:{args.port}")
    print(f"  UART   : {args.uart} @ {BAUD_RATE}")
    print(f"  Servo  : GPIO18 HW PWM (center={SERVO_CENTER:.0f} deg)")
    print("=" * 48)
    print("  파란색 물체를 카메라 앞에 두면 차량이 따라갑니다")
    print("  종료: Ctrl+C")
    print("=" * 48)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Exit] Blue Follower stopped")
    finally:
        ser.write(b"S")  # 안전 정지
        ser.close()
        pwm_teardown()
        picam.stop()


if __name__ == "__main__":
    main()
