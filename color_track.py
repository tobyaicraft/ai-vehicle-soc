"""
파랑색 물체 추적 - 카메라가 서보모터로 따라감
서보: GPIO 18 (Pin 12) - 하드웨어 PWM (pwmchip0/pwm2)
PC 브라우저에서 http://toby.local:5000 으로 영상 확인 가능
"""

from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time

app = Flask(__name__)

# === 하드웨어 PWM 서보 제어 (pwmchip0, channel 2) ===
PWM_PATH = "/sys/class/pwm/pwmchip0/pwm2"
PERIOD = 20000000       # 20ms (50Hz)
MIN_DUTY = 500000       # 0.5ms = 0도
MAX_DUTY = 2500000      # 2.5ms = 180도


def pwm_write(filename, value):
    with open(f"{PWM_PATH}/{filename}", 'w') as f:
        f.write(str(value))


def pwm_setup():
    # export (이미 되어있으면 무시)
    try:
        with open("/sys/class/pwm/pwmchip0/export", 'w') as f:
            f.write("2")
    except:
        pass
    time.sleep(0.1)
    pwm_write("period", PERIOD)
    pwm_write("duty_cycle", 1500000)  # 90도 시작
    pwm_write("enable", 1)


def set_angle(angle):
    angle = max(0, min(180, angle))
    duty = int(MIN_DUTY + (MAX_DUTY - MIN_DUTY) * angle / 180)
    pwm_write("duty_cycle", duty)


# 서보 초기화
pwm_setup()
current_angle = 90.0

# 카메라 설정
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
camera.configure(config)
camera.start()
time.sleep(1)

# 파랑색 HSV 범위
BLUE_LOW = np.array([100, 80, 50])
BLUE_HIGH = np.array([130, 255, 255])


def track_blue():
    global current_angle

    while True:
        frame = camera.capture_array()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 파랑색 마스크
        mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 500:
                M = cv2.moments(largest)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 화면에 표시
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "BLUE", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 서보 제어
                frame_center = 320
                error = cx - frame_center

                # 중앙 ±30px이면 정지
                if abs(error) > 30:
                    step = error * 0.03
                    current_angle = max(0, min(180, current_angle - step))
                    set_angle(current_angle)

                cv2.putText(frame, f"Angle: {current_angle:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No blue detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.line(frame, (320, 0), (320, 480), (255, 255, 255), 1)

        # 마스크 나란히 표시
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, mask_color))

        ret, buffer = cv2.imencode('.jpg', combined)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return '''<html><body style="background:#111;color:#fff;text-align:center;font-family:sans-serif;">
    <h1>Blue Object Tracker</h1>
    <p>파랑색 물체를 추적하여 서보모터가 따라갑니다 (HW PWM)</p>
    <img src="/video" width="1280">
    </body></html>'''


@app.route('/video')
def video():
    return Response(track_blue(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        print("파랑색 추적 시작! (하드웨어 PWM)")
        print("PC 브라우저: http://toby.local:5000")
        app.run(host='0.0.0.0', port=5000)
    finally:
        pwm_write("enable", 0)
        camera.stop()
