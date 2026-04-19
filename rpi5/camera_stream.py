"""
RPi5 카메라 MJPEG 스트리머 + 파란색 물체 검출 (OpenCV)
노트북 브라우저에서 http://<RPi_IP>:8000 으로 실시간 확인

파랑 물체를 찾아 초록 박스 + 중심점을 그려서 스트리밍한다.
"""
import io
import time
from threading import Condition, Thread
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
from picamera2 import Picamera2


# --- 설정 ---
FRAME_W = 640
FRAME_H = 480

# 파랑색 HSV 범위 (color_track.py와 동일)
BLUE_LOW = np.array([100, 80, 50])
BLUE_HIGH = np.array([130, 255, 255])

# 노이즈로 간주할 최소 컨투어 면적
MIN_AREA = 500


PAGE = """\
<!DOCTYPE html>
<html>
<head><title>RPi5 Blue Tracker</title></head>
<body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
<h2>RPi5 Camera - Blue Object Detection</h2>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""


class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def set(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


def detect_blue(frame):
    """파랑 물체 검출 + 오버레이 그리기. 입력/출력 모두 BGR numpy 배열."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > MIN_AREA:
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(largest)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"BLUE area={int(area)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No blue", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No blue", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 중앙선 참고용
    cv2.line(frame, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (255, 255, 255), 1)
    return frame


def capture_loop(picam, buffer):
    while True:
        frame = picam.capture_array()  # BGR (picamera2 RGB888은 실제로 BGR 순서)
        frame = detect_blue(frame)
        ret, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            buffer.set(jpg.tobytes())


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with buffer.condition:
                        buffer.condition.wait()
                        frame = buffer.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                print(f'[WARN] client disconnected: {e}')
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# --- 카메라 초기화 ---
picam = Picamera2()
picam.configure(picam.create_video_configuration(
    main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
))
picam.start()
time.sleep(1)

buffer = FrameBuffer()
worker = Thread(target=capture_loop, args=(picam, buffer), daemon=True)
worker.start()

try:
    addr = ('0.0.0.0', 8000)
    server = StreamingServer(addr, StreamingHandler)
    print("=" * 40)
    print("  RPi5 Camera + Blue Detection")
    print("=" * 40)
    print("  브라우저 접속: http://192.168.0.23:8000")
    print("  종료:          Ctrl+C")
    print("=" * 40)
    server.serve_forever()
finally:
    picam.stop()
