from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# YOLOv5 nano 모델 로드 (80가지 물체 인식)
# 처음 실행 시 자동으로 모델 다운로드됨 (~4MB)
model = YOLO("yolov5nu.pt")

# 카메라 설정
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
camera.configure(config)
camera.start()

def detect_objects():
    while True:
        frame = camera.capture_array()

        # AI 모델로 물체 인식 (핵심 2줄)
        results = model(frame, verbose=False)

        # 결과를 프레임에 그리기 (박스 + 이름 + 확률)
        annotated = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '<html><body style="background:#111;color:#fff;text-align:center;font-family:sans-serif;"><h1>Toby Object Detection</h1><img src="/video" width="640"></body></html>'

@app.route('/video')
def video():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting... PC browser: http://toby.local:5000")
    app.run(host='0.0.0.0', port=5000)
