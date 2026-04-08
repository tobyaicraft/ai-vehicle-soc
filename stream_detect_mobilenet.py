"""
실시간 사물인식 - MobileNet SSD (Google 모델)
20가지 물체 인식 가능
브라우저에서 http://toby.local:5000 으로 확인
"""
from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np

app = Flask(__name__)

# MobileNet SSD - 20가지 물체 인식
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# AI 모델 로드
net = cv2.dnn.readNetFromCaffe(
    "/home/toby/project/models/MobileNetSSD_deploy.prototxt",
    "/home/toby/project/models/MobileNetSSD_deploy.caffemodel"
)

# 카메라 설정
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
camera.configure(config)
camera.start()

def detect_objects():
    while True:
        frame = camera.capture_array()
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                color = COLORS[idx]

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label}: {confidence:.1%}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '<html><body style="background:#111;color:#fff;text-align:center;font-family:sans-serif;"><h1>MobileNet SSD (Google) - 20 Classes</h1><img src="/video" width="640"></body></html>'

@app.route('/video')
def video():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("MobileNet SSD (Google) - 20가지 물체 인식")
    print("PC browser: http://toby.local:5000")
    app.run(host='0.0.0.0', port=5000)
