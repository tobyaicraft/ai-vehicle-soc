"""
RPi5 카메라 MJPEG 스트리머 + 검출 모드 전환
  - none : 검출 없이 원본 스트리밍
  - blue : 파란색 물체 HSV 검출
  - cat  : YOLOv8n INT8 고양이 검출

모드 전환: HTTP GET /mode/<name>  (예: /mode/cat, /mode/blue, /mode/none)
스트리밍:  http://<RPi_IP>:8000/stream.mjpg
"""
import io
import os
import time
from threading import Condition, Thread, Lock
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
from picamera2 import Picamera2

# --- 설정 ---
FRAME_W = 640
FRAME_H = 480
MODEL_DIR = "/home/toby/project/models"

# MobileNet SSD (Caffe)
SSD_PROTOTXT = f"{MODEL_DIR}/MobileNetSSD_deploy.prototxt"
SSD_MODEL    = f"{MODEL_DIR}/MobileNetSSD_deploy.caffemodel"
SSD_CONF = 0.5
SSD_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

# YOLOv8n ONNX INT8 (COCO pretrained)
YOLO_ONNX_PATH = f"{MODEL_DIR}/yolov8n_int8.onnx"

# 커스텀 고양이 ONNX INT8 (87장 학습, mAP50=0.992)
CAT_ONNX_PATH = f"{MODEL_DIR}/best_int8.onnx"
CAT_ONNX_CONF = 0.2
CAT_ONNX_CLASS_ID = 0  # 커스텀 모델은 클래스 0 = cat
YOLO_IMGSZ = 320
YOLO_CONF = 0.2
CAT_CLASS_ID = 15  # COCO cat

# YOLOv8n TFLite INT8
YOLO_TFLITE_PATH = f"{MODEL_DIR}/yolov8n_int8.tflite"

# 파랑색 HSV 범위
BLUE_LOW = np.array([100, 80, 50])
BLUE_HIGH = np.array([130, 255, 255])
MIN_AREA = 500

# --- Detection Mode ---
# "none", "blue", "ssd", "cat_custom", "yolo_onnx", "yolo_tflite"
detect_mode = "none"
detect_lock = Lock()

# --- 최신 검출 결과 (PC 폴링용) ---
last_detection = {"box": None, "conf": 0.0, "timestamp": 0.0}

# --- 모델 인스턴스 ---
ssd_net = None
cat_onnx_session = None
cat_onnx_input = None
yolo_onnx_session = None
yolo_onnx_input = None
yolo_tflite_interp = None
yolo_tflite_input = None
yolo_tflite_output = None


def load_models():
    global ssd_net, cat_onnx_session, cat_onnx_input
    global yolo_onnx_session, yolo_onnx_input
    global yolo_tflite_interp, yolo_tflite_input, yolo_tflite_output

    # 1. MobileNet SSD
    if os.path.exists(SSD_PROTOTXT) and os.path.exists(SSD_MODEL):
        try:
            ssd_net = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT, SSD_MODEL)
            print("[MODEL] MobileNet SSD loaded")
        except Exception as e:
            print(f"[MODEL] MobileNet SSD failed: {e}")

    # 2. 커스텀 고양이 ONNX INT8
    if os.path.exists(CAT_ONNX_PATH):
        try:
            import onnxruntime as ort
            cat_onnx_session = ort.InferenceSession(CAT_ONNX_PATH,
                                                     providers=["CPUExecutionProvider"])
            cat_onnx_input = cat_onnx_session.get_inputs()[0].name
            print("[MODEL] Custom Cat ONNX INT8 loaded")
        except Exception as e:
            print(f"[MODEL] Custom Cat ONNX failed: {e}")

    # 3. YOLOv8n ONNX INT8
    if os.path.exists(YOLO_ONNX_PATH):
        try:
            import onnxruntime as ort
            yolo_onnx_session = ort.InferenceSession(YOLO_ONNX_PATH,
                                                      providers=["CPUExecutionProvider"])
            yolo_onnx_input = yolo_onnx_session.get_inputs()[0].name
            print("[MODEL] YOLOv8n ONNX INT8 loaded")
        except Exception as e:
            print(f"[MODEL] YOLOv8n ONNX failed: {e}")

    # 3. YOLOv8n TFLite INT8
    if os.path.exists(YOLO_TFLITE_PATH):
        try:
            from ai_edge_litert.interpreter import Interpreter
            yolo_tflite_interp = Interpreter(model_path=YOLO_TFLITE_PATH)
            yolo_tflite_interp.allocate_tensors()
            yolo_tflite_input = yolo_tflite_interp.get_input_details()
            yolo_tflite_output = yolo_tflite_interp.get_output_details()
            print("[MODEL] YOLOv8n TFLite INT8 loaded")
        except Exception as e:
            print(f"[MODEL] YOLOv8n TFLite failed: {e}")


def _yolo_postprocess(preds, frame_h, frame_w, model_tag):
    """YOLOv8 공통 후처리: preds (84, N) → 고양이 검출 결과"""
    if preds.shape[0] == 84:
        preds = preds.T

    sx = frame_w / YOLO_IMGSZ
    sy = frame_h / YOLO_IMGSZ
    best_conf = 0
    best_box = None

    for det in preds:
        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        scores = det[4:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])
        if class_id != CAT_CLASS_ID or conf < YOLO_CONF:
            continue
        if conf > best_conf:
            best_conf = conf
            best_box = (int((cx - bw/2)*sx), int((cy - bh/2)*sy),
                        int((cx + bw/2)*sx), int((cy + bh/2)*sy))

    return best_box, best_conf


def detect_cat_custom(frame):
    """커스텀 고양이 ONNX INT8 검출 (87장 학습 모델)"""
    if cat_onnx_session is None:
        cv2.putText(frame, "Cat model not loaded", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    h, w = frame.shape[:2]
    img = cv2.resize(frame, (YOLO_IMGSZ, YOLO_IMGSZ))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    outputs = cat_onnx_session.run(None, {cat_onnx_input: img})
    preds = outputs[0][0]
    if preds.shape[0] == 5:  # 커스텀 1클래스: (5, N)
        preds = preds.T

    sx = w / YOLO_IMGSZ
    sy = h / YOLO_IMGSZ
    best_conf = 0
    best_box = None

    for det in preds:
        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        conf = float(det[4]) if preds.shape[1] == 5 else float(det[4:].max())
        if conf < CAT_ONNX_CONF:
            continue
        if conf > best_conf:
            best_conf = conf
            best_box = (int((cx - bw/2)*sx), int((cy - bh/2)*sy),
                        int((cx + bw/2)*sx), int((cy + bh/2)*sy))

    # 검출 결과를 전역에 저장 (PC 폴링용)
    with detect_lock:
        if best_box:
            last_detection["box"] = list(best_box)
            last_detection["conf"] = best_conf
        else:
            last_detection["box"] = None
            last_detection["conf"] = 0.0
        last_detection["timestamp"] = time.time()

    if best_box:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 6, (0, 255, 0), -1)
        cv2.putText(frame, f"CAT {best_conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No cat", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 1)
    cv2.putText(frame, "[CAT CUSTOM]", (w - 190, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def detect_ssd(frame):
    """MobileNet SSD 검출"""
    if ssd_net is None:
        cv2.putText(frame, "SSD not loaded", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                  0.007843, (300, 300), 127.5)
    ssd_net.setInput(blob)
    detections = ssd_net.forward()

    found = False
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])
        if confidence < SSD_CONF or class_id >= len(SSD_CLASSES):
            continue
        label = SSD_CLASSES[class_id]

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        color = (0, 255, 0) if label == "cat" else (0, 180, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {confidence:.0%}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if label == "cat":
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 6, color, -1)
        found = True

    if not found:
        cv2.putText(frame, "No object", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 1)
    cv2.putText(frame, "[SSD]", (w - 80, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def detect_yolo_onnx(frame):
    """YOLOv8n ONNX INT8 고양이 검출"""
    if yolo_onnx_session is None:
        cv2.putText(frame, "ONNX not loaded", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    h, w = frame.shape[:2]
    img = cv2.resize(frame, (YOLO_IMGSZ, YOLO_IMGSZ))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    outputs = yolo_onnx_session.run(None, {yolo_onnx_input: img})
    box, conf = _yolo_postprocess(outputs[0][0], h, w, "ONNX")

    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 6, (0, 255, 0), -1)
        cv2.putText(frame, f"CAT {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No cat", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 1)
    cv2.putText(frame, "[YOLO ONNX]", (w - 160, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def detect_yolo_tflite(frame):
    """YOLOv8n TFLite INT8 고양이 검출"""
    if yolo_tflite_interp is None:
        cv2.putText(frame, "TFLite not loaded", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    h, w = frame.shape[:2]
    img = cv2.resize(frame, (YOLO_IMGSZ, YOLO_IMGSZ))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)

    yolo_tflite_interp.set_tensor(yolo_tflite_input[0]['index'], img)
    yolo_tflite_interp.invoke()
    output = yolo_tflite_interp.get_tensor(yolo_tflite_output[0]['index'])
    box, conf = _yolo_postprocess(output[0], h, w, "TFLite")

    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 6, (0, 255, 0), -1)
        cv2.putText(frame, f"CAT {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No cat", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 1)
    cv2.putText(frame, "[YOLO TFLite]", (w - 180, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return frame


def detect_blue(frame):
    """파랑 물체 검출 + 오버레이 그리기."""
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

    cv2.line(frame, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (255, 255, 255), 1)

    cv2.putText(frame, "[BLUE]", (FRAME_W - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    return frame


# --- Frame Buffer ---
class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def set(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


def capture_loop(picam, buffer):
    while True:
        frame = picam.capture_array()

        with detect_lock:
            mode = detect_mode

        if mode == "cat_custom":
            frame = detect_cat_custom(frame)
        elif mode == "ssd":
            frame = detect_ssd(frame)
        elif mode == "yolo_onnx":
            frame = detect_yolo_onnx(frame)
        elif mode == "yolo_tflite":
            frame = detect_yolo_tflite(frame)
        elif mode == "blue":
            frame = detect_blue(frame)
        # "none" → 원본 그대로

        ret, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            buffer.set(jpg.tobytes())


PAGE = """\
<!DOCTYPE html>
<html>
<head><title>RPi5 Camera</title></head>
<body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
<h2>RPi5 Camera Stream</h2>
<img src="stream.mjpg" width="640" height="480" /><br>
<a href="/mode/none" style="color:#888;margin:10px;">None</a>
<a href="/mode/blue" style="color:#44f;margin:10px;">Blue</a>
<a href="/mode/ssd" style="color:#0f0;margin:10px;">MobileNet SSD</a>
<a href="/mode/cat_custom" style="color:#f0f;margin:10px;">Cat Custom</a>
<a href="/mode/yolo_onnx" style="color:#ff0;margin:10px;">YOLO ONNX</a>
<a href="/mode/yolo_tflite" style="color:#f80;margin:10px;">YOLO TFLite</a>
</body>
</html>
"""


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global detect_mode

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
        elif self.path.startswith('/mode/'):
            new_mode = self.path.split('/mode/')[-1].strip('/')
            if new_mode in ("none", "blue", "ssd", "cat_custom", "yolo_onnx", "yolo_tflite"):
                with detect_lock:
                    detect_mode = new_mode
                msg = f"Mode changed to: {new_mode}"
                print(f"[MODE] {new_mode}")
            else:
                msg = f"Unknown mode: {new_mode}"
            content = msg.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/detect':
            import json
            with detect_lock:
                data = {
                    "box": last_detection["box"],
                    "conf": last_detection["conf"],
                    "frame_w": FRAME_W,
                    "frame_h": FRAME_H,
                    "timestamp": last_detection["timestamp"],
                }
            content = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(content))
            self.send_header('Access-Control-Allow-Origin', '*')
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


# --- 시작 ---
load_models()

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
    print("  RPi5 Camera + Detection")
    print("=" * 40)
    print(f"  Stream : http://0.0.0.0:8000/stream.mjpg")
    print(f"  Mode   : none | blue | ssd | yolo_onnx | yolo_tflite")
    print(f"  Current: {detect_mode}")
    print("=" * 40)
    server.serve_forever()
finally:
    picam.stop()
