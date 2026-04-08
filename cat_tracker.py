#!/usr/bin/env python3
"""
EP.07 - Cat Tracker (YOLOv8 ONNX INT8 - Custom Cat Model)
직접 학습한 YOLOv8 커스텀 고양이 모델 (양자화) + 서보모터 추적
Flask 웹 스트리밍으로 실시간 확인 가능

실행: sudo /home/toby/myenv/bin/python cat_tracker.py
웹:  http://<라즈베리파이IP>:5000
"""

import time
import threading
import numpy as np
from picamera2 import Picamera2
import cv2
from flask import Flask, Response
import onnxruntime as ort

# ──────────────────────────────────────────────
# 설정값
# ──────────────────────────────────────────────
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_CENTER_X = FRAME_WIDTH // 2     # 320

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.45

DEAD_ZONE = 60
SERVO_GAIN = 0.06
SERVO_SMOOTH = 0.3
MAX_STEP = 3.0
JITTER_THRESHOLD = 35
LOST_PATIENCE = 15
SERVO_MIN = 0
SERVO_MAX = 180
SERVO_INIT = 90

PWM_PATH = '/sys/class/pwm/pwmchip0/pwm2'
PWM_PERIOD_NS = 20000000

MODEL_PATH = '/home/toby/project/best_int8.onnx'
MODEL_INPUT_SIZE = 320

# ──────────────────────────────────────────────
# 서보 제어 (하드웨어 PWM)
# ──────────────────────────────────────────────
def servo_init():
    try:
        with open('/sys/class/pwm/pwmchip0/export', 'w') as f:
            f.write('2')
    except OSError:
        pass
    time.sleep(0.1)
    with open(f'{PWM_PATH}/period', 'w') as f:
        f.write(str(PWM_PERIOD_NS))
    with open(f'{PWM_PATH}/enable', 'w') as f:
        f.write('1')
    set_servo_angle(SERVO_INIT)


def set_servo_angle(angle):
    angle = max(SERVO_MIN, min(SERVO_MAX, angle))
    duty_ns = int((angle / 180 * 2000000) + 500000)
    with open(f'{PWM_PATH}/duty_cycle', 'w') as f:
        f.write(str(duty_ns))


def servo_cleanup():
    try:
        with open(f'{PWM_PATH}/duty_cycle', 'w') as f:
            f.write('0')
        with open(f'{PWM_PATH}/enable', 'w') as f:
            f.write('0')
    except OSError:
        pass

# ──────────────────────────────────────────────
# YOLOv8 전처리 / 후처리
# ──────────────────────────────────────────────
def preprocess(frame, input_size):
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    dx, dy = (input_size - new_w) // 2, (input_size - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, dx, dy


def postprocess(output, frame_w, frame_h, scale, dx, dy,
                conf_thresh, nms_thresh):
    # output shape: (1, 5, 2100) — 1 class (cat-detector)
    predictions = output[0].T  # (2100, 5)

    boxes_xywh = predictions[:, :4]
    scores = predictions[:, 4]  # 1 class only

    mask = scores > conf_thresh
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]

    if len(boxes_xywh) == 0:
        return []

    # cx,cy,w,h → x1,y1,x2,y2
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # letterbox → 원본 좌표
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - dx) / scale
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - dy) / scale
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - dx) / scale
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - dy) / scale

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, frame_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, frame_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, frame_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, frame_h)

    # NMS
    boxes_for_nms = []
    for b in boxes_xyxy:
        boxes_for_nms.append([int(b[0]), int(b[1]),
                              int(b[2] - b[0]), int(b[3] - b[1])])

    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(),
                               conf_thresh, nms_thresh)

    results = []
    if len(indices) > 0:
        indices = np.array(indices).flatten()
        for idx in indices:
            x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
            results.append((x1, y1, x2, y2, float(scores[idx])))

    return results

# ──────────────────────────────────────────────
# 글로벌 변수
# ──────────────────────────────────────────────
current_angle = SERVO_INIT
target_angle = SERVO_INIT
stable_cx = FRAME_CENTER_X
stable_cy = FRAME_HEIGHT // 2
output_frame = None
frame_lock = threading.Lock()

# ──────────────────────────────────────────────
# 메인 추적 루프
# ──────────────────────────────────────────────
def tracking_loop():
    global current_angle, target_angle, stable_cx, stable_cy, output_frame

    session = ort.InferenceSession(MODEL_PATH,
                                   providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print(f"[INFO] YOLOv8 ONNX INT8 모델 로드 완료 (입력: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE})")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    servo_init()

    print(f"[INFO] Cat Tracker EP.07 시작 - http://0.0.0.0:5000")
    print(f"[INFO] 서보 초기 각도: {SERVO_INIT}도")

    fps_time = time.time()
    lost_count = 0

    try:
        while True:
            frame = picam2.capture_array()
            display = frame.copy()
            h, w = frame.shape[:2]

            blob, scale, dx, dy = preprocess(frame, MODEL_INPUT_SIZE)
            outputs = session.run(None, {input_name: blob})
            detections = postprocess(outputs[0], w, h, scale, dx, dy,
                                     CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

            cat_detected = False
            best_conf = 0
            best_box = None

            for (x1, y1, x2, y2, conf) in detections:
                if conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)
                    cat_detected = True

            if cat_detected:
                lost_count = 0
                x1, y1, x2, y2 = best_box
                raw_cx = (x1 + x2) // 2
                raw_cy = (y1 + y2) // 2

                ddx = abs(raw_cx - stable_cx)
                ddy = abs(raw_cy - stable_cy)
                if ddx > JITTER_THRESHOLD or ddy > JITTER_THRESHOLD:
                    stable_cx = raw_cx
                    stable_cy = raw_cy

                cx = int(stable_cx)
                cy = int(stable_cy)
                error = cx - FRAME_CENTER_X

                if abs(error) > DEAD_ZONE:
                    step = error * SERVO_GAIN
                    step = max(-MAX_STEP, min(MAX_STEP, step))
                    target_angle = target_angle - step
                    target_angle = max(SERVO_MIN, min(SERVO_MAX, target_angle))

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                label = f"CAT {best_conf:.2f}"
                cv2.putText(display, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                err_text = f"err:{error:+d}px  servo:{current_angle:.1f}"
                cv2.putText(display, err_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                lost_count += 1
                if lost_count <= LOST_PATIENCE:
                    cv2.putText(display, f"Tracking... ({lost_count})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    target_angle = current_angle
                    cv2.putText(display, "No cat detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            current_angle = current_angle + (target_angle - current_angle) * SERVO_SMOOTH
            current_angle = max(SERVO_MIN, min(SERVO_MAX, current_angle))
            set_servo_angle(current_angle)

            now = time.time()
            fps = 1.0 / (now - fps_time)
            fps_time = now
            cv2.putText(display, f"FPS:{fps:.1f}", (FRAME_WIDTH - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.line(display, (FRAME_CENTER_X, 0), (FRAME_CENTER_X, FRAME_HEIGHT),
                     (128, 128, 128), 1)
            cv2.line(display, (FRAME_CENTER_X - DEAD_ZONE, 0),
                     (FRAME_CENTER_X - DEAD_ZONE, FRAME_HEIGHT), (100, 100, 100), 1)
            cv2.line(display, (FRAME_CENTER_X + DEAD_ZONE, 0),
                     (FRAME_CENTER_X + DEAD_ZONE, FRAME_HEIGHT), (100, 100, 100), 1)

            bar_x = int((current_angle / 180) * FRAME_WIDTH)
            cv2.rectangle(display, (0, FRAME_HEIGHT - 10),
                          (bar_x, FRAME_HEIGHT), (0, 200, 200), -1)

            with frame_lock:
                output_frame = display.copy()

    except KeyboardInterrupt:
        print("\n[INFO] 종료 중...")
    finally:
        servo_cleanup()
        picam2.stop()

# ──────────────────────────────────────────────
# Flask 웹 스트리밍
# ──────────────────────────────────────────────
app = Flask(__name__)

def generate_frames():
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            _, jpeg = cv2.imencode('.jpg', output_frame)
            frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return '''<html><head><title>Cat Tracker EP.07</title></head>
    <body style="margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh">
    <img src="/video" style="max-width:100%;max-height:100vh">
    </body></html>'''


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
if __name__ == '__main__':
    tracker = threading.Thread(target=tracking_loop, daemon=True)
    tracker.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
