"""
AI 모델 스펙 비교: MobileNet SSD vs YOLOv5 nano
두 모델의 층 수, 파라미터 수 등을 확인합니다.
"""
import os

# ============================================
# 1. MobileNet SSD 스펙
# ============================================
print("=" * 60)
print("  MobileNet SSD 모델 스펙")
print("=" * 60)

import cv2

net = cv2.dnn.readNetFromCaffe(
    "/home/toby/project/models/MobileNetSSD_deploy.prototxt",
    "/home/toby/project/models/MobileNetSSD_deploy.caffemodel"
)

layers = net.getLayerNames()
print(f"총 층(Layer) 수: {len(layers)}")
print(f"인식 가능 물체: 20가지")

# 모델 파일 크기
caffemodel = "/home/toby/project/models/MobileNetSSD_deploy.caffemodel"
size_mb = os.path.getsize(caffemodel) / (1024 * 1024)
print(f"모델 파일 크기: {size_mb:.1f} MB")

print(f"\n--- 층 목록 (일부) ---")
for i, name in enumerate(layers[:10], 1):
    print(f"  {i}: {name}")
if len(layers) > 10:
    print(f"  ... (총 {len(layers)}개)")

# ============================================
# 2. YOLOv5 nano 스펙
# ============================================
print()
print("=" * 60)
print("  YOLOv5 nano 모델 스펙")
print("=" * 60)

from ultralytics import YOLO

model = YOLO("yolov5nu.pt")
model.info(detailed=False)
print(f"인식 가능 물체: 80가지")

# ============================================
# 3. 비교 요약
# ============================================
print()
print("=" * 60)
print("  비교 요약")
print("=" * 60)
print(f"{'항목':<20} {'MobileNet SSD':<20} {'YOLOv5 nano':<20}")
print("-" * 60)
print(f"{'인식 물체':<20} {'20가지':<20} {'80가지':<20}")
print(f"{'모델 파일 크기':<20} {f'{size_mb:.1f} MB':<20} {'~4 MB':<20}")
print(f"{'용도':<20} {'경량 기기':<20} {'경량 기기':<20}")
print(f"{'정확도':<20} {'보통':<20} {'높음':<20}")
