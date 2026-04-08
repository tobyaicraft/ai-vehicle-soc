"""
AI 물체 인식 체험 - MobileNetV2
이미지를 넣으면 AI가 무엇인지 알려줍니다.
"""
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import json

# ============================================
# 1단계: 학습된 AI 모델 불러오기
# ============================================
# MobileNetV2: 이미 1000가지 물체를 학습한 모델
# (ImageNet 데이터셋으로 학습됨)
print("AI 모델을 불러오는 중...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.eval()  # 추론 모드 (학습이 아니라 예측만 할 것)

# 1000가지 물체 이름 목록 불러오기
weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
categories = weights.meta["categories"]

# ============================================
# 2단계: 이미지 전처리 함수
# ============================================
# AI 모델이 이해할 수 있는 형태로 이미지를 변환
# (크기 조정 + 정규화)
preprocess = transforms.Compose([
    transforms.Resize(256),           # 크기 조정
    transforms.CenterCrop(224),       # 224x224로 자르기
    transforms.ToTensor(),            # 숫자 배열로 변환
    transforms.Normalize(             # 정규화 (모델이 학습할 때 사용한 값)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ============================================
# 3단계: 이미지 인식 실행
# ============================================
image_dir = "./image"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

if not image_files:
    print("image 폴더에 이미지가 없습니다!")
else:
    print(f"\n총 {len(image_files)}개의 이미지를 분석합니다.\n")
    print("=" * 60)

    for filename in sorted(image_files):
        filepath = os.path.join(image_dir, filename)

        # 이미지 읽기
        img = Image.open(filepath).convert("RGB")

        # 전처리
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가

        # AI 모델로 예측
        with torch.no_grad():  # 기울기 계산 불필요 (추론만)
            output = model(input_batch)

        # 결과 해석 (상위 3개)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3_prob, top3_idx = torch.topk(probabilities, 3)

        print(f"파일: {filename}")
        print(f"  AI 판단 결과:")
        for i in range(3):
            name = categories[top3_idx[i].item()]
            prob = top3_prob[i].item() * 100
            bar = "#" * int(prob / 2)
            print(f"    {i+1}: {name:20s} {prob:5.1f}% {bar}")
        print()

    print("=" * 60)
    print("완료! AI 모델이 각 이미지에서 무엇을 인식했는지 확인하세요.")
    print()
    print("--- 임베디드 개발자를 위한 핵심 정리 ---")
    print("1. 모델 크기: MobileNetV2 ~3.4MB (MCU에 올릴 수 있는 크기)")
    print("2. 입력: 224x224 RGB 이미지 (= 224*224*3 = 150,528 바이트)")
    print("3. 출력: 1000개 클래스에 대한 확률값")
    print("4. 이 모델을 C로 변환하면 MCU에서도 동작 가능!")
