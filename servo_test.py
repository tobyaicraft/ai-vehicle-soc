"""
서보모터 테스트 - GPIO 18 (Pin 12)
서보를 0도 → 90도 → 180도 → 90도 → 0도로 움직입니다.
"""

import time
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory

# 라즈베리파이5는 lgpio 사용
factory = LGPIOFactory()

# GPIO 18에 서보 연결
# min_pulse_width, max_pulse_width는 SG90 서보 기준
servo = Servo(18, pin_factory=factory,
              min_pulse_width=0.5/1000,
              max_pulse_width=2.5/1000)

print("서보모터 테스트 시작!")

try:
    while True:
        print("→ 0도 (최소)")
        servo.min()
        time.sleep(1)

        print("→ 90도 (중앙)")
        servo.mid()
        time.sleep(1)

        print("→ 180도 (최대)")
        servo.max()
        time.sleep(1)

        print("→ 90도 (중앙)")
        servo.mid()
        time.sleep(1)

except KeyboardInterrupt:
    print("\n종료합니다.")
    servo.close()
