# === RC Car 공통 설정 ===
# 새 WiFi 추가 시 RPI5_HOSTS 리스트에 추가하면 됩니다

RPI5_HOSTS = [
    "192.168.0.23",     # 집 WiFi
    "toby.local",       # mDNS
    "192.168.43.1",     # 핸드폰 핫스팟 (예시)
]
RPI5_HOST = RPI5_HOSTS[0]  # 기본값

CMD_PORT = 9000
SENSOR_PORT = 9001
CAM_PORT = 8000
