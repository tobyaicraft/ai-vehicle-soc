# 음성 제어 RC카 프로젝트 컨텍스트

## 프로젝트 개요
키보드 제어(W/A/S/D/Space → TCP → RPi5 → UART → TC237) 프로젝트를 기반으로,
**입력 방식을 키보드에서 음성으로 교체**하는 프로젝트입니다.

---

## PC 사양
| 항목 | 스펙 |
|------|------|
| CPU | i7-13700K (13세대, 랩터레이크) |
| RAM | DDR4 32GB |
| 메인보드 | MSI MAG B660M (PCIe 4.0) |
| SSD | 삼성 980 M.2 NVMe 1TB |
| GPU | RTX 3060 12GB |
| OS | Windows |

---

## 전체 아키텍처

```
마이크
  ↓
sounddevice (소리 → 숫자 배열 raw 데이터)
  ↓
faster-whisper (숫자 배열 → 텍스트 "전진")
  ↓
명령어 파싱 ("전진" → 'W')  ← PC에서 처리
  ↓
TCP 소켓 (Wi-Fi, port 9000)
  ↓
Raspberry Pi 5 (192.168.30.34) ← 코드 변경 없음
  ↓
UART (/dev/ttyAMA0, 115200 8N1)
  ↓
TC237 (모터 제어)
```

---

## 핵심 설계 결정사항

### 1. 파싱 위치: PC에서 처리
- PC에서 "전진" → 'W' 변환 후 TCP 전송
- RPi5는 키보드 프로젝트와 **동일한 코드 그대로 사용**
- TC237도 변경 없음

### 2. STT 엔진: faster-whisper
- 로컬 실행 (인터넷 불필요)
- GPU(CUDA) 사용으로 빠른 처리
- medium 모델: 한국어 인식 정확도 충분

### 3. GPU 사용 명시 필요
- Python은 기본적으로 CPU 사용
- 모델 로딩 시 반드시 `device="cuda"` 명시해야 GPU 사용
- `compute_type="float16"`: RTX 3060 최적 연산 방식

---

## faster-whisper 핵심 설정

```python
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda", compute_type="float16")

segments, info = model.transcribe(audio)
for segment in segments:
    print(segment.text)  # "전진"
```

### output 구조
```python
segment.text      # "전진"       ← 우리가 사용하는 값
segment.start     # 0.0          # 말 시작 시간(초)
segment.end       # 0.8          # 말 끝 시간(초)
segment.language  # "ko"         # 언어
```

---

## 명령어 파싱 테이블

| 음성 | 전송 값 | 동작 |
|------|---------|------|
| "전진" | 'W' | 전진 |
| "후진" | 'S' | 후진 |
| "좌회전" | 'A' | 좌회전 |
| "우회전" | 'D' | 우회전 |
| "정지" | ' ' (Space) | 정지 |

---

## 필요한 라이브러리 설치

```bash
pip install sounddevice
pip install faster-whisper
pip install numpy
```

### CUDA 요구사항
- NVIDIA CUDA 드라이버 설치 필요
- RTX 3060이면 기존 게임 드라이버로 이미 설치되어 있을 가능성 높음
- 확인 방법: `nvidia-smi` 명령어

---

## 예상 지연시간

```
말 끝남 → VAD 감지 → Whisper 처리 → TCP 전송 → 차량 반응
  0.0초     0.1초       0.2~0.3초      ~0ms
                총 약 0.3~0.4초
```

---

## 변경 범위 요약

| 위치 | 키보드 프로젝트 | 음성 프로젝트 |
|------|--------------|-------------|
| PC 코드 | pynput 키 감지 | faster-whisper 음성 감지 |
| RPi5 코드 | 변경 없음 ✅ | 변경 없음 ✅ |
| TC237 코드 | 변경 없음 ✅ | 변경 없음 ✅ |

**PC 파이썬 코드 1개만 새로 작성하면 됩니다.**

---

## Claude Code 작업 요청사항

1. `sounddevice`로 마이크 실시간 음성 감지 (VAD 포함)
2. `faster-whisper` medium 모델, device="cuda", compute_type="float16" 로 STT
3. 텍스트 파싱 → W/S/A/D/Space 변환
4. 기존 TCP 소켓 코드와 연결 (RPi5: 192.168.30.34:9000)
5. 단계별 동작 확인 가능하도록 로그 출력 포함
