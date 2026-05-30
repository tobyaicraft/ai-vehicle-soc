"""
RC Car Voice Client - Claude AI 음성 제어
마이크 음성 입력 → Claude API (tool_use) → TTS 음성 출력 + RC카 명령 전송

Usage:
    python voice_client.py                          # 기본: 192.168.0.23
    python voice_client.py 192.168.0.23             # IP 지정
    python voice_client.py 192.168.0.23 9000 9001   # IP/포트 지정

필요 패키지:
    pip install anthropic SpeechRecognition sounddevice gTTS

환경변수:
    ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import sys
import socket
import threading
import queue
import time
import tempfile
import subprocess
import io
import wave
import anthropic
import speech_recognition as sr
import sounddevice as sd
from gtts import gTTS
import numpy as np

# --- Configuration ---
from config import RPI5_HOST as DEFAULT_HOST, CMD_PORT as DEFAULT_CMD_PORT, SENSOR_PORT as DEFAULT_SENSOR_PORT
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_HISTORY = 20  # 대화 히스토리 최대 메시지 수

# --- 센서 변환 함수 (sensor_monitor.py 기반) ---
ADC_MAX = 4095
VAREF = 5.0


def adc_to_voltage(adc_val):
    return adc_val / ADC_MAX * VAREF


def voltage_to_distance_cm(voltage):
    """GP2Y0A21YK0F approximate conversion."""
    if voltage < 0.3:
        return 80.0
    if voltage > 3.2:
        return 10.0
    try:
        dist = 29.988 * pow(voltage, -1.173)
    except (ValueError, ZeroDivisionError):
        return 80.0
    return max(10.0, min(80.0, dist))


def play_mp3_async(filepath):
    """Windows WPF MediaPlayer로 MP3 재생 (백그라운드 프로세스, 중단 가능)"""
    ps_script = f'''
Add-Type -AssemblyName presentationCore
$player = New-Object System.Windows.Media.MediaPlayer
$player.Open([Uri]"file:///{filepath.replace(os.sep, '/')}")
Start-Sleep -Milliseconds 500
$player.Play()
while($player.NaturalDuration.HasTimeSpan -eq $false) {{ Start-Sleep -Milliseconds 100 }}
$dur = $player.NaturalDuration.TimeSpan.TotalMilliseconds
Start-Sleep -Milliseconds $dur
$player.Close()
'''
    return subprocess.Popen(
        ["powershell", "-c", ps_script],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# --- Claude API 도구 정의 ---
TOOLS = [
    {
        "name": "move_forward",
        "description": "RC카를 전진시킵니다. 장애물 감지 시 자동으로 멈춥니다. 전방 초음파 거리가 20cm 미만이면 사용하지 마세요.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "move_backward",
        "description": "RC카를 후진시킵니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": "후진 시간 (초). 기본값 1.0, 최대 5.0",
                }
            },
            "required": [],
        },
    },
    {
        "name": "turn_left",
        "description": "RC카를 제자리에서 왼쪽으로 회전합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": "회전 시간 (초). 기본값 0.5, 최대 3.0",
                }
            },
            "required": [],
        },
    },
    {
        "name": "turn_right",
        "description": "RC카를 제자리에서 오른쪽으로 회전합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": "회전 시간 (초). 기본값 0.5, 최대 3.0",
                }
            },
            "required": [],
        },
    },
    {
        "name": "stop",
        "description": "RC카를 즉시 정지시킵니다.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "avoid_obstacle",
        "description": "장애물을 실시간 센서 기반으로 자동 회피합니다. 좌우 IR 센서를 확인하여 넓은 쪽으로 회전한 뒤 전진합니다.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "servo_left",
        "description": "카메라 서보를 왼쪽으로 회전합니다 (10도 단위).",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "회전 스텝 수. 기본 1 (=10도)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "servo_right",
        "description": "카메라 서보를 오른쪽으로 회전합니다 (10도 단위).",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "회전 스텝 수. 기본 1 (=10도)",
                }
            },
            "required": [],
        },
    },
]

SYSTEM_PROMPT = """당신은 RC카에 탑재된 AI 어시스턴트입니다. 사용자와 한국어로 대화하며 차량을 제어합니다.

## 당신의 역할
- 사용자의 음성 명령을 이해하고 RC카를 제어합니다
- 센서 데이터를 분석하여 안전한 주행을 돕습니다
- 친근하고 자연스러운 한국어로 대화합니다
- 응답은 반드시 1문장으로 합니다 (TTS로 읽히므로 최대한 짧게)

## 센서 배치
- 초음파 센서: 차량 정면 중앙, 전방 장애물까지 거리(cm) 측정
- IR 왼쪽: 차량 전면 왼쪽 45도 대각선 방향, 왼쪽 앞 공간 거리(cm) 측정
- IR 오른쪽: 차량 전면 오른쪽 45도 대각선 방향, 오른쪽 앞 공간 거리(cm) 측정
- 4WD 차량이므로 turn_left/turn_right는 제자리 스핀턴(회전)

## 안전 규칙 (반드시 준수)
1. 전방 초음파 거리가 20cm 미만이면 전진 금지. 사용자에게 장애물을 경고하고 좌/우 IR 값을 보고하세요.
2. 좌/우 IR 거리가 15cm 미만이면 해당 방향 회전 주의. 경고 후 진행하세요.
3. 센서 데이터가 없거나 연결이 끊긴 경우, 이동 명령 전에 사용자에게 알리세요.
4. 사용자가 위험한 명령을 해도 안전을 우선시하세요.

## 장애물 회피
사용자가 "피해서 가" 등 회피 명령을 하면 avoid_obstacle 도구를 사용하세요.
이 도구는 실시간으로 센서를 읽으며 자동으로 회피 동작을 수행합니다.

## 사용 가능한 명령
- move_forward: 전진 (장애물 감지 시 자동 정지, 계속 전진)
- move_backward: 후진 (duration초 동안 실행 후 자동 정지)
- turn_left: 제자리 좌회전 (duration초 동안 실행 후 자동 정지)
- turn_right: 제자리 우회전 (duration초 동안 실행 후 자동 정지)
- stop: 즉시 정지
- avoid_obstacle: 실시간 센서 기반 장애물 자동 회피 (회전 + 전진)
- servo_left: 카메라 서보 왼쪽 (10도 단위)
- servo_right: 카메라 서보 오른쪽 (10도 단위)

## 대화 스타일
- 자연스러운 한국어 존댓말
- 이모지 사용 금지 (TTS가 읽을 수 없음)
- 숫자와 단위는 명확히 (예: "전방 35센티미터")

## 현재 센서 상태
{sensor_context}"""


# ============================================================
# SensorReceiver: TCP:9001에서 센서 데이터 수신
# ============================================================
class SensorReceiver:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        self.thread = None
        # 최신 센서값
        self.ir_left_adc = 0
        self.ir_right_adc = 0
        self.ultrasonic_cm = 0
        self.connected = False
        self._lock = threading.Lock()
        # 긴급 정지
        self._emergency_stop_cb = None
        self._emergency_active = False
        self.EMERGENCY_DIST = 20  # cm 미만이면 긴급 정지
        self.emergency_queue = queue.Queue()  # 긴급 이벤트 → VoiceAssistant에 전달

    def set_emergency_callback(self, callback):
        """긴급 정지 시 호출할 콜백 (VehicleController.send_command(b'S'))"""
        self._emergency_stop_cb = callback

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

    def _receive_loop(self):
        buf = ""
        while self.running:
            # 연결 시도
            if not self.connected:
                try:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.settimeout(3.0)
                    self.sock.connect((self.host, self.port))
                    self.sock.settimeout(5.0)
                    self.connected = True
                    print(f"[Sensor] 연결됨: {self.host}:{self.port}")
                except (ConnectionRefusedError, OSError, TimeoutError):
                    self.connected = False
                    time.sleep(3)
                    continue

            # 데이터 수신
            try:
                data = self.sock.recv(1024)
                if not data:
                    self.connected = False
                    continue
                buf += data.decode("utf-8", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    self._parse_line(line.strip())
            except (socket.timeout, OSError):
                self.connected = False

    def _parse_line(self, line):
        """'L:xxxx,R:xxxx,U:xxx' 형식 파싱"""
        if not line or not line.startswith("L:"):
            return
        try:
            parts = {}
            for token in line.split(","):
                key, val = token.split(":")
                parts[key.strip()] = int(val.strip())
            with self._lock:
                self.ir_left_adc = parts.get("L", 0)
                self.ir_right_adc = parts.get("R", 0)
                self.ultrasonic_cm = parts.get("U", 0)
                # 긴급 정지 체크
                if self.ultrasonic_cm < self.EMERGENCY_DIST and self._emergency_stop_cb:
                    if not self._emergency_active:
                        self._emergency_active = True
                        dist = self.ultrasonic_cm
                        print(f"\n[긴급 정지] 전방 {dist}cm 장애물 감지!")
                        self._emergency_stop_cb()
                        # VoiceAssistant에 긴급 이벤트 전달
                        self.emergency_queue.put(dist)
                elif self.ultrasonic_cm >= self.EMERGENCY_DIST:
                    self._emergency_active = False
        except (ValueError, KeyError):
            pass

    def get_context_string(self):
        """Claude 시스템 프롬프트에 삽입할 센서 상태 문자열"""
        if not self.connected:
            return "센서 연결 안 됨 (RPi5 미연결)"

        with self._lock:
            l_adc = self.ir_left_adc
            r_adc = self.ir_right_adc
            u_cm = self.ultrasonic_cm

        l_v = adc_to_voltage(l_adc)
        r_v = adc_to_voltage(r_adc)
        l_dist = voltage_to_distance_cm(l_v)
        r_dist = voltage_to_distance_cm(r_v)

        return (
            f"- 전방 초음파: {u_cm}cm\n"
            f"- 왼쪽 IR: 약 {l_dist:.0f}cm (ADC {l_adc})\n"
            f"- 오른쪽 IR: 약 {r_dist:.0f}cm (ADC {r_adc})"
        )

    def get_distances(self):
        """(전방cm, 좌IR cm, 우IR cm) 반환"""
        with self._lock:
            u_cm = self.ultrasonic_cm
            l_adc = self.ir_left_adc
            r_adc = self.ir_right_adc
        l_dist = voltage_to_distance_cm(adc_to_voltage(l_adc))
        r_dist = voltage_to_distance_cm(adc_to_voltage(r_adc))
        return u_cm, l_dist, r_dist


# ============================================================
# VehicleController: TCP:9000으로 명령 전송
# ============================================================
class VehicleController:
    CMD_MAP = {
        "move_forward": b"F",
        "move_backward": b"B",
        "turn_left": b"L",
        "turn_right": b"R",
        "stop": b"S",
        "servo_left": b"U",
        "servo_right": b"I",
    }

    def __init__(self, host, port, sensor=None):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.sensor = sensor

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(3)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(None)
            self.connected = True
            print(f"[Vehicle] 연결됨: {self.host}:{self.port}")
            return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            print(f"[Vehicle] 연결 실패: {self.host}:{self.port} (RPi5 미연결 - 대화만 가능)")
            self.connected = False
            return False

    def send_command(self, cmd_byte):
        if not self.connected:
            print("[Vehicle] 미연결 - 명령 무시")
            return False
        try:
            self.sock.sendall(cmd_byte)
            print(f"  [TX] {cmd_byte.decode()}")
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[Vehicle] 연결 끊김")
            self.connected = False
            return False

    def execute_tool_call(self, tool_name, tool_input):
        """Claude tool_use 결과를 실행하고 결과 문자열 반환"""
        if not self.connected:
            return "차량 미연결 상태"

        # 장애물 자동 회피
        if tool_name == "avoid_obstacle":
            return self._avoid_obstacle()

        cmd_byte = self.CMD_MAP.get(tool_name)
        if not cmd_byte:
            return f"알 수 없는 명령: {tool_name}"

        # 서보 명령: steps만큼 반복
        if tool_name in ("servo_left", "servo_right"):
            steps = tool_input.get("steps", 1)
            steps = max(1, min(9, steps))
            for _ in range(steps):
                self.send_command(cmd_byte)
                time.sleep(0.1)
            return f"서보 {'왼쪽' if tool_name == 'servo_left' else '오른쪽'} {steps * 10}도 회전"

        # 정지 명령
        if tool_name == "stop":
            self.send_command(cmd_byte)
            return "정지 완료"

        # 전진: 장애물 감지 전까지 계속 전진 (긴급 정지가 알아서 멈춤)
        if tool_name == "move_forward":
            self.send_command(b"F")
            # 긴급 정지가 걸릴 때까지 대기 (최대 30초)
            for _ in range(300):
                time.sleep(0.1)
                if self.sensor and self.sensor._emergency_active:
                    front, left, right = self.sensor.get_distances()
                    return f"전진 중 전방 {front}cm 장애물 감지, 긴급 정지"
            self.send_command(b"S")
            return "전진 30초 경과, 자동 정지"

        # 이동 명령 (후진, 회전): duration 동안 실행 후 정지
        duration = tool_input.get("duration", 1.0)
        duration = max(0.3, min(5.0, duration))
        self.send_command(cmd_byte)
        time.sleep(duration)
        self.send_command(b"S")

        direction_names = {
            "move_backward": "후진",
            "turn_left": "좌회전",
            "turn_right": "우회전",
        }
        print(f"  [Auto Stop] {duration}초 경과")
        return f"{direction_names[tool_name]} {duration}초 실행 완료"

    def _avoid_obstacle(self):
        """실시간 센서 기반 장애물 회피 루틴"""
        if not self.sensor:
            return "센서 미연결"

        front, left, right = self.sensor.get_distances()
        print(f"  [회피] 센서: 전방={front}cm, 좌={left:.0f}cm, 우={right:.0f}cm")

        # 1단계: 넓은 쪽으로 제자리 회전
        if left >= right:
            direction = "왼쪽"
            turn_cmd = b"L"
        else:
            direction = "오른쪽"
            turn_cmd = b"R"

        # 반대 방향 명령
        reverse_cmd = b"R" if turn_cmd == b"L" else b"L"
        reverse_dir = "오른쪽" if direction == "왼쪽" else "왼쪽"

        # 1단계: 넓은 쪽으로 회전 (전방 확보될 때까지, 최대 1.5초)
        print(f"  [회피] {direction}으로 회전")
        self.send_command(turn_cmd)
        for _ in range(15):
            time.sleep(0.1)
            front, _, _ = self.sensor.get_distances()
            if front >= 30:
                break
        self.send_command(b"S")
        time.sleep(0.2)

        # 2단계: 전진 (2초 또는 장애물 감지까지)
        print(f"  [회피] 전진")
        self.sensor._emergency_active = False
        self.send_command(b"F")
        for _ in range(20):
            time.sleep(0.1)
            if self.sensor._emergency_active:
                return f"{direction}으로 회피 중 새 장애물 감지 정지"
        self.send_command(b"S")
        time.sleep(0.2)

        # 3단계: 반대 방향으로 회전 (원래 방향 복귀, 1.5초)
        print(f"  [회피] {reverse_dir}으로 복귀 회전")
        self.send_command(reverse_cmd)
        for _ in range(15):
            time.sleep(0.1)
        self.send_command(b"S")
        time.sleep(0.2)

        # 4단계: 다시 전진 (장애물 감지 전까지)
        print(f"  [회피] 전진 재개")
        self.sensor._emergency_active = False
        self.send_command(b"F")
        for _ in range(300):
            time.sleep(0.1)
            if self.sensor._emergency_active:
                front, _, _ = self.sensor.get_distances()
                return f"회피 완료 후 전진, 전방 {front}cm 장애물 감지 정지"
        self.send_command(b"S")
        return f"회피 완료 후 전진 완료"

    def close(self):
        if self.connected:
            self.send_command(b"S")
        if self.sock:
            self.sock.close()


# ============================================================
# VoiceAssistant: STT → Claude API → TTS → 명령 실행
# ============================================================
class VoiceAssistant:
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.02   # RMS 기준 무음 판정
    SILENCE_DURATION = 1.5     # 이 시간(초) 동안 무음이면 녹음 종료
    MAX_RECORD_SEC = 15        # 최대 녹음 시간

    def __init__(self, sensor, vehicle):
        self.sensor = sensor
        self.vehicle = vehicle
        self.client = anthropic.Anthropic()
        self.history = []
        self.recognizer = sr.Recognizer()
        # TTS 임시파일
        self._tts_file = os.path.join(tempfile.gettempdir(), "rc_tts.mp3")
        print("[STT] sounddevice 마이크 준비 완료")

    def _record_until_silence(self):
        """sounddevice로 녹음, 무음 감지 시 자동 종료. WAV bytes 반환."""
        chunk_size = 1024
        chunks = []
        silent_chunks = 0
        max_silent = int(self.SILENCE_DURATION * self.SAMPLE_RATE / chunk_size)
        max_chunks = int(self.MAX_RECORD_SEC * self.SAMPLE_RATE / chunk_size)
        started = False

        stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE, channels=1,
            dtype="int16", blocksize=chunk_size,
        )
        stream.start()
        try:
            for _ in range(max_chunks):
                data, _ = stream.read(chunk_size)
                rms = np.sqrt(np.mean(data.astype(np.float32) ** 2)) / 32768.0
                if rms > self.SILENCE_THRESHOLD:
                    started = True
                    silent_chunks = 0
                    chunks.append(data.copy())
                elif started:
                    silent_chunks += 1
                    chunks.append(data.copy())
                    if silent_chunks >= max_silent:
                        break
        finally:
            stream.stop()
            stream.close()

        if not chunks:
            return None

        # WAV bytes로 변환
        audio_data = np.concatenate(chunks)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()

    def listen(self):
        """마이크 입력을 텍스트로 변환 (sounddevice + Google STT)"""
        print("\n>>> 듣고 있습니다... (말씀하세요)")
        wav_bytes = self._record_until_silence()
        if wav_bytes is None:
            return None

        # sr.AudioData로 변환하여 Google STT 호출
        audio = sr.AudioData(
            wav_bytes[44:],  # WAV 헤더(44bytes) 제거 → raw PCM
            self.SAMPLE_RATE, 2,
        )
        try:
            text = self.recognizer.recognize_google(audio, language="ko-KR")
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[STT Error] {e}")
            return None

    def think(self, user_text):
        """Claude API 호출 → 응답 텍스트 반환"""
        # 센서 컨텍스트로 시스템 프롬프트 구성
        sensor_ctx = self.sensor.get_context_string()
        system = SYSTEM_PROMPT.format(sensor_context=sensor_ctx)

        # 히스토리에 사용자 메시지 추가
        self.history.append({"role": "user", "content": user_text})

        # 히스토리 길이 제한
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        # API 호출
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=100,
            system=system,
            tools=TOOLS,
            messages=self.history,
        )

        # 응답 파싱
        response_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                response_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {"id": block.id, "name": block.name, "input": block.input}
                )

        # 히스토리에 어시스턴트 메시지 추가
        self.history.append({"role": "assistant", "content": response.content})

        # tool_result 처리 (tool_use가 있으면 결과를 보내고 후속 응답 받기)
        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                result = self.vehicle.execute_tool_call(tc["name"], tc["input"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    }
                )
                print(f"  [Tool] {tc['name']} → {result}")

            # tool_result를 히스토리에 추가하고 후속 응답 받기
            self.history.append({"role": "user", "content": tool_results})

            followup = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=system,
                tools=TOOLS,
                messages=self.history,
            )

            followup_text = ""
            for block in followup.content:
                if block.type == "text":
                    followup_text += block.text

            self.history.append({"role": "assistant", "content": followup.content})

            # 후속 응답이 있으면 그걸 사용, 없으면 원래 응답 사용
            if followup_text:
                response_text = followup_text

        return response_text

    def speak(self, text):
        """텍스트를 음성으로 출력 (백그라운드 재생, 중단 가능)"""
        if not text:
            return
        try:
            tts = gTTS(text, lang="ko")
            tts.save(self._tts_file)
            self._tts_proc = play_mp3_async(self._tts_file)
        except Exception as e:
            print(f"[TTS Error] {e}")
            self._tts_proc = None

    def _stop_tts(self):
        """TTS 재생 중이면 즉시 중단"""
        if self._tts_proc and self._tts_proc.poll() is None:
            self._tts_proc.terminate()
            self._tts_proc = None

    def _wait_tts(self):
        """TTS 재생 완료 대기"""
        if self._tts_proc:
            self._tts_proc.wait(timeout=30)
            self._tts_proc = None

    def run(self):
        """메인 대화 루프"""
        self._tts_proc = None
        print("=" * 50)
        print("  RC Car Voice Controller - Claude AI")
        print("=" * 50)
        print(f"  AI 모델: {CLAUDE_MODEL}")
        print(f"  차량: {self.vehicle.host}:{self.vehicle.port}")
        print(f"  센서: {self.sensor.host}:{self.sensor.port}")
        print(f"  종료: '종료' 또는 Ctrl+C")
        print("=" * 50)

        # 시작 인사 (이건 끝까지 재생)
        self.speak("안녕하세요, RC카 AI 어시스턴트입니다. 명령을 말씀해 주세요.")
        self._wait_tts()

        while True:
            # 긴급 정지 이벤트 체크
            self._check_emergency()

            user_text = self.listen()

            # 듣는 동안 긴급 정지가 발생했을 수 있음
            self._check_emergency()

            if user_text is None:
                print("  (인식 실패 - 다시 시도)")
                continue

            # 사용자가 말하면 TTS 즉시 중단
            self._stop_tts()

            print(f"\n사용자: {user_text}")

            # 종료 키워드
            if user_text.strip() in ("종료", "끝", "그만", "종료해", "끝내자"):
                self.speak("종료하겠습니다. 안녕히 가세요.")
                self._wait_tts()
                break

            # Claude API 호출
            try:
                response_text = self.think(user_text)
                print(f"Claude: {response_text}")
                self.speak(response_text)
            except anthropic.APIError as e:
                print(f"[API Error] {e}")
                self.speak("AI 서버 연결 오류입니다. 다시 시도해 주세요.")

    def _check_emergency(self):
        """긴급 정지 이벤트가 있으면 TTS 알림"""
        try:
            dist = self.sensor.emergency_queue.get_nowait()
        except queue.Empty:
            return

        # 즉시 TTS 경고 (짧게)
        alert = f"전방 {dist}센티, 장애물 감지, 정지했습니다."
        print(f"\nClaude: {alert}")
        self.speak(alert)

        # 히스토리 정리: 마지막 메시지가 tool_use를 포함하면 꼬이므로 제거
        while self.history and self.history[-1].get("role") == "assistant":
            last = self.history[-1]
            # content가 리스트이고 tool_use 블록이 있으면 해당 쌍 제거
            if isinstance(last.get("content"), list):
                self.history.pop()
                if self.history and self.history[-1].get("role") == "user":
                    self.history.pop()
            else:
                break


# ============================================================
# Main
# ============================================================
def main():
    # API 키 확인
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[Error] ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        print("  set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # 인자 파싱
    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    cmd_port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CMD_PORT
    sensor_port = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SENSOR_PORT

    # 센서 수신 시작 (백그라운드)
    sensor = SensorReceiver(host, sensor_port)
    sensor.start()

    # 차량 제어 연결
    vehicle = VehicleController(host, cmd_port, sensor=sensor)
    vehicle.connect()  # 실패해도 계속 (대화는 가능)

    # 센서 긴급 정지 연결: 초음파 20cm 미만 → 즉시 Stop
    sensor.set_emergency_callback(lambda: vehicle.send_command(b"S"))

    # 음성 어시스턴트 시작
    try:
        assistant = VoiceAssistant(sensor, vehicle)
        assistant.run()
    except KeyboardInterrupt:
        print("\n[Exit] Ctrl+C")
    finally:
        vehicle.close()
        sensor.stop()
        print("[Done]")


if __name__ == "__main__":
    main()
