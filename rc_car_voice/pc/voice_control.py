"""
RC Car Voice Client
PC 마이크 입력 → faster-whisper(STT) → 명령 파싱 → TCP 소켓으로 RPi5에 1바이트 전송

Pipeline:
    sounddevice(InputStream) → audio_queue → process_audio(thread)
        → 무음 1초 감지 → Whisper transcribe → parse_command → TCP send

Usage:
    python voice_control.py                        # 기본: 192.168.30.28:9000
    python voice_control.py 192.168.30.28 9000     # IP/포트 지정
    python voice_control.py --dry-run              # 마이크/STT 단독 테스트(네트워크 X)

Requirements:
    pip install sounddevice faster-whisper numpy
    (CUDA 지원 GPU 권장)
"""

import os
import socket
import sys
import threading
import queue


def _register_cuda_dlls():
    """Windows: nvidia-* pip 휠이 푼 cuBLAS/cuDNN DLL 경로를 로더에 등록.
    ctranslate2(faster-whisper의 백엔드)는 PATH가 아닌 DLL search path를 본다."""
    if os.name != 'nt':
        return

    import glob
    import sysconfig

    # site-packages 안의 nvidia\*\bin 모두 수집
    candidates = set()
    for key in ("purelib", "platlib"):
        sp = sysconfig.get_paths().get(key)
        if sp:
            candidates.update(glob.glob(os.path.join(sp, "nvidia", "*", "bin")))

    # importlib 경로도 보조로
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if spec and spec.submodule_search_locations:
            for root in spec.submodule_search_locations:
                candidates.update(glob.glob(os.path.join(root, "*", "bin")))
    except Exception:
        pass

    registered = []
    for bin_dir in sorted(candidates):
        if os.path.isdir(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
                # PATH에도 추가 (이중 보험)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                registered.append(bin_dir)
            except (OSError, AttributeError) as e:
                print(f"[CUDA] add_dll_directory failed: {bin_dir} ({e})")

    if registered:
        print(f"[CUDA] registered {len(registered)} DLL dir(s):")
        for d in registered:
            has_cublas = os.path.isfile(os.path.join(d, "cublas64_12.dll"))
            mark = "  <-- cublas64_12.dll" if has_cublas else ""
            print(f"  - {d}{mark}")
    else:
        print("[CUDA] WARNING: no nvidia\\*\\bin directories found in site-packages")


_register_cuda_dlls()

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# --- Network ---
DEFAULT_HOST = "192.168.30.28"
DEFAULT_PORT = 9000

# --- Audio ---
SAMPLE_RATE = 16000          # Whisper 표준 샘플레이트
BLOCK_SIZE = 1600            # 100ms 단위 콜백
CHANNELS = 1
SILENCE_THRESHOLD = 0.03     # 이 값보다 작으면 무음 취급 (모터 소음 위로 설정)
SILENCE_DURATION = 0.7       # 무음 N초 = 발화 종료
MIN_SPEECH_DURATION = 0.3    # 너무 짧은 잡음은 버림
MAX_RECORDING_DURATION = 2.5 # 무음이 안 와도 N초 지나면 강제 transcribe (모터 노이즈 대응)

# --- Whisper ---
MODEL_SIZE = "medium"
DEVICE = "cuda"              # CPU 사용 시 "cpu" 로 변경
COMPUTE_TYPE = "float16"     # CPU 사용 시 "int8" 권장
LANGUAGE = "ko"

# --- Command parsing ---
CMD_FORWARD  = b'F'
CMD_BACKWARD = b'B'
CMD_LEFT     = b'L'
CMD_RIGHT    = b'R'
CMD_STOP     = b'S'

CMD_NAMES = {
    b'F': 'Forward',
    b'B': 'Backward',
    b'L': 'Left',
    b'R': 'Right',
    b'S': 'Stop',
}

# 한국어 키워드 → 명령 매핑 (부분 일치)
# 주의: 1글자 키워드는 오탐이 잦아 사용하지 않음 ("우지네" → 우회전 오탐 사례)
KEYWORD_MAP = [
    (('전진', '앞으로', '가자', '출발'), CMD_FORWARD),
    (('후진', '뒤로'),                   CMD_BACKWARD),
    (('좌회전', '왼쪽'),                 CMD_LEFT),
    (('우회전', '오른쪽'),               CMD_RIGHT),
    (('정지', '멈춰', '스톱', '스탑'),   CMD_STOP),
]


def parse_command(text: str):
    """인식된 텍스트에서 명령 키워드를 찾아 1바이트 명령으로 변환."""
    if not text:
        return None
    t = text.replace(' ', '')
    for keywords, cmd in KEYWORD_MAP:
        for kw in keywords:
            if kw in t:
                return cmd
    return None


class VoiceClient:
    def __init__(self, host, port, dry_run=False):
        self.host = host
        self.port = port
        self.dry_run = dry_run
        self.sock = None
        self.last_cmd = None

        self.audio_queue = queue.Queue()
        self.recording_buffer = []
        self.silence_counter = 0.0

        self.model = None

    # ---------- Network ----------
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.host, self.port))
        print(f"[Connected] {self.host}:{self.port}")

    def send_command(self, cmd: bytes):
        if cmd == self.last_cmd:
            return  # 동일 명령 중복 억제
        if self.dry_run:
            self.last_cmd = cmd
            print(f"  [DRY] {cmd.decode()} ({CMD_NAMES.get(cmd, '?')})")
            return
        try:
            self.sock.sendall(cmd)
            self.last_cmd = cmd
            print(f"  [TX] {cmd.decode()} ({CMD_NAMES.get(cmd, '?')})")
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[Disconnected] {e}")
            self.last_cmd = None
            raise

    # ---------- Audio ----------
    def audio_callback(self, indata, frames, time_info, status):
        """sounddevice가 BLOCK_SIZE 마다 호출. 큐에 적재만 한다."""
        if status:
            print(f"[Audio] {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def process_audio(self):
        """별도 스레드: 큐에서 오디오 꺼내 무음/최대길이 감지 → Whisper 처리."""
        block_seconds = BLOCK_SIZE / SAMPLE_RATE

        while True:
            chunk = self.audio_queue.get()
            volume = float(np.abs(chunk).mean())

            if volume > SILENCE_THRESHOLD:
                # 발화(혹은 노이즈) 중
                self.recording_buffer.append(chunk)
                self.silence_counter = 0.0
            else:
                # 무음
                if not self.recording_buffer:
                    continue

                self.silence_counter += block_seconds
                if self.silence_counter < SILENCE_DURATION:
                    # 발화 중간의 짧은 무음일 수 있어 버퍼에 포함
                    self.recording_buffer.append(chunk)

            # 종료 조건 — 무음 도달 OR 최대 녹음 길이 초과
            if not self.recording_buffer:
                continue

            buffer_duration = len(self.recording_buffer) * block_seconds
            silence_done = self.silence_counter >= SILENCE_DURATION
            max_reached = buffer_duration >= MAX_RECORDING_DURATION

            if not (silence_done or max_reached):
                continue

            audio_data = np.concatenate(self.recording_buffer).flatten()
            self.recording_buffer = []
            self.silence_counter = 0.0

            duration = len(audio_data) / SAMPLE_RATE
            if duration < MIN_SPEECH_DURATION:
                continue

            reason = "max" if max_reached and not silence_done else "silence"
            self.transcribe_and_send(audio_data, duration, reason)

    def transcribe_and_send(self, audio_data: np.ndarray, duration: float, reason: str = "silence"):
        try:
            segments, _ = self.model.transcribe(
                audio_data,
                language=LANGUAGE,
                beam_size=1,
                vad_filter=False,
            )
            text = " ".join(s.text for s in segments).strip()
        except Exception as e:
            print(f"[Whisper Error] {e}")
            return

        if not text:
            return

        print(f"  [STT] ({duration:.1f}s/{reason}) \"{text}\"")
        cmd = parse_command(text)
        if cmd is None:
            print("  [Parse] (no match)")
            return

        try:
            self.send_command(cmd)
        except OSError:
            pass

    # ---------- Lifecycle ----------
    def load_model(self):
        print(f"[Whisper] loading {MODEL_SIZE} ({DEVICE}/{COMPUTE_TYPE}) ...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("[Whisper] ready")

    def run(self):
        print("=" * 44)
        print("  RC Car Voice Controller")
        print("=" * 44)
        target = "DRY-RUN (no network)" if self.dry_run else f"{self.host}:{self.port}"
        print(f"  Target : {target}")
        print(f"  Model  : faster-whisper {MODEL_SIZE} / {DEVICE}")
        print(f"  Audio  : {SAMPLE_RATE}Hz, block {BLOCK_SIZE} samples")
        print("  Commands: 전진해 / 후진해 / 좌회전 / 우회전 / 정지")
        print("  Ctrl+C to exit")
        print("=" * 44)

        self.load_model()
        if not self.dry_run:
            self.connect()

        worker = threading.Thread(target=self.process_audio, daemon=True)
        worker.start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype='float32',
            callback=self.audio_callback,
        ):
            print("[Mic] listening...")
            try:
                while True:
                    sd.sleep(1000)
            finally:
                if self.sock:
                    self.sock.close()


def main():
    args = [a for a in sys.argv[1:] if a]
    dry_run = False
    if "--dry-run" in args:
        dry_run = True
        args.remove("--dry-run")

    host = args[0] if len(args) > 0 else DEFAULT_HOST
    port = int(args[1]) if len(args) > 1 else DEFAULT_PORT

    client = VoiceClient(host, port, dry_run=dry_run)
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[Exit] Ctrl+C")
    except ConnectionRefusedError:
        print(f"[Error] Cannot connect to {host}:{port}")
        print("  - RPi5에서 uart_server.py 실행 중인지 확인")
        print("  - 같은 Wi-Fi 네트워크인지 확인")


if __name__ == "__main__":
    main()
