"""
RC Car Keyboard Client
PC에서 키보드 입력(방향키 + U/I)을 감지하여 TCP 소켓으로 RPi5에 1바이트 명령 전송

Usage:
    python keyboard_client.py                    # 기본: 192.168.0.23:9000
    python keyboard_client.py 192.168.0.23 9000  # IP/포트 지정
"""

import socket
import sys
from pynput import keyboard

# --- Configuration ---
DEFAULT_HOST = "192.168.0.23"
DEFAULT_PORT = 9000

# 방향키 → UART 명령
ARROW_MAP = {
    keyboard.Key.up:    b'F',   # Forward
    keyboard.Key.down:  b'B',   # Backward
    keyboard.Key.left:  b'L',   # Left
    keyboard.Key.right: b'R',   # Right
}

# 서보 제어 키 (누를 때마다 스텝 이동)
SERVO_MAP = {
    'u': b'U',  # 서보 왼쪽으로 스텝
    'i': b'I',  # 서보 오른쪽으로 스텝
}

STOP_CMD = b'S'

CMD_NAMES = {
    b'F': 'Forward',
    b'B': 'Backward',
    b'L': 'Left',
    b'R': 'Right',
    b'S': 'Stop',
    b'U': 'Servo Left',
    b'I': 'Servo Right',
}


class KeyboardClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.last_move_cmd = None
        self.pressed_arrows = set()

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.host, self.port))
        print(f"[Connected] {self.host}:{self.port}")

    def send_cmd(self, cmd, is_move=True):
        # 이동 명령은 동일 명령 중복 방지, 서보 명령은 매번 전송
        if is_move and cmd == self.last_move_cmd:
            return
        try:
            self.sock.sendall(cmd)
            if is_move:
                self.last_move_cmd = cmd
            print(f"  [TX] {cmd.decode()} ({CMD_NAMES.get(cmd, '?')})")
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[Disconnected] Connection lost")
            self.last_move_cmd = None
            raise

    def on_press(self, key):
        # 방향키
        if key in ARROW_MAP:
            self.pressed_arrows.add(key)
            self.send_cmd(ARROW_MAP[key], is_move=True)
            return

        # 특수키
        if key == keyboard.Key.space:
            self.send_cmd(STOP_CMD, is_move=True)
            return
        if key == keyboard.Key.esc:
            print("\n[Exit] ESC pressed")
            return False

        # 문자키 (서보)
        try:
            ch = key.char.lower() if key.char else None
        except AttributeError:
            return

        if ch in SERVO_MAP:
            self.send_cmd(SERVO_MAP[ch], is_move=False)

    def on_release(self, key):
        if key in self.pressed_arrows:
            self.pressed_arrows.discard(key)
            if not self.pressed_arrows:
                self.send_cmd(STOP_CMD, is_move=True)

    def run(self):
        print("=" * 40)
        print("  RC Car Keyboard Controller")
        print("=" * 40)
        print(f"  Target: {self.host}:{self.port}")
        print("  ↑=Forward   ↓=Backward")
        print("  ←=Left      →=Right")
        print("  U=Servo←    I=Servo→")
        print("  Space=Stop  ESC=Exit")
        print("=" * 40)

        self.connect()

        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as listener:
            listener.join()

        if self.sock:
            self.sock.close()
        print("[Done]")


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

    client = KeyboardClient(host, port)
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
