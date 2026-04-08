"""
RC Car Keyboard Client
PC에서 키보드 입력(WASD + Space)을 감지하여 TCP 소켓으로 RPi5에 1바이트 명령 전송

Usage:
    python keyboard_client.py                    # 기본: 192.168.30.34:9000
    python keyboard_client.py 192.168.30.34 9000 # IP/포트 지정
"""

import socket
import sys
from pynput import keyboard

# --- Configuration ---
DEFAULT_HOST = "192.168.30.28"
DEFAULT_PORT = 9000

# Key → UART command mapping
KEY_MAP = {
    'w': b'F',  # Forward
    's': b'B',  # Backward
    'a': b'L',  # Left
    'd': b'R',  # Right
}

STOP_CMD = b'S'

CMD_NAMES = {
    b'F': 'Forward',
    b'B': 'Backward',
    b'L': 'Left',
    b'R': 'Right',
    b'S': 'Stop',
}


class KeyboardClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.last_cmd = None
        self.pressed_keys = set()

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.host, self.port))
        print(f"[Connected] {self.host}:{self.port}")

    def send_cmd(self, cmd):
        if cmd == self.last_cmd:
            return
        try:
            self.sock.sendall(cmd)
            self.last_cmd = cmd
            print(f"  [TX] {cmd.decode()} ({CMD_NAMES.get(cmd, '?')})")
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[Disconnected] Connection lost")
            self.last_cmd = None
            raise

    def on_press(self, key):
        try:
            ch = key.char.lower() if key.char else None
        except AttributeError:
            if key == keyboard.Key.space:
                self.send_cmd(STOP_CMD)
            elif key == keyboard.Key.esc:
                print("\n[Exit] ESC pressed")
                return False
            return

        if ch in KEY_MAP:
            self.pressed_keys.add(ch)
            self.send_cmd(KEY_MAP[ch])

    def on_release(self, key):
        try:
            ch = key.char.lower() if key.char else None
        except AttributeError:
            ch = None

        if ch in self.pressed_keys:
            self.pressed_keys.discard(ch)

        if not self.pressed_keys:
            self.send_cmd(STOP_CMD)

    def run(self):
        print("=" * 40)
        print("  RC Car Keyboard Controller")
        print("=" * 40)
        print(f"  Target: {self.host}:{self.port}")
        print("  W=Forward  S=Backward")
        print("  A=Left     D=Right")
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
