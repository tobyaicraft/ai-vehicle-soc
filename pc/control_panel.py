"""
RC Car Unified Control Panel
카메라 뷰 + 키보드 조종 + 센서 모니터를 하나의 GUI로 통합

Usage:
    python control_panel.py                      # 기본: 192.168.0.23
    python control_panel.py 192.168.0.23         # IP 지정

조작:
    방향키      : 전진/후진/좌/우
    Space       : 정지
    U / I       : 서보 좌/우
    P           : 자동 주차 모드
    ESC         : 종료
"""

import io
import math
import socket
import sys
import threading
import time
import urllib.request

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Configuration ---
from config import RPI5_HOST as DEFAULT_HOST, RPI5_HOSTS, CMD_PORT, SENSOR_PORT, CAM_PORT
ADC_MAX = 4095
VAREF = 5.0

# --- TC237 패킷 프로토콜 ---
PROTO_STX = 0xAA
PROTO_ETX = 0x55
CMD_MOVE = 0x01
CMD_MODE = 0x02
DEFAULT_SPEED = 100

DIR_MAP = {
    "Up": 1,       # FORWARD
    "Down": 2,     # REVERSE
    "Left": 3,     # LEFT
    "Right": 4,    # RIGHT
}

# --- Color Palette ---
BG_COLOR = "#1e1e2e"
FG_COLOR = "#cdd6f4"
ACCENT_L = "#89b4fa"
ACCENT_R = "#f38ba8"
ACCENT_U = "#a6e3a1"
WARN_COLOR = "#f9e2af"
DANGER_COLOR = "#f38ba8"
CHART_BG = "#181825"
GRID_COLOR = "#313244"
SURFACE = "#313244"
KEY_ACTIVE = "#a6e3a1"
KEY_INACTIVE = "#45475a"
ACCENT_Y = "#ffcc00"

# --- 3D Car Model ---
CAR_BODY = np.array([
    [-1.0, -1.8, -0.3], [ 1.0, -1.8, -0.3],
    [ 1.0,  1.8, -0.3], [-1.0,  1.8, -0.3],
    [-1.0, -1.8,  0.3], [ 1.0, -1.8,  0.3],
    [ 1.0,  1.8,  0.3], [-1.0,  1.8,  0.3],
])
CAR_FACES = [
    [0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]
]
CAR_COLORS = [
    (0.08,0.08,0.15,0.9), (0.12,0.12,0.25,0.9),
    (0.8,0.2,0.2,0.8), (0.15,0.15,0.4,0.8),
    (0.9,0.7,0.1,0.8), (0.1,0.8,0.3,0.8),
]

WHEEL_W, WHEEL_H, WHEEL_D = 0.25, 0.5, 0.2
def make_wheel(cx, cy, cz):
    return np.array([
        [cx-WHEEL_W, cy-WHEEL_H, cz-WHEEL_D],[cx+WHEEL_W, cy-WHEEL_H, cz-WHEEL_D],
        [cx+WHEEL_W, cy+WHEEL_H, cz-WHEEL_D],[cx-WHEEL_W, cy+WHEEL_H, cz-WHEEL_D],
        [cx-WHEEL_W, cy-WHEEL_H, cz+WHEEL_D],[cx+WHEEL_W, cy-WHEEL_H, cz+WHEEL_D],
        [cx+WHEEL_W, cy+WHEEL_H, cz+WHEEL_D],[cx-WHEEL_W, cy+WHEEL_H, cz+WHEEL_D],
    ])

WHEELS = [
    make_wheel(-1.3, -1.2, -0.3), make_wheel(1.3, -1.2, -0.3),
    make_wheel(-1.3,  1.2, -0.3), make_wheel(1.3,  1.2, -0.3),
]

def rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
    Ry = np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])
    Rx = np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx


def adc_to_voltage(adc_val):
    return adc_val / ADC_MAX * VAREF


def voltage_to_distance_cm(voltage):
    if voltage < 0.3:
        return 80.0
    if voltage > 3.2:
        return 10.0
    try:
        dist = 29.988 * pow(voltage, -1.173)
    except (ValueError, ZeroDivisionError):
        return 80.0
    return max(10.0, min(80.0, dist))


def build_move_packet(direction, speed=DEFAULT_SPEED):
    cmd = CMD_MOVE
    chk = cmd ^ direction ^ speed
    return bytes([PROTO_STX, 0x03, cmd, direction, speed, chk, PROTO_ETX])


def build_mode_packet(mode):
    cmd = CMD_MODE
    chk = cmd ^ mode
    return bytes([PROTO_STX, 0x02, cmd, mode, chk, PROTO_ETX])


class ControlPanel:
    def __init__(self, root, host):
        self.root = root
        self.host = host
        self.root.title("RC Car Control Panel")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1280x900")
        self.root.minsize(1100, 800)

        # --- State ---
        self.cmd_sock = None
        self.sensor_sock = None
        self.connected = False
        self.stop_event = threading.Event()
        self.pressed_keys = set()
        self.last_cmd = None

        # Sensor
        self.ir_left = 0
        self.ir_right = 0
        self.us_dist = 0
        self.bat_mv = 0

        # Camera
        self.cam_photo = None
        self.cam_frame = None  # 최신 JPEG bytes (스레드에서 저장)
        self.cam_running = False

        # IMU
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # GPS
        self.gps_lat = 0.0
        self.gps_lon = 0.0
        self.gps_speed = 0.0
        self.gps_sats = 0
        self.gps_home = None  # (lat, lon)
        self.gps_track = []   # [(lat, lon), ...]
        self.gps_returning = False

        # Driving Mode
        self.drive_mode = "MANUAL"  # "MANUAL", "CAT_TRACK", "GPS_RETURN"
        self.cat_track_running = False
        self.cat_no_detect_count = 0

        # Servo
        self.servo_angle = 90.0

        self._build_ui()
        self._bind_keys()
        self._start_animation()

    # ======================== UI ========================
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=BG_COLOR)
        style.configure("Dark.TLabel", background=BG_COLOR, foreground=FG_COLOR,
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG_COLOR, foreground=ACCENT_L,
                        font=("Segoe UI", 14, "bold"))
        style.configure("Big.TLabel", background=BG_COLOR, foreground=FG_COLOR,
                        font=("Consolas", 24, "bold"))
        style.configure("Dist.TLabel", background=BG_COLOR, foreground=ACCENT_U,
                        font=("Consolas", 16, "bold"))
        style.configure("Stat.TLabel", background=BG_COLOR, foreground=FG_COLOR,
                        font=("Consolas", 11))
        style.configure("Key.TLabel", background=KEY_INACTIVE, foreground=FG_COLOR,
                        font=("Consolas", 14, "bold"), padding=8)

        # === Top Bar ===
        top = ttk.Frame(self.root, style="Dark.TFrame")
        top.pack(fill=tk.X, padx=10, pady=(8, 4))

        ttk.Label(top, text="RC Car Control Panel",
                  style="Title.TLabel").pack(side=tk.LEFT)

        self.btn_connect = ttk.Button(top, text="Connect",
                                       command=self._toggle_connect)
        self.btn_connect.pack(side=tk.RIGHT, padx=(5, 0))

        self.entry_host = ttk.Combobox(top, width=18, font=("Consolas", 10))
        self.entry_host["values"] = RPI5_HOSTS
        self.entry_host.set(self.host)
        self.entry_host.pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Label(top, text="RPi5 IP:", style="Dark.TLabel").pack(side=tk.RIGHT)

        # Driving mode selector
        self.drive_combo = ttk.Combobox(top, width=10, state="readonly",
                                          font=("Consolas", 10))
        self.drive_combo["values"] = ["MANUAL", "CAT_TRACK", "GPS_RETURN"]
        self.drive_combo.set("MANUAL")
        self.drive_combo.pack(side=tk.RIGHT, padx=(5, 0))
        self.drive_combo.bind("<<ComboboxSelected>>", self._on_drive_mode_change)
        ttk.Label(top, text="Mode:", style="Dark.TLabel").pack(side=tk.RIGHT)

        # Detection mode selector
        self.detect_combo = ttk.Combobox(top, width=8, state="readonly",
                                          font=("Consolas", 10))
        self.detect_combo["values"] = ["none", "blue", "ssd", "cat_custom", "yolo_onnx", "yolo_tflite"]
        self.detect_combo.set("none")
        self.detect_combo.pack(side=tk.RIGHT, padx=(5, 0))
        self.detect_combo.bind("<<ComboboxSelected>>", self._on_detect_mode_change)
        ttk.Label(top, text="Detect:", style="Dark.TLabel").pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(top, textvariable=self.status_var, style="Dark.TLabel").pack(
            side=tk.RIGHT, padx=(15, 10))

        # === Main Content: Left (Camera + Car) | Right (Chart + Sensors) ===
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # --- Left Column ---
        left_col = ttk.Frame(main, style="Dark.TFrame")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # Camera
        cam_frame = ttk.Frame(left_col, style="Dark.TFrame")
        cam_frame.pack(fill=tk.X, padx=(0, 5), pady=(0, 5))
        self.cam_label = tk.Label(cam_frame, bg=CHART_BG, width=640, height=480,
                                  text="Camera\n(Disconnected)", fg="#6c7086",
                                  font=("Segoe UI", 14))
        self.cam_label.pack()

        # Key indicator
        key_frame = ttk.Frame(left_col, style="Dark.TFrame")
        key_frame.pack(fill=tk.X, padx=(0, 5), pady=(5, 0))

        self.key_labels = {}
        key_layout = [
            [("", None), ("\u2191", "Up"), ("", None)],
            [("\u2190", "Left"), ("\u2193", "Down"), ("\u2192", "Right")],
        ]

        for row_data in key_layout:
            row_frame = ttk.Frame(key_frame, style="Dark.TFrame")
            row_frame.pack()
            for text, key_name in row_data:
                lbl = tk.Label(row_frame, text=text, width=4, height=1,
                               bg=KEY_INACTIVE, fg=FG_COLOR,
                               font=("Consolas", 16, "bold"), relief="raised", bd=2)
                lbl.pack(side=tk.LEFT, padx=2, pady=2)
                if key_name:
                    self.key_labels[key_name] = lbl

        # Extra keys row
        extra_frame = ttk.Frame(key_frame, style="Dark.TFrame")
        extra_frame.pack(pady=(4, 0))
        for text, key_name in [("U", "u"), ("I", "i"), ("SPC", "space"), ("P", "p"), ("R", "r")]:
            lbl = tk.Label(extra_frame, text=text, width=4, height=1,
                           bg=KEY_INACTIVE, fg=FG_COLOR,
                           font=("Consolas", 12, "bold"), relief="raised", bd=2)
            lbl.pack(side=tk.LEFT, padx=2, pady=2)
            self.key_labels[key_name] = lbl

        # Servo angle display
        self.servo_var = tk.StringVar(value="Servo: 90.0\u00b0")
        ttk.Label(key_frame, textvariable=self.servo_var,
                  style="Stat.TLabel").pack(pady=(4, 0))

        # --- Right Column ---
        right_col = ttk.Frame(main, style="Dark.TFrame")
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # === 상단: AI Vision (왼쪽) + 3D IMU (오른쪽) ===
        top_row = ttk.Frame(right_col, style="Dark.TFrame")
        top_row.pack(fill=tk.BOTH, expand=True)

        # GPS Panel
        gps_frame = ttk.Frame(top_row, style="Dark.TFrame")
        gps_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # GPS info labels
        gps_info = ttk.Frame(gps_frame, style="Dark.TFrame")
        gps_info.pack(fill=tk.X)
        tk.Label(gps_info, text="GPS", bg=BG_COLOR, fg=ACCENT_U,
                 font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        self.gps_sats_var = tk.StringVar(value="SAT:0")
        tk.Label(gps_info, textvariable=self.gps_sats_var, bg=BG_COLOR, fg="#6c7086",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 8))
        self.gps_speed_var = tk.StringVar(value="0.0 km/h")
        tk.Label(gps_info, textvariable=self.gps_speed_var, bg=BG_COLOR, fg=ACCENT_U,
                 font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=(0, 8))

        # Home / Return buttons
        self.btn_home = tk.Button(gps_info, text="H:Set Home", width=10,
                                   bg="#45475a", fg=FG_COLOR, relief=tk.FLAT,
                                   font=("Consolas", 8, "bold"), cursor="hand2",
                                   command=self._gps_set_home)
        self.btn_home.pack(side=tk.RIGHT, padx=2)
        self.btn_return = tk.Button(gps_info, text="G:Return", width=10,
                                     bg="#45475a", fg=WARN_COLOR, relief=tk.FLAT,
                                     font=("Consolas", 8, "bold"), cursor="hand2",
                                     command=self._gps_return_home)
        self.btn_return.pack(side=tk.RIGHT, padx=2)

        # GPS coordinate display
        gps_coord = ttk.Frame(gps_frame, style="Dark.TFrame")
        gps_coord.pack(fill=tk.X)
        self.gps_lat_var = tk.StringVar(value="LAT: --")
        self.gps_lon_var = tk.StringVar(value="LON: --")
        self.gps_home_var = tk.StringVar(value="HOME: not set")
        self.gps_dist_var = tk.StringVar(value="")
        tk.Label(gps_coord, textvariable=self.gps_lat_var, bg=BG_COLOR, fg=FG_COLOR,
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(gps_coord, textvariable=self.gps_lon_var, bg=BG_COLOR, fg=FG_COLOR,
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(gps_coord, textvariable=self.gps_home_var, bg=BG_COLOR, fg="#6c7086",
                 font=("Consolas", 8)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(gps_coord, textvariable=self.gps_dist_var, bg=BG_COLOR, fg=WARN_COLOR,
                 font=("Consolas", 9, "bold")).pack(side=tk.LEFT)

        # GPS track canvas (이동 경로 시각화)
        self.gps_canvas = tk.Canvas(gps_frame, bg=CHART_BG, highlightthickness=0)
        self.gps_canvas.pack(fill=tk.BOTH, expand=True)

        # 3D IMU (compact, right side)
        imu_frame = ttk.Frame(top_row, style="Dark.TFrame", width=300)
        imu_frame.pack(side=tk.RIGHT, fill=tk.Y)
        imu_frame.pack_propagate(False)

        # IMU RPY labels
        imu_label_frame = ttk.Frame(imu_frame, style="Dark.TFrame")
        imu_label_frame.pack(fill=tk.X)
        ttk.Label(imu_label_frame, text="IMU", foreground=ACCENT_Y,
                  background=BG_COLOR, font=("Consolas", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        self.imu_labels = {}
        for name, color in [("R", "#5b9aff"), ("P", "#ff9a5b"), ("Y", "#ffcc00")]:
            tk.Label(imu_label_frame, text=name, bg=BG_COLOR, fg="#6c7086",
                     font=("Consolas", 8)).pack(side=tk.LEFT)
            lbl = tk.Label(imu_label_frame, text="+000.0", bg=BG_COLOR, fg=color,
                           font=("Consolas", 10, "bold"))
            lbl.pack(side=tk.LEFT, padx=(0, 4))
            self.imu_labels[name] = lbl

        # 3D matplotlib figure (compact)
        self.fig3d = Figure(figsize=(2.8, 2.5), dpi=100, facecolor=CHART_BG)
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.ax3d.set_facecolor(CHART_BG)
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=imu_frame)
        self.canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === 하단: 센서 시각화 (Left IR | Car Top View | Right IR) ===
        sensor_bottom = ttk.Frame(right_col, style="Dark.TFrame")
        sensor_bottom.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Left IR
        lp = ttk.Frame(sensor_bottom, style="Dark.TFrame", width=130)
        lp.pack(side=tk.LEFT, fill=tk.Y)
        lp.pack_propagate(False)
        ttk.Label(lp, text="LEFT IR", foreground=ACCENT_L,
                  background=BG_COLOR, font=("Segoe UI", 10, "bold")).pack(pady=(5, 0))
        self.lbl_left_adc = ttk.Label(lp, text="0", style="Big.TLabel")
        self.lbl_left_adc.pack()
        self.lbl_left_dist = ttk.Label(lp, text="-- cm", style="Dist.TLabel")
        self.lbl_left_dist.pack(pady=(3, 0))
        self.left_gauge = tk.Canvas(lp, height=14, bg=CHART_BG, highlightthickness=0)
        self.left_gauge.pack(fill=tk.X, padx=10, pady=(6, 0))

        # Car top-view (2D sensor visualization)
        center = ttk.Frame(sensor_bottom, style="Dark.TFrame")
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.car_canvas = tk.Canvas(center, bg=CHART_BG, highlightthickness=0, height=220)
        self.car_canvas.pack(fill=tk.BOTH, expand=True)

        # Right IR
        rp = ttk.Frame(sensor_bottom, style="Dark.TFrame", width=130)
        rp.pack(side=tk.RIGHT, fill=tk.Y)
        rp.pack_propagate(False)
        ttk.Label(rp, text="RIGHT IR", foreground=ACCENT_R,
                  background=BG_COLOR, font=("Segoe UI", 10, "bold")).pack(pady=(5, 0))
        self.lbl_right_adc = ttk.Label(rp, text="0", style="Big.TLabel")
        self.lbl_right_adc.pack()
        self.lbl_right_dist = ttk.Label(rp, text="-- cm", style="Dist.TLabel")
        self.lbl_right_dist.pack(pady=(3, 0))
        self.right_gauge = tk.Canvas(rp, height=14, bg=CHART_BG, highlightthickness=0)
        self.right_gauge.pack(fill=tk.X, padx=10, pady=(6, 0))

        # Battery bar at very bottom
        bat_frame = ttk.Frame(self.root, style="Dark.TFrame")
        bat_frame.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.bat_var = tk.StringVar(value="BAT: -- mV")
        ttk.Label(bat_frame, textvariable=self.bat_var,
                  style="Stat.TLabel").pack(side=tk.LEFT)
        self.cmd_var = tk.StringVar(value="")
        ttk.Label(bat_frame, textvariable=self.cmd_var,
                  foreground=ACCENT_U, background=BG_COLOR,
                  font=("Consolas", 11)).pack(side=tk.RIGHT)

    def _draw_car(self, left_dist, right_dist, us_dist):
        c = self.car_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1:
            return

        cx, cy = w // 2, h // 2 + 20
        car_w, car_h = 60, 90

        # Car body
        c.create_rectangle(cx - car_w // 2, cy - car_h // 2,
                           cx + car_w // 2, cy + car_h // 2,
                           fill="#45475a", outline="#585b70", width=2)
        c.create_text(cx, cy + 5, text="TC237", fill=FG_COLOR,
                      font=("Consolas", 9, "bold"))

        # Wheels
        for dy in [-30, 30]:
            c.create_rectangle(cx - car_w // 2 - 7, cy + dy - 10,
                               cx - car_w // 2, cy + dy + 10,
                               fill="#6c7086", outline="")
            c.create_rectangle(cx + car_w // 2, cy + dy - 10,
                               cx + car_w // 2 + 7, cy + dy + 10,
                               fill="#6c7086", outline="")

        # Ultrasonic cone
        front_y = cy - car_h // 2
        us_len = max(15, min(70, us_dist * 0.5))
        us_color = DANGER_COLOR if us_dist < 20 else (
            WARN_COLOR if us_dist < 50 else ACCENT_U)
        c.create_polygon(cx - 12, front_y, cx + 12, front_y,
                         cx + 30, front_y - us_len, cx - 30, front_y - us_len,
                         fill="", outline=us_color, width=2)
        c.create_text(cx, front_y - us_len - 12,
                      text=f"{us_dist:.0f} cm", fill=us_color,
                      font=("Consolas", 11, "bold"))

        # IR Left
        ir_angle = math.radians(45)
        left_len = max(12, min(60, left_dist * 0.7))
        left_color = DANGER_COLOR if left_dist < 20 else (
            WARN_COLOR if left_dist < 40 else ACCENT_L)
        lx0 = cx - car_w // 2
        ly0 = front_y
        lx1 = lx0 - left_len * math.sin(ir_angle)
        ly1 = ly0 - left_len * math.cos(ir_angle)
        c.create_line(lx0, ly0, lx1, ly1, fill=left_color, width=2, arrow=tk.LAST)
        c.create_text(lx1 - 5, ly1 - 12,
                      text=f"{left_dist:.0f}", fill=left_color,
                      font=("Consolas", 10, "bold"))

        # IR Right
        right_len = max(12, min(60, right_dist * 0.7))
        right_color = DANGER_COLOR if right_dist < 20 else (
            WARN_COLOR if right_dist < 40 else ACCENT_R)
        rx0 = cx + car_w // 2
        ry0 = front_y
        rx1 = rx0 + right_len * math.sin(ir_angle)
        ry1 = ry0 - right_len * math.cos(ir_angle)
        c.create_line(rx0, ry0, rx1, ry1, fill=right_color, width=2, arrow=tk.LAST)
        c.create_text(rx1 + 5, ry1 - 12,
                      text=f"{right_dist:.0f}", fill=right_color,
                      font=("Consolas", 10, "bold"))

        c.create_text(cx, front_y - 8, text="FRONT",
                      fill="#6c7086", font=("Segoe UI", 8))

    # ======================== Keys ========================
    def _bind_keys(self):
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

    def _on_key_press(self, event):
        key = event.keysym

        # 자동 모드에서 방향키/스페이스 차단
        if self.drive_mode != "MANUAL":
            if key in DIR_MAP or key == "space":
                return

        # Arrow keys
        if key in DIR_MAP:
            if key not in self.pressed_keys:
                self.pressed_keys.add(key)
                self._highlight_key(key, True)
            self._send_move(DIR_MAP[key])
            return

        # Space = stop
        if key == "space":
            self._highlight_key("space", True)
            self._send_move(0)
            self.root.after(150, lambda: self._highlight_key("space", False))
            return

        # Servo
        ch = key.lower()
        if ch == "u":
            self._highlight_key("u", True)
            self.servo_angle = max(0.0, self.servo_angle - 10.0)
            self.servo_var.set(f"Servo: {self.servo_angle:.1f}\u00b0")
            self._send_servo(b'U')
            self.root.after(150, lambda: self._highlight_key("u", False))
        elif ch == "i":
            self._highlight_key("i", True)
            self.servo_angle = min(180.0, self.servo_angle + 10.0)
            self.servo_var.set(f"Servo: {self.servo_angle:.1f}\u00b0")
            self._send_servo(b'I')
            self.root.after(150, lambda: self._highlight_key("i", False))

        # Parking
        if ch == "p":
            self._highlight_key("p", True)
            self._send_mode(2)  # AUTO
            self.root.after(150, lambda: self._highlight_key("p", False))

        # MCU Reset
        if ch == "r":
            self._highlight_key("r", True)
            self._send_reset()
            self.root.after(150, lambda: self._highlight_key("r", False))

        # GPS Home
        if ch == "h":
            self._gps_set_home()

        # GPS Return Home
        if ch == "g":
            self._gps_return_home()

        # ESC
        if key == "Escape":
            self.on_close()

    def _on_key_release(self, event):
        key = event.keysym
        if key in DIR_MAP:
            self.pressed_keys.discard(key)
            self._highlight_key(key, False)
            if not self.pressed_keys and self.drive_mode == "MANUAL":
                self._send_move(0)  # STOP

    def _highlight_key(self, key, active):
        lbl = self.key_labels.get(key)
        if lbl:
            lbl.configure(bg=KEY_ACTIVE if active else KEY_INACTIVE)

    # ======================== Network ========================
    def _send_move(self, direction):
        if not self.cmd_sock:
            return
        # RPi5 uart_server.py가 ASCII 1바이트를 받아서 패킷으로 변환함
        dir_to_ascii = {0: b'S', 1: b'F', 2: b'B', 3: b'L', 4: b'R'}
        cmd = dir_to_ascii.get(direction, b'S')
        try:
            self.cmd_sock.sendall(cmd)
            names = {0: "STOP", 1: "FWD", 2: "REV", 3: "LEFT", 4: "RIGHT"}
            self.cmd_var.set(f"CMD: {names.get(direction, '?')}")
        except OSError:
            pass

    def _send_servo(self, cmd):
        if not self.cmd_sock:
            return
        try:
            self.cmd_sock.sendall(cmd)
        except OSError:
            pass

    def _on_detect_mode_change(self, event=None):
        mode = self.detect_combo.get()
        host = self.entry_host.get().strip()
        if not host:
            return
        try:
            url = f"http://{host}:{CAM_PORT}/mode/{mode}"
            urllib.request.urlopen(url, timeout=2)
            self.cmd_var.set(f"DETECT: {mode.upper()}")
        except Exception:
            pass
        self.root.focus_set()

    def _send_reset(self):
        if not self.cmd_sock:
            return
        try:
            self.cmd_sock.sendall(b'X')
            self.cmd_var.set("CMD: MCU RESET")
        except OSError:
            pass

    def _send_mode(self, mode):
        if not self.cmd_sock:
            return
        try:
            self.cmd_sock.sendall(b'P')
            self.cmd_var.set("CMD: AUTO PARKING")
        except OSError:
            pass

    def _toggle_connect(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        host = self.entry_host.get().strip()
        if not host:
            return

        # CMD socket (9000)
        try:
            self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cmd_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.cmd_sock.settimeout(3.0)
            self.cmd_sock.connect((host, CMD_PORT))
            self.cmd_sock.settimeout(None)
        except Exception as e:
            self.status_var.set(f"CMD connect failed")
            self.cmd_sock = None
            return

        # Sensor socket (9001)
        try:
            self.sensor_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sensor_sock.settimeout(3.0)
            self.sensor_sock.connect((host, SENSOR_PORT))
            self.sensor_sock.settimeout(None)
        except Exception:
            self.status_var.set(f"Sensor connect failed")
            self.sensor_sock = None

        self.connected = True
        self.stop_event.clear()
        self.btn_connect.configure(text="Disconnect")
        self.status_var.set(f"Connected: {host}")

        # Sensor reader thread
        if self.sensor_sock:
            threading.Thread(target=self._sensor_reader, daemon=True).start()

        # Camera stream thread + UI update loop
        self.cam_running = True
        threading.Thread(target=self._camera_reader, args=(host,), daemon=True).start()
        self._update_camera()

        # Move resend thread (MCU 200ms timeout)
        threading.Thread(target=self._move_resend, daemon=True).start()

    def _disconnect(self):
        self.connected = False
        self.cam_running = False
        self.stop_event.set()

        for s in [self.cmd_sock, self.sensor_sock]:
            if s:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass
        self.cmd_sock = None
        self.sensor_sock = None

        self.btn_connect.configure(text="Connect")
        self.status_var.set("Disconnected")
        self.cam_label.configure(image="", text="Camera\n(Disconnected)")

    # ======================== Sensor Reader ========================
    def _sensor_reader(self):
        buf = ""
        while self.connected and self.sensor_sock:
            try:
                raw = self.sensor_sock.recv(512)
                if not raw:
                    break
                buf += raw.decode("ascii", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    # Sensor: "L:xxxx,R:xxxx,U:xxx,B:xxxx,O:x"
                    if line.startswith("L:") and ",R:" in line and ",U:" in line:
                        try:
                            parts = line.split(",")
                            self.ir_left = int(parts[0][2:])
                            self.ir_right = int(parts[1][2:])
                            self.us_dist = int(parts[2][2:])
                            if len(parts) > 3 and parts[3].startswith("B:"):
                                self.bat_mv = int(parts[3][2:])
                        except (ValueError, IndexError):
                            pass
                    # IMU: "R:+012.3,P:-005.7,Y:+045.2"
                    elif line.startswith("R:") and ",P:" in line and ",Y:" in line:
                        try:
                            parts = line.split(",")
                            self.roll = float(parts[0][2:])
                            self.pitch = float(parts[1][2:])
                            self.yaw = float(parts[2][2:])
                        except (ValueError, IndexError):
                            pass
                    # GPS: "G:lat,lon,speed,sats"
                    elif line.startswith("G:"):
                        try:
                            parts = line[2:].split(",")
                            self.gps_lat = float(parts[0])
                            self.gps_lon = float(parts[1])
                            self.gps_speed = float(parts[2])
                            self.gps_sats = int(parts[3])
                            if self.gps_lat != 0.0 and self.gps_lon != 0.0:
                                self.gps_track.append((self.gps_lat, self.gps_lon))
                                if len(self.gps_track) > 1000:
                                    self.gps_track = self.gps_track[-500:]
                        except (ValueError, IndexError):
                            pass
            except OSError:
                break

    # ======================== Camera Reader ========================
    def _camera_reader(self, host):
        """백그라운드: MJPEG 프레임을 읽어서 cam_frame에 저장만 함"""
        url = f"http://{host}:{CAM_PORT}/stream.mjpg"
        try:
            stream = urllib.request.urlopen(url, timeout=5)
        except Exception:
            return

        # Content-Length 기반 MJPEG 파싱
        while self.cam_running and self.connected:
            try:
                line = b""
                content_length = 0
                while True:
                    byte = stream.read(1)
                    if not byte:
                        return
                    line += byte
                    if line.endswith(b'\r\n'):
                        header = line.decode('ascii', errors='ignore').strip()
                        if header.lower().startswith('content-length:'):
                            content_length = int(header.split(':')[1].strip())
                        if line == b'\r\n' and content_length > 0:
                            break
                        line = b""

                jpg = stream.read(content_length)
                if len(jpg) == content_length:
                    self.cam_frame = jpg  # 저장만, UI 업데이트는 메인 스레드에서
            except Exception:
                break

    def _update_camera(self):
        """메인 스레드: cam_frame을 화면에 표시 (30ms 주기)"""
        if self.cam_frame:
            try:
                img = Image.open(io.BytesIO(self.cam_frame))
                photo = ImageTk.PhotoImage(img)
                self.cam_photo = photo
                self.cam_label.configure(image=photo, text="")
            except Exception:
                pass
        if self.connected:
            self.root.after(33, self._update_camera)  # ~30fps

    # ======================== Move Resend ========================
    def _move_resend(self):
        """100ms마다 마지막 방향키 재전송 (MCU 200ms 타임아웃 대응)"""
        dir_to_ascii = {1: b'F', 2: b'B', 3: b'L', 4: b'R'}
        while self.connected and not self.stop_event.is_set():
            if self.pressed_keys and self.cmd_sock and self.drive_mode == "MANUAL":
                for key in list(self.pressed_keys):
                    if key in DIR_MAP:
                        cmd = dir_to_ascii.get(DIR_MAP[key])
                        if cmd:
                            try:
                                self.cmd_sock.sendall(cmd)
                            except OSError:
                                pass
                        break
            time.sleep(0.1)

    # ======================== Animation ========================
    def _start_animation(self):
        self._update_sensors()

    def _update_sensors(self):
        left_v = adc_to_voltage(self.ir_left)
        right_v = adc_to_voltage(self.ir_right)
        left_dist = voltage_to_distance_cm(left_v)
        right_dist = voltage_to_distance_cm(right_v)
        us_dist = float(self.us_dist)

        # Sensor labels
        self.lbl_left_adc.configure(text=str(self.ir_left))
        self.lbl_left_dist.configure(text=f"{left_dist:.1f} cm")
        self._draw_gauge(self.left_gauge, left_dist, 80, ACCENT_L)

        self.lbl_right_adc.configure(text=str(self.ir_right))
        self.lbl_right_dist.configure(text=f"{right_dist:.1f} cm")
        self._draw_gauge(self.right_gauge, right_dist, 80, ACCENT_R)

        self.bat_var.set(f"BAT: {self.bat_mv} mV")

        # IMU labels
        self.imu_labels["R"].configure(text=f"{self.roll:+07.1f}")
        self.imu_labels["P"].configure(text=f"{self.pitch:+07.1f}")
        self.imu_labels["Y"].configure(text=f"{self.yaw:+07.1f}")

        # 2D car + 3D IMU + GPS
        self._draw_car(left_dist, right_dist, us_dist)
        self._draw_3d()
        self._update_gps_display()

        self.root.after(100, self._update_sensors)

    def _draw_gauge(self, canvas, dist, max_dist, color):
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 1:
            return
        ratio = min(dist / max_dist, 1.0)
        bar_w = int(w * ratio)
        bar_color = DANGER_COLOR if dist < 20 else (WARN_COLOR if dist < 40 else color)
        canvas.create_rectangle(0, 0, w, h, fill=GRID_COLOR, outline="")
        if bar_w > 0:
            canvas.create_rectangle(0, 0, bar_w, h, fill=bar_color, outline="")

    def _draw_3d(self):
        ax = self.ax3d
        ax.cla()
        ax.set_facecolor(CHART_BG)
        R = rotation_matrix(self.roll, self.pitch, self.yaw)

        # Car body
        rotated = (R @ CAR_BODY.T).T
        faces = [[rotated[v] for v in f] for f in CAR_FACES]
        poly = Poly3DCollection(faces, facecolors=CAR_COLORS,
                                edgecolors="#ffffff40", linewidths=0.8)
        ax.add_collection3d(poly)

        # Wheels
        for wh in WHEELS:
            rw = (R @ wh.T).T
            wf = [[rw[v] for v in f] for f in CAR_FACES]
            wp = Poly3DCollection(wf, facecolors=[(0.2,0.2,0.2,0.9)]*6,
                                  edgecolors="#333333", linewidths=0.5)
            ax.add_collection3d(wp)

        # Nose arrow (front direction)
        nose = R @ np.array([0, 2.8, 0])
        ax.quiver(0,0,0, nose[0],nose[1],nose[2], color=ACCENT_U,
                  arrow_length_ratio=0.15, linewidth=2.5)

        # Axis reference
        al = 3.0
        ax.quiver(0,0,0, al,0,0, color="#ff444444", arrow_length_ratio=0.08, linewidth=0.8)
        ax.quiver(0,0,0, 0,al,0, color="#44ff4444", arrow_length_ratio=0.08, linewidth=0.8)
        ax.quiver(0,0,0, 0,0,al, color="#4444ff44", arrow_length_ratio=0.08, linewidth=0.8)

        # Grid
        grid_pts = np.linspace(-3, 3, 7)
        for g in grid_pts:
            ax.plot([g,g],[-3,3],[-3], color=GRID_COLOR, linewidth=0.3, alpha=0.3)
            ax.plot([-3,3],[g,g],[-3], color=GRID_COLOR, linewidth=0.3, alpha=0.3)

        lim = 3.5
        ax.set_xlim([-lim,lim]); ax.set_ylim([-lim,lim]); ax.set_zlim([-lim,lim])
        ax.set_box_aspect([1,1,1])
        ax.set_axis_off()
        self.canvas3d.draw_idle()

    # ======================== Drive Mode ========================
    def _on_drive_mode_change(self, event=None):
        new_mode = self.drive_combo.get()
        old_mode = self.drive_mode

        if new_mode == old_mode:
            self.root.focus_set()
            return

        # 차량 정지
        self._send_move(0)

        # 이전 모드 정리
        if old_mode == "CAT_TRACK":
            self.cat_track_running = False
        elif old_mode == "GPS_RETURN":
            self.gps_returning = False
            self.btn_return.configure(bg="#45475a", fg=WARN_COLOR)

        self.drive_mode = new_mode

        # 새 모드 진입
        if new_mode == "MANUAL":
            self.detect_combo.configure(state="readonly")
            self.cmd_var.set("MODE: MANUAL")

        elif new_mode == "CAT_TRACK":
            self.detect_combo.set("cat_custom")
            self.detect_combo.configure(state="disabled")
            self._on_detect_mode_change()
            self.cat_track_running = True
            self.cat_no_detect_count = 0
            threading.Thread(target=self._cat_track_loop, daemon=True).start()
            self.cmd_var.set("MODE: CAT_TRACK")

        elif new_mode == "GPS_RETURN":
            self.detect_combo.configure(state="disabled")
            if self.gps_home is None:
                self.cmd_var.set("GPS: Home not set")
                self.drive_mode = "MANUAL"
                self.drive_combo.set("MANUAL")
                self.detect_combo.configure(state="readonly")
            else:
                self.gps_returning = True
                self.btn_return.configure(bg=WARN_COLOR, fg="#1e1e2e")
                threading.Thread(target=self._gps_return_loop, daemon=True).start()
                self.cmd_var.set("MODE: GPS_RETURN")

        self.root.focus_set()

    def _cat_track_loop(self):
        """CAT_TRACK: /detect 폴링 → 바운딩 박스 기반 자동 조향"""
        import json

        host = self.entry_host.get().strip()
        url = f"http://{host}:{CAM_PORT}/detect"

        while self.cat_track_running and self.connected:
            try:
                resp = urllib.request.urlopen(url, timeout=1)
                data = json.loads(resp.read().decode('utf-8'))
            except Exception:
                time.sleep(0.2)
                continue

            box = data.get("box")
            frame_w = data.get("frame_w", 640)
            frame_h = data.get("frame_h", 480)

            if box is None:
                self.cat_no_detect_count += 1
                if self.cat_no_detect_count > 5:
                    self._send_move(0)  # target lost → STOP
                    self.root.after(0, lambda: self.cmd_var.set("CAT: LOST"))
                time.sleep(0.15)
                continue

            self.cat_no_detect_count = 0
            x1, y1, x2, y2 = box
            box_cx = (x1 + x2) / 2.0
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            frame_area = frame_w * frame_h

            # 수평 오프셋: -1.0 (왼쪽) ~ +1.0 (오른쪽)
            offset_x = (box_cx - frame_w / 2.0) / (frame_w / 2.0)
            area_ratio = box_area / frame_area

            TURN_THRESHOLD = 0.25
            CLOSE_THRESHOLD = 0.25

            if area_ratio > CLOSE_THRESHOLD:
                self._send_move(0)  # 너무 가까움 → STOP
                self.root.after(0, lambda: self.cmd_var.set("CAT: CLOSE"))
            elif offset_x < -TURN_THRESHOLD:
                self._send_move(3)  # LEFT
                self.root.after(0, lambda: self.cmd_var.set("CAT: LEFT"))
            elif offset_x > TURN_THRESHOLD:
                self._send_move(4)  # RIGHT
                self.root.after(0, lambda: self.cmd_var.set("CAT: RIGHT"))
            else:
                self._send_move(1)  # FORWARD
                self.root.after(0, lambda: self.cmd_var.set("CAT: FORWARD"))

            time.sleep(0.15)

    # ======================== GPS ========================
    def _gps_set_home(self):
        if self.gps_lat != 0.0 and self.gps_lon != 0.0:
            self.gps_home = (self.gps_lat, self.gps_lon)
            self.gps_home_var.set(f"HOME: {self.gps_lat:.4f},{self.gps_lon:.4f}")
            self.cmd_var.set("GPS: Home set")

    def _gps_return_home(self):
        if self.gps_home is None:
            self.cmd_var.set("GPS: Home not set")
            return
        self.gps_returning = not self.gps_returning
        if self.gps_returning:
            self.btn_return.configure(bg=WARN_COLOR, fg=BG_COLOR)
            self.cmd_var.set("GPS: Returning home...")
            threading.Thread(target=self._gps_return_loop, daemon=True).start()
        else:
            self.btn_return.configure(bg="#45475a", fg=WARN_COLOR)
            self.cmd_var.set("GPS: Return cancelled")
            self._send_move(0)

    def _gps_distance(self, lat1, lon1, lat2, lon2):
        """두 GPS 좌표 간 거리 (미터)"""
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon/2)**2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def _gps_bearing(self, lat1, lon1, lat2, lon2):
        """현재→목표 방위각 (도, 0=북)"""
        dlon = math.radians(lon2 - lon1)
        lat1r = math.radians(lat1)
        lat2r = math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2r)
        y = (math.cos(lat1r) * math.sin(lat2r) -
             math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def _gps_return_loop(self):
        """귀환 루프: GPS + IMU Yaw로 Home까지 이동"""
        while self.gps_returning and self.connected:
            if self.gps_lat == 0.0 or self.gps_home is None:
                time.sleep(0.5)
                continue

            dist = self._gps_distance(self.gps_lat, self.gps_lon,
                                       self.gps_home[0], self.gps_home[1])

            # 도착 (3m 이내)
            if dist < 3.0:
                self._send_move(0)
                self.gps_returning = False
                self.root.after(0, lambda: self.btn_return.configure(
                    bg="#45475a", fg=WARN_COLOR))
                self.root.after(0, lambda: self.cmd_var.set(
                    f"GPS: Home reached ({dist:.1f}m)"))
                # 자동으로 MANUAL 모드 복귀
                self.drive_mode = "MANUAL"
                self.root.after(0, lambda: self.drive_combo.set("MANUAL"))
                self.root.after(0, lambda: self.detect_combo.configure(state="readonly"))
                break

            # 목표 방위각 vs 현재 Yaw
            target_bearing = self._gps_bearing(
                self.gps_lat, self.gps_lon,
                self.gps_home[0], self.gps_home[1])
            yaw = self.yaw % 360
            diff = (target_bearing - yaw + 360) % 360

            # 방향 결정
            if diff > 30 and diff < 330:
                if diff < 180:
                    self._send_move(4)  # RIGHT
                else:
                    self._send_move(3)  # LEFT
            else:
                self._send_move(1)  # FORWARD

            time.sleep(0.3)

    def _update_gps_display(self):
        """GPS UI 업데이트"""
        self.gps_lat_var.set(f"LAT:{self.gps_lat:.6f}")
        self.gps_lon_var.set(f"LON:{self.gps_lon:.6f}")
        self.gps_speed_var.set(f"{self.gps_speed:.1f} km/h")
        self.gps_sats_var.set(f"SAT:{self.gps_sats}")

        # Home까지 거리
        if self.gps_home and self.gps_lat != 0.0:
            dist = self._gps_distance(self.gps_lat, self.gps_lon,
                                       self.gps_home[0], self.gps_home[1])
            self.gps_dist_var.set(f"→HOME: {dist:.1f}m")
        else:
            self.gps_dist_var.set("")

        # 경로 그리기
        self._draw_gps_track()

    def _draw_gps_track(self):
        """GPS 이동 경로 캔버스에 그리기"""
        c = self.gps_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1 or not self.gps_track:
            c.create_text(w//2, h//2, text="Waiting for GPS signal...",
                          fill="#6c7086", font=("Consolas", 10))
            return

        # 경로 범위 계산
        lats = [p[0] for p in self.gps_track]
        lons = [p[1] for p in self.gps_track]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        # 최소 범위 확보 (너무 작으면 확대 과도)
        lat_range = max(lat_max - lat_min, 0.0001)
        lon_range = max(lon_max - lon_min, 0.0001)
        margin = 0.1  # 10% 여백

        def to_pixel(lat, lon):
            x = int((lon - lon_min) / lon_range * (w * (1 - 2*margin)) + w * margin)
            y = int((1 - (lat - lat_min) / lat_range) * (h * (1 - 2*margin)) + h * margin)
            return x, y

        # 경로 선 그리기
        if len(self.gps_track) >= 2:
            points = [to_pixel(p[0], p[1]) for p in self.gps_track]
            for i in range(1, len(points)):
                c.create_line(points[i-1][0], points[i-1][1],
                              points[i][0], points[i][1],
                              fill=ACCENT_U, width=2)

        # 현재 위치 (빨간 점)
        cx, cy = to_pixel(self.gps_lat, self.gps_lon)
        c.create_oval(cx-5, cy-5, cx+5, cy+5, fill="#f38ba8", outline="")
        c.create_text(cx, cy-12, text="NOW", fill="#f38ba8",
                      font=("Consolas", 8, "bold"))

        # Home 위치 (초록 점)
        if self.gps_home:
            hx, hy = to_pixel(self.gps_home[0], self.gps_home[1])
            c.create_oval(hx-5, hy-5, hx+5, hy+5, fill=ACCENT_U, outline="")
            c.create_text(hx, hy-12, text="HOME", fill=ACCENT_U,
                          font=("Consolas", 8, "bold"))

    # ======================== Cleanup ========================
    def on_close(self):
        self._disconnect()
        self.root.destroy()


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST

    root = tk.Tk()
    app = ControlPanel(root, host)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
