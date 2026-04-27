"""
RC Car Sensor Monitor (PC, WiFi)
RPi5의 TCP 소켓(기본 9001)에 접속하여 TC237 센서 데이터를 실시간 시각화.

프로토콜 (RPi5 uart_server.py 가 TC237 ASCLIN0에서 수신한 그대로 중계):
    "L:xxxx,R:xxxx,U:xxx\n"
        L = 왼쪽 IR ADC (0~4095)
        R = 오른쪽 IR ADC (0~4095)
        U = 전면 초음파 거리 (cm)

Usage:
    python sensor_monitor.py                      # 기본: 192.168.0.23:9001
    python sensor_monitor.py 192.168.0.23 9001    # IP/포트 지정
"""

import math
import socket
import sys
import threading
import time
from collections import deque

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

# --- Configuration ---
DEFAULT_HOST = "192.168.30.22"
DEFAULT_PORT = 9001
ADC_MAX = 4095
VAREF = 5.0
HISTORY_SIZE = 200

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


class SensorMonitorApp:
    def __init__(self, root, default_host, default_port):
        self.root = root
        self.root.title("RC Car Sensor Monitor (WiFi)")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1100x850")
        self.root.minsize(900, 750)

        self.sock = None
        self.running = False
        self.read_thread = None

        self.ir_left = 0
        self.ir_right = 0
        self.us_dist = 0
        self.parking_state = ""

        self.left_history = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
        self.right_history = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)
        self.us_history = deque([0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)

        self._build_ui(default_host, default_port)
        self._start_animation()

    def _build_ui(self, default_host, default_port):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=BG_COLOR)
        style.configure("Dark.TLabel", background=BG_COLOR, foreground=FG_COLOR,
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG_COLOR, foreground=ACCENT_L,
                        font=("Segoe UI", 13, "bold"))
        style.configure("TitleR.TLabel", background=BG_COLOR, foreground=ACCENT_R,
                        font=("Segoe UI", 13, "bold"))
        style.configure("TitleU.TLabel", background=BG_COLOR, foreground=ACCENT_U,
                        font=("Segoe UI", 13, "bold"))
        style.configure("Big.TLabel", background=BG_COLOR, foreground=FG_COLOR,
                        font=("Consolas", 32, "bold"))
        style.configure("Dist.TLabel", background=BG_COLOR, foreground=ACCENT_U,
                        font=("Consolas", 22, "bold"))
        style.configure("Stat.TLabel", background=BG_COLOR, foreground=FG_COLOR,
                        font=("Consolas", 12))

        # --- Top bar ---
        top = ttk.Frame(self.root, style="Dark.TFrame")
        top.pack(fill=tk.X, padx=15, pady=(10, 5))

        ttk.Label(top, text="RC Car Sensor Monitor",
                  style="Title.TLabel").pack(side=tk.LEFT)

        self.btn_connect = ttk.Button(top, text="Connect",
                                       command=self._toggle_connect)
        self.btn_connect.pack(side=tk.RIGHT, padx=(5, 0))

        self.entry_port = ttk.Entry(top, width=8)
        self.entry_port.insert(0, str(default_port))
        self.entry_port.pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Label(top, text="Port:", style="Dark.TLabel").pack(side=tk.RIGHT)

        self.entry_host = ttk.Entry(top, width=18)
        self.entry_host.insert(0, default_host)
        self.entry_host.pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Label(top, text="RPi5 IP:", style="Dark.TLabel").pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(top, textvariable=self.status_var, style="Dark.TLabel").pack(
            side=tk.RIGHT, padx=(15, 10))

        # --- Parking state bar ---
        park_bar = ttk.Frame(self.root, style="Dark.TFrame")
        park_bar.pack(fill=tk.X, padx=15, pady=(0, 3))
        style.configure("Park.TLabel", background=BG_COLOR, foreground=WARN_COLOR,
                        font=("Consolas", 12, "bold"))
        self.parking_var = tk.StringVar(value="")
        self.lbl_parking = ttk.Label(park_bar, textvariable=self.parking_var,
                                     style="Park.TLabel")
        self.lbl_parking.pack(side=tk.LEFT)

        # --- Chart ---
        chart_frame = ttk.Frame(self.root, style="Dark.TFrame")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.fig = Figure(figsize=(9, 3), dpi=100, facecolor=CHART_BG)
        self.ax = self.fig.add_subplot(111)
        self._setup_chart()

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Bottom: Left IR | Car Visual | Right IR ---
        bottom = ttk.Frame(self.root, style="Dark.TFrame")
        bottom.pack(fill=tk.BOTH, padx=15, pady=(5, 10))

        left_panel = ttk.Frame(bottom, style="Dark.TFrame", width=230)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        ttk.Label(left_panel, text="LEFT IR (AN1)", style="Title.TLabel").pack(pady=(5, 0))
        self.lbl_left_adc = ttk.Label(left_panel, text="0", style="Big.TLabel")
        self.lbl_left_adc.pack()
        self.lbl_left_volt = ttk.Label(left_panel, text="0.000 V", style="Stat.TLabel")
        self.lbl_left_volt.pack()
        self.lbl_left_dist = ttk.Label(left_panel, text="-- cm", style="Dist.TLabel")
        self.lbl_left_dist.pack(pady=(5, 0))
        self.left_gauge = tk.Canvas(left_panel, height=16, bg=CHART_BG,
                                     highlightthickness=0)
        self.left_gauge.pack(fill=tk.X, padx=15, pady=(8, 0))

        center_panel = ttk.Frame(bottom, style="Dark.TFrame")
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.car_canvas = tk.Canvas(center_panel, bg=CHART_BG, highlightthickness=0,
                                     height=320)
        self.car_canvas.pack(fill=tk.BOTH, expand=True)

        right_panel = ttk.Frame(bottom, style="Dark.TFrame", width=230)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        ttk.Label(right_panel, text="RIGHT IR (AN12)", style="TitleR.TLabel").pack(
            pady=(5, 0))
        self.lbl_right_adc = ttk.Label(right_panel, text="0", style="Big.TLabel")
        self.lbl_right_adc.pack()
        self.lbl_right_volt = ttk.Label(right_panel, text="0.000 V", style="Stat.TLabel")
        self.lbl_right_volt.pack()
        self.lbl_right_dist = ttk.Label(right_panel, text="-- cm", style="Dist.TLabel")
        self.lbl_right_dist.pack(pady=(5, 0))
        self.right_gauge = tk.Canvas(right_panel, height=16, bg=CHART_BG,
                                      highlightthickness=0)
        self.right_gauge.pack(fill=tk.X, padx=15, pady=(8, 0))

    def _setup_chart(self):
        self.ax.set_facecolor(CHART_BG)
        self.ax.set_xlim(0, HISTORY_SIZE)
        self.ax.set_ylim(0, 200)
        self.ax.set_ylabel("Distance (cm)", color=FG_COLOR, fontsize=10)
        self.ax.set_xlabel("Samples", color=FG_COLOR, fontsize=10)
        self.ax.tick_params(colors=FG_COLOR, labelsize=8)
        self.ax.grid(True, color=GRID_COLOR, alpha=0.5, linestyle="--")
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        self.ax.axhspan(0, 20, alpha=0.08, color=DANGER_COLOR)
        self.ax.axhline(y=20, color=DANGER_COLOR, alpha=0.3, linestyle="--", linewidth=1)

        self.line_left, = self.ax.plot([], [], color=ACCENT_L, linewidth=1.5,
                                        label="IR Left", alpha=0.9)
        self.line_right, = self.ax.plot([], [], color=ACCENT_R, linewidth=1.5,
                                         label="IR Right", alpha=0.9)
        self.line_us, = self.ax.plot([], [], color=ACCENT_U, linewidth=2.5,
                                      label="Ultrasonic", alpha=0.9)
        self.ax.legend(loc="upper right", facecolor=CHART_BG, edgecolor=GRID_COLOR,
                       labelcolor=FG_COLOR, fontsize=9)
        self.fig.tight_layout(pad=2)

    def _start_animation(self):
        self.ani = animation.FuncAnimation(self.fig, self._update_chart,
                                           interval=100, blit=False,
                                           cache_frame_data=False)

    def _update_chart(self, frame):
        left_v = adc_to_voltage(self.ir_left)
        right_v = adc_to_voltage(self.ir_right)
        left_dist = voltage_to_distance_cm(left_v)
        right_dist = voltage_to_distance_cm(right_v)
        us_dist = float(self.us_dist)

        self.left_history.append(left_dist)
        self.right_history.append(right_dist)
        self.us_history.append(us_dist)

        x = list(range(HISTORY_SIZE))
        self.line_left.set_data(x, list(self.left_history))
        self.line_right.set_data(x, list(self.right_history))
        self.line_us.set_data(x, list(self.us_history))

        self.lbl_left_adc.configure(text=str(self.ir_left))
        self.lbl_left_volt.configure(text=f"{left_v:.3f} V")
        self.lbl_left_dist.configure(text=f"{left_dist:.1f} cm")
        self._draw_gauge(self.left_gauge, left_dist, 80, ACCENT_L)

        self.lbl_right_adc.configure(text=str(self.ir_right))
        self.lbl_right_volt.configure(text=f"{right_v:.3f} V")
        self.lbl_right_dist.configure(text=f"{right_dist:.1f} cm")
        self._draw_gauge(self.right_gauge, right_dist, 80, ACCENT_R)

        self._draw_car(left_dist, right_dist, us_dist)

        # Update parking state display
        if self.parking_state:
            self.parking_var.set(f"PARKING: {self.parking_state}")
        else:
            self.parking_var.set("")

        self.canvas.draw_idle()
        return [self.line_left, self.line_right, self.line_us]

    def _draw_gauge(self, canvas, dist, max_dist, color):
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 1:
            return

        ratio = min(dist / max_dist, 1.0)
        bar_w = int(w * ratio)

        if dist < 20:
            bar_color = DANGER_COLOR
        elif dist < 40:
            bar_color = WARN_COLOR
        else:
            bar_color = color

        canvas.create_rectangle(0, 0, w, h, fill=GRID_COLOR, outline="")
        if bar_w > 0:
            canvas.create_rectangle(0, 0, bar_w, h, fill=bar_color, outline="")

    def _draw_car(self, left_dist, right_dist, us_dist):
        c = self.car_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1:
            return

        cx, cy = w // 2, h // 2 + 30

        car_w, car_h = 90, 130
        c.create_rectangle(cx - car_w // 2, cy - car_h // 2,
                           cx + car_w // 2, cy + car_h // 2,
                           fill="#45475a", outline="#585b70", width=2)
        c.create_text(cx, cy + 8, text="TC237", fill=FG_COLOR,
                      font=("Consolas", 12, "bold"))

        for dy in [-45, 45]:
            c.create_rectangle(cx - car_w // 2 - 10, cy + dy - 15,
                               cx - car_w // 2, cy + dy + 15,
                               fill="#6c7086", outline="")
            c.create_rectangle(cx + car_w // 2, cy + dy - 15,
                               cx + car_w // 2 + 10, cy + dy + 15,
                               fill="#6c7086", outline="")

        us_len = max(20, min(100, us_dist * 0.8))
        if us_dist < 20:
            us_color = DANGER_COLOR
        elif us_dist < 50:
            us_color = WARN_COLOR
        else:
            us_color = ACCENT_U

        front_y = cy - car_h // 2
        cone_w = 50
        c.create_polygon(
            cx - 20, front_y,
            cx + 20, front_y,
            cx + cone_w, front_y - us_len,
            cx - cone_w, front_y - us_len,
            fill="", outline=us_color, width=2
        )
        for i in range(3):
            alpha_offset = i * us_len // 4
            c.create_line(cx - 20 - alpha_offset // 2, front_y - alpha_offset,
                          cx + 20 + alpha_offset // 2, front_y - alpha_offset,
                          fill=us_color, width=1, dash=(4, 4))

        c.create_rectangle(cx - 22, front_y - 6, cx + 22, front_y + 3,
                           fill="#585b70", outline=us_color, width=1)

        c.create_text(cx, front_y - us_len - 18,
                      text=f"{us_dist:.0f} cm", fill=us_color,
                      font=("Consolas", 16, "bold"))
        c.create_text(cx, front_y - us_len - 38,
                      text="ULTRASONIC", fill=us_color,
                      font=("Segoe UI", 9))

        # IR sensors: mounted at front corners, pointing 45 degrees outward-forward
        ir_angle = math.radians(45)
        left_len = max(15, min(90, left_dist * 1.0))
        left_color = DANGER_COLOR if left_dist < 20 else (
            WARN_COLOR if left_dist < 40 else ACCENT_L)
        # Left IR: front-left corner, 45 deg forward-left
        lx0 = cx - car_w // 2
        ly0 = front_y
        lx1 = lx0 - left_len * math.sin(ir_angle)
        ly1 = ly0 - left_len * math.cos(ir_angle)
        spread = 15
        c.create_line(lx0, ly0, lx1 - spread * 0.3, ly1 - spread * 0.3,
                      fill=left_color, width=2, arrow=tk.LAST)
        c.create_line(lx0, ly0, lx1 + spread * 0.5, ly1 - spread * 0.5,
                      fill=left_color, width=2, arrow=tk.LAST)
        c.create_text(lx1 - 5, ly1 - 20,
                      text=f"{left_dist:.0f}cm", fill=left_color,
                      font=("Consolas", 13, "bold"))

        right_len = max(15, min(90, right_dist * 1.0))
        right_color = DANGER_COLOR if right_dist < 20 else (
            WARN_COLOR if right_dist < 40 else ACCENT_R)
        # Right IR: front-right corner, 45 deg forward-right
        rx0 = cx + car_w // 2
        ry0 = front_y
        rx1 = rx0 + right_len * math.sin(ir_angle)
        ry1 = ry0 - right_len * math.cos(ir_angle)
        c.create_line(rx0, ry0, rx1 + spread * 0.3, ry1 - spread * 0.3,
                      fill=right_color, width=2, arrow=tk.LAST)
        c.create_line(rx0, ry0, rx1 - spread * 0.5, ry1 - spread * 0.5,
                      fill=right_color, width=2, arrow=tk.LAST)
        c.create_text(rx1 + 5, ry1 - 20,
                      text=f"{right_dist:.0f}cm", fill=right_color,
                      font=("Consolas", 13, "bold"))

        c.create_text(cx, cy - car_h // 2 - 10, text="FRONT",
                      fill="#6c7086", font=("Segoe UI", 9))

    # ======================== TCP ========================
    def _toggle_connect(self):
        if self.running:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        host = self.entry_host.get().strip()
        try:
            port = int(self.entry_port.get().strip())
        except ValueError:
            messagebox.showwarning("Warning", "Port는 숫자여야 합니다.")
            return
        if not host:
            messagebox.showwarning("Warning", "RPi5 IP를 입력하세요.")
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)
            self.sock.connect((host, port))
            self.sock.settimeout(None)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot connect to {host}:{port}\n{e}")
            self.sock = None
            return

        self.running = True
        self.btn_connect.configure(text="Disconnect")
        self.status_var.set(f"Connected: {host}:{port}")
        self.read_thread = threading.Thread(target=self._read_socket, daemon=True)
        self.read_thread.start()

    def _disconnect(self):
        self.running = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self.btn_connect.configure(text="Connect")
        self.status_var.set("Disconnected")

    def _read_socket(self):
        buffer = ""
        while self.running and self.sock:
            try:
                raw = self.sock.recv(256)
                if not raw:
                    break
                buffer += raw.decode("ascii", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    # Parse "L:xxxx,R:xxxx,U:xxx"
                    if line.startswith("L:") and ",R:" in line and ",U:" in line:
                        try:
                            parts = line.split(",")
                            self.ir_left = int(parts[0][2:])
                            self.ir_right = int(parts[1][2:])
                            self.us_dist = int(parts[2][2:])
                        except (ValueError, IndexError):
                            pass
                    # Parse parking state "P:SEARCH", "P:REV_TURN", etc.
                    elif line.startswith("P:"):
                        self.parking_state = line[2:]
            except OSError:
                break

        # 연결 종료 시 UI 상태 업데이트
        self.running = False
        try:
            self.root.after(0, lambda: self.status_var.set("Disconnected"))
            self.root.after(0, lambda: self.btn_connect.configure(text="Connect"))
        except tk.TclError:
            pass

    def on_close(self):
        self._disconnect()
        self.root.destroy()


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

    root = tk.Tk()
    app = SensorMonitorApp(root, host, port)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
