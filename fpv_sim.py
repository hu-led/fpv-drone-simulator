"""
FPV 四轴飞行器仿真模拟工具 (raylib版)
- 键盘 / 串口 / 滤波IMU 三种输入源
- FPV第一人称 + 第三人称(CAMERA_ORBITAL)双视角
- 虚拟IMU数据面板（加速度计+陀螺仪）
- 互补滤波姿态解算演示模式

环境要求: pip install raylib (1.7MB)
"""
import math
import time
import random
from raylib import *
import raylib._raylib_cffi as _cffi
_ffi = _cffi.ffi

def _T(text):
    """Convert Python str to bytes for raylib CFFI."""
    return text.encode('utf-8')

# ============================================================
# 可选串口模块
# ============================================================
try:
    import serial as pyserial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ============================================================
# 配置常量
# ============================================================
WINDOW_W, WINDOW_H = 1280, 720
SERIAL_PORT = 'COM6'
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 0.1
SERIAL_IMU_PREFIX = 'I:'
SERIAL_AUTO_RECONNECT = True
SERIAL_RECONNECT_DELAY = 2.0
GRAVITY = 9.81
FILTER_ALPHA = 0.98
ARM_LENGTH = 0.55

# IMU/滤波
YAW_DEADBAND_DPS = 1.5
IMU_INT_SCALE = 0.01
DT_DEFAULT = 0.016
DT_MAX = 0.5

# 键盘操控
ANGLE_RATE = 60.0
HOVER_THROTTLE = 30.0
THROTTLE_RATE_SPACE = 120.0
THROTTLE_RATE_UP = 50.0
THROTTLE_RATE_DOWN = 60.0
THROTTLE_DECAY = 8.0
ALT_DAMPING = 0.8

# 高度
ALT_MIN = 0.1
ALT_MAX = 10.0

# 高度跟踪（真实IMU）
ALT_BASELINE_SAMPLES = 50
ALT_DEADBAND = 0.3
ALT_TRACK_DAMPING = 0.5
ALT_RETURN_VEL_THRESH = 0.15
ALT_RETURN_OFFSET_THRESH = 0.02
ALT_RETURN_KP = 0.15

# 电机指令
MOTOR_SEND_INTERVAL = 0.05

# 手控IMU模式
MANUAL_IMU_GAIN = 1.8
MANUAL_IMU_THROTTLE = 30.0

# 混控
MIXER_MAX = 80.0

# 中文字体路径（Windows 系统自带）
_CHINESE_FONT_PATHS = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
]
# HUD 需要用到的所有中文字符
_HUD_CHINESE_CHARS = (
    "滚转俯仰偏航高度姿态角真实虚拟加速度计陀螺仪"
    "无噪声低高滤波演示键盘串口操作提示上下左右升降空格升高重置"
    "模式等级开关大小中调试"
    "连接失败已断开未检测到端口切换至"
    "手控"
    "退出"
)
# 还需要 ASCII 基础字符
_HUD_ALL_CHARS = _HUD_CHINESE_CHARS + (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789.:+-/%=()[]<>|_  "
)

# 自定义颜色 - 深色主题
BG_COLOR = (18, 22, 28, 255)           # 深灰背景（天空）
GROUND_COLOR = (35, 40, 50, 255)        # 深灰地面
GRID_COLOR = (50, 55, 68, 255)         # 网格线
BODY_COLOR = (255, 60, 60, 255)        # 机身亮红
NOSE_COLOR = (255, 220, 0, 255)        # 机头金黄
ARM_FRONT_COLOR = (220, 80, 80, 255)   # 前臂浅红
ARM_BACK_COLOR = (160, 165, 175, 255)  # 后臂浅灰
MOTOR_COLOR = (70, 72, 80, 255)        # 电机深灰
PAD_COLOR = (100, 105, 115, 255)       # 起降平台
HUD_MAIN_COLOR = (240, 240, 245, 255)  # HUD 主文字白
HUD_IMU_COLOR = (100, 220, 180, 255)   # IMU 面板青绿
HUD_MODE_COLOR = (255, 200, 70, 255)   # 模式状态金黄
HUD_HINT_COLOR = (140, 145, 155, 255)  # 提示文字灰
HUD_TITLE_COLOR = (80, 190, 230, 255)  # 标题蓝


# ============================================================
# 欧拉角 → 方向向量
# 旋转顺序: Y(yaw) → X(pitch) → Z(roll), 对应矩阵 Rz*Rx*Ry
# ============================================================
def compute_vectors(roll_deg, pitch_deg, yaw_deg):
    cr = math.cos(math.radians(roll_deg))
    sr = math.sin(math.radians(roll_deg))
    cp = math.cos(math.radians(pitch_deg))
    sp = math.sin(math.radians(pitch_deg))
    cy = math.cos(math.radians(yaw_deg))
    sy = math.sin(math.radians(yaw_deg))

    # Forward = 第三列
    fx = cr*sy + sr*sp*cy
    fy = sr*sy - cr*sp*cy
    fz = cp*cy
    # Up = 第二列
    ux = -sr*cp
    uy = cr*cp
    uz = sp

    return (fx, fy, fz), (ux, uy, uz)


# ============================================================
# 虚拟IMU数据生成
# ============================================================
def virtual_imu(roll_deg, pitch_deg, yaw_deg,
                prev_roll, prev_pitch, prev_yaw, dt_val, noise=0):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    cr = math.cos(r)
    cp = math.cos(p)
    sr = math.sin(r)
    sp = math.sin(p)

    ax = GRAVITY * sp
    ay = -GRAVITY * sr * cp
    az = -GRAVITY * cr * cp

    if dt_val > 0.0001:
        gx = math.radians(roll_deg - prev_roll) / dt_val
        gy = math.radians(pitch_deg - prev_pitch) / dt_val
        gz = math.radians(yaw_deg - prev_yaw) / dt_val
    else:
        gx = gy = gz = 0.0

    if noise >= 1:
        an = 0.4 + noise * 0.6
        ax += random.gauss(0, an)
        ay += random.gauss(0, an)
        az += random.gauss(0, an)
        gn = 0.03 + noise * 0.04
        gx += random.gauss(0, gn)
        gy += random.gauss(0, gn)
        gz += random.gauss(0, gn)

    return {'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz}


# ============================================================
# 加速度计 → 姿态角
# ============================================================
def accel_to_angles(ax, ay, az):
    roll = math.degrees(math.atan2(-ay, -az))
    pitch = math.degrees(math.atan2(ax, math.sqrt(ay*ay + az*az)))
    return roll, pitch


# ============================================================
# 互补滤波器
# ============================================================
class ComplementaryFilter:
    def __init__(self, alpha=FILTER_ALPHA):
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def update(self, ax, ay, az, gx, gy, gz, dt_val):
        a_roll, a_pitch = accel_to_angles(ax, ay, az)
        g_roll = self.roll + math.degrees(gx) * dt_val
        g_pitch = self.pitch + math.degrees(gy) * dt_val
        # 偏航死区：低于 1.5 deg/s 视作零漂，不积分
        gz_dps = math.degrees(gz)
        if abs(gz_dps) < YAW_DEADBAND_DPS:
            gz_dps = 0.0
        g_yaw = self.yaw + gz_dps * dt_val

        self.roll = self.alpha * g_roll + (1 - self.alpha) * a_roll
        self.pitch = self.alpha * g_pitch + (1 - self.alpha) * a_pitch
        self.yaw = g_yaw

        return self.roll, self.pitch, self.yaw, a_roll, a_pitch

    def reset(self):
        self.roll = self.pitch = self.yaw = 0.0


# ============================================================
# 串口输入源
# ============================================================
class SerialInput:
    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD):
        self.roll = self.pitch = self.yaw = 0.0
        self.alt = 1.0
        self.port = port
        self.baud = baud
        self._conn = None
        self._buffer = ""
        self.mode = None
        self.raw_imu = None
        self.filter = ComplementaryFilter()
        self.prev_parse_time = 0.0
        self.last_connect_attempt = 0.0

    def connect(self):
        if not HAS_SERIAL:
            return False
        self.disconnect()
        try:
            self._conn = pyserial.Serial(self.port, self.baud, timeout=SERIAL_TIMEOUT)
            self._buffer = ""
            self.mode = None
            self.raw_imu = None
            self.filter.reset()
            self.prev_parse_time = time.time()
            self.last_connect_attempt = time.time()
            return True
        except Exception:
            self._conn = None
            return False

    def disconnect(self):
        if self._conn:
            try: self._conn.close()
            except: pass
        self._conn = None

    def send_cmd(self, data):
        """发送字符串到串口（电机指令等）"""
        if self.connected:
            try:
                self._conn.write(data.encode('utf-8'))
                return True
            except Exception:
                self.disconnect()
        return False

    @property
    def connected(self):
        return self._conn is not None and self._conn.is_open

    def _parse_line(self, line):
        if not line:
            return False

        if line.startswith(SERIAL_IMU_PREFIX):
            parts = line[len(SERIAL_IMU_PREFIX):].split(',')
            if len(parts) < 6:
                return False
            try:
                v = [int(p) for p in parts[:6]]
                ax, ay, az = v[0] * IMU_INT_SCALE, v[1] * IMU_INT_SCALE, v[2] * IMU_INT_SCALE
                gx_dps, gy_dps, gz_dps = v[3] * IMU_INT_SCALE, v[4] * IMU_INT_SCALE, v[5] * IMU_INT_SCALE
            except ValueError:
                try:
                    ax = float(parts[0]); ay = float(parts[1]); az = float(parts[2])
                    gx_dps = float(parts[3]); gy_dps = float(parts[4]); gz_dps = float(parts[5])
                except ValueError:
                    return False

            now = time.time()
            dt_val = now - self.prev_parse_time
            self.prev_parse_time = now
            if dt_val <= 0:
                dt_val = DT_DEFAULT
            if dt_val > DT_MAX:
                dt_val = DT_DEFAULT

            gx_rad = math.radians(gx_dps)
            gy_rad = math.radians(gy_dps)
            gz_rad = math.radians(gz_dps)

            self.raw_imu = {'ax': ax, 'ay': ay, 'az': az,
                            'gx': gx_rad, 'gy': gy_rad, 'gz': gz_rad}

            self.filter.update(ax, ay, az, gx_rad, gy_rad, gz_rad, dt_val)
            self.roll = self.filter.roll
            self.pitch = self.filter.pitch
            self.yaw = self.filter.yaw
            self.mode = "imu_raw"
            return True
        else:
            parts = line.split(',')
            if len(parts) < 3:
                return False
            try:
                self.roll = float(parts[0])
                self.pitch = float(parts[1])
                self.yaw = float(parts[2])
                self.alt = float(parts[3]) if len(parts) >= 4 else 1.0
            except ValueError:
                return False
            self.mode = "euler"
            self.raw_imu = None
            return True

    def read_line(self):
        if not self.connected:
            return False
        got_any = False
        try:
            while self._conn.in_waiting > 0:
                b = self._conn.read().decode('utf-8', errors='ignore')
                if b == '\n':
                    line = self._buffer.strip()
                    self._buffer = ""
                    if line:
                        ok = self._parse_line(line)
                        if not ok and self.mode is None:
                            print(f"[Serial] 未识别: {line[:60]}")
                        if ok:
                            got_any = True
                else:
                    self._buffer += b
        except Exception:
            self.disconnect()
        return got_any

    def try_reconnect(self):
        if not SERIAL_AUTO_RECONNECT:
            return False
        now = time.time()
        if now - self.last_connect_attempt < SERIAL_RECONNECT_DELAY:
            return False
        self.last_connect_attempt = now
        return self.connect()


# ============================================================
# 滤波IMU输入源
# ============================================================
class FilteredIMUInput:
    def __init__(self):
        self.target_roll = self.target_pitch = self.target_yaw = 0.0
        self.alt = 1.0
        self.prev_roll = self.prev_pitch = self.prev_yaw = 0.0
        self.filter = ComplementaryFilter()
        self.raw_imu = None
        self.accel_roll = self.accel_pitch = 0.0
        self.last_time = time.time()

    def set_target(self, r, p, y, a):
        self.target_roll, self.target_pitch, self.target_yaw, self.alt = r, p, y, a

    def update(self, noise=0):
        now = time.time()
        dt_val = now - self.last_time
        self.last_time = now
        if dt_val <= 0:
            dt_val = DT_DEFAULT

        imu = virtual_imu(self.target_roll, self.target_pitch, self.target_yaw,
                          self.prev_roll, self.prev_pitch, self.prev_yaw,
                          dt_val, noise=noise)
        self.raw_imu = imu
        f_roll, f_pitch, f_yaw, self.accel_roll, self.accel_pitch = \
            self.filter.update(imu['ax'], imu['ay'], imu['az'],
                               imu['gx'], imu['gy'], imu['gz'], dt_val)
        self.prev_roll, self.prev_pitch, self.prev_yaw = f_roll, f_pitch, f_yaw
        return f_roll, f_pitch, f_yaw

    def reset(self):
        self.filter.reset()
        self.prev_roll = self.prev_pitch = self.prev_yaw = 0.0
        self.last_time = time.time()


# ============================================================
# 障碍物数据
# ============================================================
OBSTACLES = [
    ( 1.5,  2.0, 0.15, 1.2, (200, 120, 60, 255)),
    ( 3.0, -2.5, 0.12, 0.9, (180, 150, 70, 255)),
    (-2.0,  3.0, 0.18, 1.5, (110, 120, 180, 255)),
    (-3.5, -1.5, 0.14, 1.0, (170, 100, 80, 255)),
    ( 2.5, -3.5, 0.10, 1.8, (100, 130, 170, 255)),
    (-1.0, -3.0, 0.20, 0.8, (190, 130, 80, 255)),
    ( 4.0,  1.0, 0.13, 1.3, (140, 130, 160, 255)),
    ( 0.0, -4.0, 0.16, 1.1, (160, 110, 90, 255)),
]


# ============================================================
# 3D场景绘制
# ============================================================
def draw_scene():
    # 地面
    DrawPlane((0, 0, 0), (GRID, GRID), GROUND_COLOR)
    DrawGrid(GRID, 1)

    # 障碍物
    for ox, oz, r, h, color in OBSTACLES:
        DrawCylinder((ox, h * 0.5, oz), r, r, h, 16, color)

    # 起降平台
    DrawCylinder((0, 0.02, 0), 0.50, 0.50, 0.04, 24, PAD_COLOR)


def draw_drone(altitude, roll_deg, pitch_deg, yaw_deg):
    rlPushMatrix()
    rlTranslatef(0, altitude, 0)
    rlRotatef(roll_deg, 0, 0, 1)
    rlRotatef(pitch_deg, 1, 0, 0)
    rlRotatef(yaw_deg, 0, 1, 0)

    # 机身中心块
    DrawCube((0, 0.04, 0), 0.30, 0.05, 0.12, BODY_COLOR)
    # 机头顶部小球（比原来方块小，不遮挡 FPV 视线）
    DrawSphere((0, 0.08, 0), 0.04, (0, 200, 255, 255))

    # 机臂 + 电机
    for idx, ang_deg in enumerate([45, 135, 225, 315]):
        a = math.radians(ang_deg)
        dx = ARM_LENGTH * math.cos(a)
        dz = ARM_LENGTH * math.sin(a)

        arm_color = ARM_FRONT_COLOR if idx < 2 else ARM_BACK_COLOR
        DrawCylinderEx(
            (0, 0.04, 0), (dx, 0.04, dz),
            0.022, 0.022, 8, arm_color
        )
        DrawSphere((dx, 0.04, dz), 0.06, MOTOR_COLOR)

    rlPopMatrix()


# ============================================================
# HUD 绘制
# ============================================================
def draw_hud_dash(r, p, y, a, imu, src_label, flt_demo, noise_lvl, debug,
                  font, font_big, is_real_imu=False,
                  status_msg="", status_msg_time=0.0):
    SW = GetScreenWidth()
    SH = GetScreenHeight()
    scale = min(SW / 1280.0, SH / 720.0)

    L = int(16 * scale)
    FS = int(24 * scale)
    FS_BIG = int(30 * scale)
    FS_HINT = int(17 * scale)
    LINE_H = int(30 * scale)

    use_ch = font is not None
    F = font if font else GetFontDefault()
    FB = font_big if font_big else F

    # ── 左上：姿态角 ──
    y_pos = int(14 * scale)
    title = "[ Attitude ]" if not use_ch else "[ 姿态角 ]"
    DrawTextEx(FB, _T(title), (L, y_pos), FS_BIG, 1, HUD_TITLE_COLOR)
    y_pos += int(36 * scale)

    labels = [
        ("滚转 Roll  ", r, "deg"),
        ("俯仰 Pitch  ", p, "deg"),
        ("偏航 Yaw   ", y, "deg"),
        ("高度 Alt  ", a, "m"),
    ]
    for label, val, unit in labels:
        DrawTextEx(F, _T(f"{label} {val:+7.2f} {unit}"), (L, y_pos), FS, 1, HUD_MAIN_COLOR)
        y_pos += LINE_H

    # ── 右上：IMU ──
    rx = SW - int(350 * scale)
    ry = int(14 * scale)
    if is_real_imu:
        title2 = "[ Real IMU ]" if not use_ch else "[ 真实IMU ]"
    else:
        title2 = "[ Virtual IMU ]" if not use_ch else "[ 虚拟IMU ]"
    DrawTextEx(FB, _T(title2), (rx, ry), FS_BIG, 1, HUD_TITLE_COLOR)
    ry += int(36 * scale)

    DrawTextEx(F, _T(f"Accel(g)  X:{imu['ax']:+6.2f}  Y:{imu['ay']:+6.2f}  Z:{imu['az']:+6.2f}"),
               (rx, ry), FS, 1, HUD_IMU_COLOR)
    ry += LINE_H
    DrawTextEx(F, _T(f"Gyro(d/s) X:{math.degrees(imu['gx']):+6.1f}  Y:{math.degrees(imu['gy']):+6.1f}  Z:{math.degrees(imu['gz']):+6.1f}"),
               (rx, ry), FS, 1, HUD_IMU_COLOR)

    # ── 中上：模式状态 ──
    cx = SW // 2 - int(150 * scale)
    mode_text = f"[ {src_label} ]"
    if flt_demo:
        nl_names = ["NoNoise", "LowNoise", "HighNoise"] if not use_ch else \
                   ["无噪声", "低噪声", "高噪声"]
        mode_text += f"  [{nl_names[noise_lvl]}]"
    DrawTextEx(F, _T(mode_text), (cx, int(14 * scale)), FS, 1, HUD_MODE_COLOR)

    # ── 状态消息（3秒超时）──
    if status_msg and time.time() - status_msg_time < 3.0:
        status_color = (255, 120, 100, 255) if "失败" in status_msg or "断开" in status_msg or "fail" in status_msg.lower() or "fail" in status_msg.lower() else (100, 220, 140, 255)
        tw = int(360 * scale)
        DrawTextEx(F, _T(status_msg), (SW // 2 - tw // 2, int(48 * scale)), FS_HINT, 1, status_color)

    # ── 滤波调试信息（左下方，姿态面板之后）──
    if flt_demo and debug:
        y_pos += int(10 * scale)
        a_r, a_p, f_r, f_p = debug
        DrawTextEx(F, _T("-- Filter Debug --" if not use_ch else "-- 滤波调试 --"),
                   (L, y_pos), FS_HINT, 1, HUD_MODE_COLOR)
        y_pos += LINE_H
        DrawTextEx(F, _T(f"Accel ->  Roll {a_r:+7.2f} deg   Pitch {a_p:+7.2f} deg"),
                   (L, y_pos), FS, 1, HUD_MODE_COLOR)
        y_pos += LINE_H
        DrawTextEx(F, _T(f"Fusion ->  Roll {f_r:+7.2f} deg   Pitch {f_p:+7.2f} deg"),
                   (L, y_pos), FS, 1, HUD_MODE_COLOR)

    # ── 底部：操作提示 ──
    hint = ("W/S:Pitch A/D:Roll Q/E:Yaw Up/Down:Alt Space:Throttle | "
            "V:FPV F:Filter N:Noise 1:Key 2:Serial R:Reset")
    DrawTextEx(F, _T(hint), (L, SH - int(30 * scale)), FS_HINT, 1, HUD_HINT_COLOR)

    DrawTextEx(F, _T(f"FPS {GetFPS()}"), (SW - int(100 * scale), int(6 * scale)), FS_HINT, 1, HUD_HINT_COLOR)


# ============================================================
# 主函数
# ============================================================
def main():
    global GRID
    GRID = 12

    # ── 窗口初始化 ──
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE)
    InitWindow(WINDOW_W, WINDOW_H, b"FPV Quadcopter Simulator")
    SetTargetFPS(60)

    # ── 加载中文字体 ──
    hfont = None
    hfont_big = None
    use_chinese = False
    cp_list = [ord(c) for c in _HUD_ALL_CHARS]
    cp_array = _ffi.new('int[]', cp_list)
    cp_count = len(cp_list)

    for font_path in _CHINESE_FONT_PATHS:
        try:
            hfont = LoadFontEx(_T(font_path), 32, cp_array, cp_count)
            hfont_big = LoadFontEx(_T(font_path), 40, cp_array, cp_count)
            SetTextureFilter(hfont.texture, TEXTURE_FILTER_BILINEAR)
            SetTextureFilter(hfont_big.texture, TEXTURE_FILTER_BILINEAR)
            use_chinese = True
            print(f"[Font] Loaded: {font_path}")
            break
        except Exception as e:
            print(f"[Font] Failed {font_path}: {e}")
            pass

    if not use_chinese:
        print("[字体] 未找到中文字体，使用英文显示")

    # ── 固定视角相机 ──
    cam = _ffi.new('Camera3D *')
    cam.position = (6.0, 4.5, 7.0)
    cam.target = (0.0, 1.0, 0.0)
    cam.up = (0.0, 1.0, 0.0)
    cam.fovy = 60.0
    cam.projection = CAMERA_PERSPECTIVE

    # ── 状态 ──
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    altitude = ALT_MIN
    alt_velocity = 0.0
    alt_baseline_az = 0.0
    alt_baseline_samples = 0
    prev_roll = prev_pitch = prev_yaw = 0.0

    fpv_mode = False
    input_mode = "keyboard"
    filter_demo = False
    noise_level = 0
    serial_imu_mode = False
    status_msg = ""
    status_msg_time = 0.0
    throttle = 0.0
    last_motor_send = 0.0
    mode_m = False

    # ── 输入源实例 ──
    serial_src = SerialInput()
    filtered_src = FilteredIMUInput()

    print("=" * 50)
    print("  FPV Quadcopter Simulator (raylib)")
    print("  W/S Pitch | A/D Roll | Q/E Yaw")
    print("  Space/Up/Down Throttle (drives Alt)")
    print("  F FilterDemo | N Noise | R Reset")
    print("  1 Keyboard | 2 Serial")
    print("=" * 50)

    # ── 主循环 ──
    while not WindowShouldClose():
        dt = GetFrameTime()
        if dt <= 0.0:
            dt = DT_DEFAULT

        # ====================================================
        # 串口自动连接/重连
        # ====================================================
        if HAS_SERIAL and not serial_src.connected:
            serial_src.connect()

        # ====================================================
        # 输入处理
        # ====================================================
        if IsKeyPressed(KEY_ONE):
            input_mode = "keyboard"
            filter_demo = False
            serial_imu_mode = False
        if IsKeyPressed(KEY_TWO):
            if input_mode == "serial":
                input_mode = "keyboard"
                serial_imu_mode = False
                status_msg = "切至键盘模式" if use_chinese else "Switched to Keyboard"
                status_msg_time = time.time()
            elif HAS_SERIAL and serial_src.connected:
                input_mode = "serial"
                filter_demo = False
                serial_imu_mode = False
                serial_src.filter.reset()
                serial_src.prev_parse_time = time.time()
                status_msg = "串口IMU模式" if use_chinese else "Serial IMU Mode"
                status_msg_time = time.time()
            elif HAS_SERIAL and serial_src.connect():
                input_mode = "serial"
                filter_demo = False
                serial_imu_mode = False
                status_msg = "串口已连接" if use_chinese else "Serial Connected"
                status_msg_time = time.time()
                print("[Serial] Connected")
            else:
                reason = f"未检测到 {SERIAL_PORT}" if use_chinese else f"No {SERIAL_PORT} detected"
                status_msg = f"串口连接失败 — {reason}" if use_chinese else f"Connect failed — {reason}"
                status_msg_time = time.time()
                print("[Serial] Connect failed")
        if IsKeyPressed(KEY_V):
            fpv_mode = not fpv_mode
        if IsKeyPressed(KEY_F):
            filter_demo = not filter_demo
            if filter_demo:
                filtered_src.reset()
                filtered_src.set_target(roll, pitch, yaw, altitude)
            serial_src.disconnect()
            serial_imu_mode = False
        if IsKeyPressed(KEY_N):
            if filter_demo:
                noise_level = (noise_level + 1) % 3
        if IsKeyPressed(KEY_R):
            roll = pitch = yaw = 0.0
            throttle = 0.0
            altitude = ALT_MIN
            alt_velocity = 0.0
            alt_baseline_az = 0.0
            alt_baseline_samples = 0
            prev_roll = prev_pitch = prev_yaw = 0.0
            filtered_src.reset()
            serial_src.filter.reset()
            mode_m = False
        if IsKeyPressed(KEY_M):
            mode_m = not mode_m
            if mode_m:
                serial_src.filter.reset()
                serial_src.prev_parse_time = time.time()
                throttle = MANUAL_IMU_THROTTLE
                altitude = 1.0
                alt_baseline_az = 0.0
                alt_baseline_samples = 0
                alt_velocity = 0.0
                status_msg = "手控IMU模式" if use_chinese else "Manual IMU Mode"
            else:
                throttle = 0.0
                status_msg = "键盘模式" if use_chinese else "Keyboard Mode"
            status_msg_time = time.time()

        # --- 连续控制 ---
        if not mode_m and input_mode == "keyboard":

            if IsKeyDown(KEY_W):    pitch += ANGLE_RATE * dt
            if IsKeyDown(KEY_S):    pitch -= ANGLE_RATE * dt
            if IsKeyDown(KEY_A):    roll -= ANGLE_RATE * dt
            if IsKeyDown(KEY_D):    roll += ANGLE_RATE * dt
            if IsKeyDown(KEY_Q):    yaw += ANGLE_RATE * dt
            if IsKeyDown(KEY_E):    yaw -= ANGLE_RATE * dt

            roll %= 360; pitch %= 360; yaw %= 360
            if roll > 180:  roll -= 360
            if pitch > 180: pitch -= 360
            if yaw > 180:   yaw -= 360
            # 油门 → 高度物理模型
            net_accel = (throttle - HOVER_THROTTLE) / HOVER_THROTTLE * GRAVITY
            alt_velocity += net_accel * dt
            alt_velocity *= max(0.0, 1.0 - ALT_DAMPING * dt)
            altitude += alt_velocity * dt
            if altitude < ALT_MIN:
                altitude = ALT_MIN
                if alt_velocity < 0:
                    alt_velocity = 0
            altitude = min(altitude, ALT_MAX)

            # 油门：空格快速增加，↑↓ 微调，无操作缓慢衰减
            if IsKeyDown(KEY_SPACE):
                throttle = min(100.0, throttle + THROTTLE_RATE_SPACE * dt)
            elif IsKeyDown(KEY_UP):
                throttle = min(100.0, throttle + THROTTLE_RATE_UP * dt)
            elif IsKeyDown(KEY_DOWN):
                throttle = max(0.0, throttle - THROTTLE_RATE_DOWN * dt)
            else:
                if throttle > 0:
                    throttle = max(0.0, throttle - THROTTLE_DECAY * dt)

            state_roll, state_pitch, state_yaw, state_alt = roll, pitch, yaw, altitude
            src_label = "Keyboard" if not use_chinese else "键盘"
            raw_imu_data = virtual_imu(state_roll, state_pitch, state_yaw,
                                       prev_roll, prev_pitch, prev_yaw, dt)

        elif input_mode == "serial":
            serial_src.read_line()
            if not serial_src.connected:
                if not serial_src.try_reconnect():
                    status_msg = "串口已断开，切换至键盘模式" if use_chinese else "Disconnected, switched to keyboard"
                    status_msg_time = time.time()
                    print("[Serial] Disconnected, switch to keyboard")
                    input_mode = "keyboard"
                    serial_imu_mode = False
            state_roll = serial_src.roll
            state_pitch = serial_src.pitch
            state_yaw = serial_src.yaw
            state_alt = serial_src.alt

            if serial_src.mode == "imu_raw":
                serial_imu_mode = True
                src_label = "Serial(IMU)" if not use_chinese else "串口(IMU)"
                raw_imu_data = serial_src.raw_imu
            else:
                serial_imu_mode = False
                src_label = "Serial" if not use_chinese else "串口"
                raw_imu_data = virtual_imu(state_roll, state_pitch, state_yaw,
                                           prev_roll, prev_pitch, prev_yaw, dt)

        # ====================================================
        # 互补滤波演示
        # ====================================================
        debug_data = None
        if filter_demo and not mode_m:
            filtered_src.set_target(state_roll, state_pitch, state_yaw, state_alt)
            state_roll, state_pitch, state_yaw = filtered_src.update(noise=noise_level)
            raw_imu_data = filtered_src.raw_imu or raw_imu_data
            src_label = "FilterIMU" if not use_chinese else "滤波IMU"
            a_r = filtered_src.accel_roll
            a_p = filtered_src.accel_pitch
            f_r = filtered_src.filter.roll
            f_p = filtered_src.filter.pitch
            debug_data = (a_r, a_p, f_r, f_p)

        # ====================================================
        # 电机指令发送（键盘模式，20Hz）+ 串口IMU回显
        # ====================================================
        now = time.time()

        # ── 模式 M：手控 IMU → 电机反向修正 ──
        if mode_m:
            if serial_src.connected:
                serial_src.read_line()
                if not serial_src.connected:
                    mode_m = False
                    throttle = 0.0
                    status_msg = "串口断开，退出M模式" if use_chinese else "Serial lost, exit M mode"
                    status_msg_time = time.time()

                # 真实 IMU 角度 = 无人机姿态
                state_roll  = serial_src.roll
                state_pitch = serial_src.pitch
                state_yaw   = serial_src.yaw
                src_label   = "ManualIMU" if not use_chinese else "手控IMU"
                if serial_src.raw_imu:
                    raw_imu_data = serial_src.raw_imu
                    serial_imu_mode = True

                # 20Hz 发送电机指令（反向修正）
                if now - last_motor_send > MOTOR_SEND_INTERVAL:
                    last_motor_send = now
                    # 稳平：roll/pitch 偏离 0° 越大，反向修正越强
                    gain = MANUAL_IMU_GAIN
                    r = -state_roll * gain
                    p = -state_pitch * gain
                    y = 0.0
                    cmd = f"M:{int(throttle)},{int(r)},{int(p)},{int(y)}\r\n"
                    serial_src.send_cmd(cmd)
            else:
                mode_m = False
                throttle = 0.0
                status_msg = "串口未连接" if use_chinese else "Serial not connected"
                status_msg_time = time.time()

        elif input_mode == "keyboard":
            if serial_src.connected:
                # 读取真实 IMU 数据用于 HUD 显示
                serial_src.read_line()

                # 20Hz 发送电机混控指令
                if now - last_motor_send > MOTOR_SEND_INTERVAL:
                    last_motor_send = now
                    # 姿态角 → 混控修正量（±180° → ±80）
                    r = roll / 180.0 * MIXER_MAX
                    p = pitch / 180.0 * MIXER_MAX
                    # 偏航速率：Q/E 按下时 ±35
                    y = 0.0
                    if IsKeyDown(KEY_Q):
                        y = -35.0
                    elif IsKeyDown(KEY_E):
                        y = 35.0
                    cmd = f"M:{int(throttle)},{int(r)},{int(p)},{int(y)}\r\n"
                    serial_src.send_cmd(cmd)

        elif input_mode == "serial" and serial_src.connected:
            # 串口模式下也回传电机指令（throttle=0 停转）
            if now - last_motor_send > 0.05:
                last_motor_send = now
                serial_src.send_cmd("M:0,0,0,0\r\n")

        # ── 真实 IMU Z 加速度 → 高度跟踪（含倾斜补偿）──
        if (mode_m or serial_imu_mode) and raw_imu_data is not None:
            az = raw_imu_data.get('az', 0)
            if alt_baseline_samples < ALT_BASELINE_SAMPLES:
                alt_baseline_az = alt_baseline_az * 0.9 + az * 0.1
                alt_baseline_samples += 1
            else:
                cr = math.cos(math.radians(state_roll))
                cp = math.cos(math.radians(state_pitch))
                expected_az = alt_baseline_az * cr * cp
                net_accel = -(az - expected_az)
                if abs(net_accel) < ALT_DEADBAND:
                    net_accel = 0.0
                alt_velocity += net_accel * dt
                alt_velocity *= max(0.0, 1.0 - ALT_TRACK_DAMPING * dt)
                if abs(alt_velocity) < ALT_RETURN_VEL_THRESH and abs(altitude - 1.0) > ALT_RETURN_OFFSET_THRESH:
                    alt_velocity += (1.0 - altitude) * ALT_RETURN_KP * dt
                altitude += alt_velocity * dt
                if altitude < ALT_MIN:
                    altitude = ALT_MIN
                    if alt_velocity < 0:
                        alt_velocity = 0
                altitude = min(altitude, ALT_MAX)
            state_alt = altitude

        # ── 真实 IMU 显示覆盖 ──
        if not mode_m:
            if input_mode == "keyboard" and serial_src.connected and serial_src.raw_imu:
                raw_imu_data = serial_src.raw_imu
                serial_imu_mode = True
            else:
                serial_imu_mode = (input_mode == "serial" and serial_src.mode == "imu_raw")

        # ====================================================
        # 相机：V 键切换 FPV / 拉近固定视角
        # ====================================================
        if fpv_mode:
            fwd, upv = compute_vectors(state_roll, state_pitch, state_yaw)
            cam.position = (0 + fwd[0]*0.10 + upv[0]*0.04,
                            state_alt + fwd[1]*0.10 + upv[1]*0.04,
                            0 + fwd[2]*0.10 + upv[2]*0.04)
            cam.target = (cam.position.x + fwd[0]*8,
                          cam.position.y + fwd[1]*8,
                          cam.position.z + fwd[2]*8)
            cam.up = (upv[0], upv[1], upv[2])
        else:
            cam.target = (0, state_alt, 0)
            cam.position = (0.0, state_alt + 2.0, -4.0)
            cam.up = (0.0, 1.0, 0.0)

        # ====================================================
        # 渲染
        # ====================================================
        BeginDrawing()
        ClearBackground(BG_COLOR)

        BeginMode3D(cam[0])
        draw_scene()
        draw_drone(state_alt, state_roll, state_pitch, state_yaw)
        EndMode3D()

        draw_hud_dash(state_roll, state_pitch, state_yaw, state_alt,
                      raw_imu_data, src_label,
                      filter_demo, noise_level, debug_data,
                      hfont, hfont_big,
                      is_real_imu=(input_mode == "serial" and serial_imu_mode),
                      status_msg=status_msg, status_msg_time=status_msg_time)

        EndDrawing()

        # ── 保存状态 ──
        prev_roll, prev_pitch, prev_yaw = state_roll, state_pitch, state_yaw
        roll, pitch, yaw, altitude = state_roll, state_pitch, state_yaw, state_alt

    # ── 清理 ──
    if hfont:
        UnloadFont(hfont)
    if hfont_big:
        UnloadFont(hfont_big)
    serial_src.disconnect()
    CloseWindow()


if __name__ == "__main__":
    main()
