#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PickBoxJoyconHandler - 使用evdev的版本（支持Joy-Con）
基于evdev库，不依赖pygame，可以正确检测Joy-Con
"""

import threading
import time
import numpy as np
import os
import platform
from evdev import InputDevice, ecodes
from loop_rate_limiters import RateLimiter

from ..handler import Handler


class PickBoxJoyconEvdevHandler(Handler):
    _name = "pick_box_joycon_evdev"
    

    def __init__(self, device_path=None):
        super().__init__()

        self._timestep = 0.01
        self._action = np.zeros(4)
        self._device_path = device_path or "/dev/input/event16"  # 默认路径

        # Joy-Con按钮映射（根据evtest输出）
        # 注意：Joy-Con (R)的物理按钮映射
        # 根据 read_joycon.py 的映射：
        # 304 (BTN_SOUTH) = A按钮（物理位置：下面）
        # 305 (BTN_EAST) = B按钮（物理位置：右边）
        # 但在实际使用中，可能需要根据实际设备调整
        self._button_map = {
            304: "A",      # BTN_SOUTH (物理A按钮，下面)
            305: "B",      # BTN_EAST (物理B按钮，右边)
            307: "X",      # BTN_NORTH
            308: "Y",      # BTN_WEST
            310: "L",      # BTN_TL
            311: "R",      # BTN_TR
            312: "ZL",     # BTN_TL2
            313: "ZR",     # BTN_TR2
            315: "+",      # BTN_START
            316: "Home",   # BTN_MODE
            318: "SR",     # BTN_THUMBR
        }

        # 轴映射
        self._axis_map = {
            3: "RX",  # ABS_RX (右摇杆X轴)
            4: "RY",  # ABS_RY (右摇杆Y轴)
        }

        # 初始化evdev设备
        self._device = None
        self._device_fd = None
        
        # 当前状态
        self._button_states = {}
        self._axis_values = {}
        
        # 初始化轴值
        self._axis_values[3] = 0  # RX
        self._axis_values[4] = 0  # RY
        
        # 用于跟踪action变化
        self._last_action = np.zeros(4)
        self._action_change_threshold = 1e-6

        # 自动检测Joy-Con设备
        if device_path and os.path.exists(device_path):
            # 使用指定的设备路径
            try:
                dev = InputDevice(device_path)
                # 验证不是IMU设备
                if "IMU" not in dev.name:
                    # 验证设备有按钮和摇杆功能
                    caps = dev.capabilities()
                    has_buttons = ecodes.EV_KEY in caps
                    has_axes = ecodes.EV_ABS in caps
                    if has_buttons and has_axes:
                        self._device = dev
                        self._device_fd = self._device.fd
                        print(f"✅ 成功打开设备: {dev.name}")
                        print(f"   路径: {device_path}")
                        print(f"   支持按键: {has_buttons}")
                        print(f"   支持轴: {has_axes}")
                    else:
                        print(f"⚠️  设备缺少必要功能: 按键={has_buttons}, 轴={has_axes}")
                        self._device = None
                else:
                    self._device = None
            except Exception as e:
                self._device = None
        
        # 如果指定路径失败，自动搜索Joy-Con（排除IMU设备）
        if self._device is None:
            try:
                from evdev import list_devices
                devices = list_devices()
                for dev_path in devices:
                    try:
                        dev = InputDevice(dev_path)
                        # 检查是否是Joy-Con，但排除IMU设备
                        if "Joy-Con" in dev.name and "IMU" not in dev.name:
                            # 验证设备有按钮和摇杆功能
                            caps = dev.capabilities()
                            has_buttons = ecodes.EV_KEY in caps
                            has_axes = ecodes.EV_ABS in caps
                            if has_buttons and has_axes:
                                self._device = dev
                                self._device_path = dev_path
                                self._device_fd = dev.fd
                                print(f"✅ 自动检测到设备: {dev.name}")
                                print(f"   路径: {dev_path}")
                                break
                    except:
                        pass
            except Exception as e:
                pass

        self._joystick_calibration_offset = np.zeros(2)  # 2D摇杆校准偏移
        self._deadzone = 0.1

        self._sync = False
        self._done = False

        self._thread: threading.Thread = None
        self._running = True
        
        # 初始化所有按钮状态为 0（未按下）
        # 确保 L 和 R 按钮的初始状态是 0
        self._button_states[311] = 0  # L 按钮
        self._button_states[313] = 0  # R 按钮

    def _calibrate(self):
        """校准摇杆中心位置"""
        if self._device is None:
            return

        num_samples = 100
        joystick_samples = []

        for _ in range(num_samples):
            # 读取当前轴值
            try:
                rx_info = self._device.absinfo(3)  # ABS_RX
                ry_info = self._device.absinfo(4)  # ABS_RY
                
                if rx_info and ry_info:
                    # 归一化到[-1, 1]
                    rx_norm = rx_info.value / 32767.0
                    ry_norm = ry_info.value / 32767.0
                    joystick_samples.append([rx_norm, ry_norm])
            except:
                pass

            time.sleep(0.01)

        if joystick_samples:
            self._joystick_calibration_offset[:] = np.mean(joystick_samples, axis=0)

    def start(self):
        """启动处理线程"""
        if self._device is None:
            print("⚠️  警告: 未检测到 Joy-Con 设备!")
            print("   请检查设备连接或指定正确的设备路径")
            return

        print(f"✅ 找到设备: {self._device.name}")
        print(f"   设备路径: {self._device_path}")
        time.sleep(1.0)
        self._calibrate()

        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        print("✅ Handler 线程已启动")

    def _update_loop(self):
        """更新循环 - 在独立线程中运行"""
        if self._device is None:
            print("⚠️  错误: 设备未初始化，无法启动更新循环")
            return
            
        rate_limiter = RateLimiter(frequency=1.0 / self._timestep)
        
        # 使用队列在线程间传递事件
        import queue
        event_queue = queue.Queue()
        first_event_received = False
        start_time = time.time()
        event_count = 0
        
        def read_events():
            """在单独线程中读取事件"""
            try:
                print(f"📡 事件读取线程已启动，监听设备: {self._device.name}")
                for event in self._device.read_loop():
                    if not self._running:
                        print("📡 事件读取线程停止")
                        break
                    event_queue.put(event)
            except OSError as e:
                # [修复] 在关闭期间忽略 Bad file descriptor 错误 (errno 9)
                # 当 read_loop 正在等待时调用 self._device.close() 会发生这种情况
                if e.errno == 9:
                    pass
                else:
                    print(f"❌ 事件读取错误 (OSError): {e}")
            except Exception as e:
                # 仅当我们仍在运行时才报告错误
                if self._running:
                    print(f"❌ 事件读取错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 启动事件读取线程
        read_thread = threading.Thread(target=read_events, daemon=True)
        read_thread.start()
        
        # 等待一下让线程启动
        time.sleep(0.1)
        
        print("✅ 更新循环已启动，等待事件...")
        
        while self._running:
            try:
                # 从队列中读取事件（非阻塞）
                try:
                    event = event_queue.get(timeout=self._timestep)
                    event_count += 1
                    
                    if not first_event_received:
                        first_event_received = True
                        print(f"✅ 收到第一个事件! (类型: {event.type}, 代码: {event.code}, 值: {event.value})")
                    
                    self._process_event(event)
                    
                except queue.Empty:
                    pass  # 没有新事件，继续
                
                # 更新动作（每次循环都更新）
                self._joycon_update()
                
            except Exception as e:
                print(f"❌ 更新循环错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
            
            rate_limiter.sleep()
        
        print(f"📊 更新循环结束，共处理 {event_count} 个事件")

    def _process_event(self, event):
        """处理输入事件"""
        if event.type == ecodes.EV_KEY:
            # 按键事件
            button_code = event.code
            button_value = event.value
            
            self._button_states[button_code] = button_value
            
            # 调试：打印所有按钮事件
            button_name = self._button_map.get(button_code, f"Btn{button_code}")
            if button_value == 1:
                print(f"🔘 按钮按下: {button_name} (代码: {button_code})")
            elif button_value == 0:
                print(f"🔘 按钮释放: {button_name} (代码: {button_code})")
            
            # 处理按钮按下事件（立即处理sync状态）
            if button_value == 1:  # 按下
                # 开始记录按钮：尝试多个可能的按钮代码
                # 根据实际设备，可能是 304 (BTN_SOUTH/A) 或 305 (BTN_EAST/B)
                if button_code == 304 or button_code == 305:  # BTN_SOUTH 或 BTN_EAST
                    if not self._sync:
                        self._sync = True
                        print("\n" + "="*60)
                        print(f"🎮 开始记录 - 机器人现在可以移动了! (按钮: {button_name}, 代码: {button_code})")
                        print("="*60)
                        self._print_status(f"Button {button_name} pressed - Started recording")
                    else:
                        print(f"⚠️  已经在记录中")
                        self._print_status(f"Button {button_name} pressed - Already recording")
                
                # Y按钮：暂停记录
                elif button_code == 308:  # BTN_WEST (Y)
                    if self._sync:
                        self._sync = False
                        self._print_status(f"Button {button_name} pressed - Paused recording")
                    else:
                        self._print_status(f"Button {button_name} pressed - Already paused")
                
                # Home按钮：停止
                elif button_code == 316:  # BTN_MODE (Home)
                    self._done = True
                    self._print_status(f"Button {button_name} pressed - Stopping")
                
                # 其他按钮按下
                else:
                    self._print_status(f"Button {button_name} pressed")
            
        elif event.type == ecodes.EV_ABS:
            # 摇杆/轴事件
            if event.code in [3, 4]:  # ABS_RX, ABS_RY
                self._axis_values[event.code] = event.value

    def _apply_deadzone(self, value):
        """应用死区"""
        return value if abs(value) >= self._deadzone else 0.0

    def _joycon_update(self):
        """更新动作（Joy-Con版本）"""
        if self._device is None:
            return

        # 读取当前轴值（从设备或缓存）
        try:
            rx_info = self._device.absinfo(3)  # ABS_RX
            ry_info = self._device.absinfo(4)  # ABS_RY
            
            if rx_info:
                self._axis_values[3] = rx_info.value
            if ry_info:
                self._axis_values[4] = ry_info.value
        except:
            pass

        # 注意：即使 sync=False，也更新 action，这样机器人可以响应控制
        # sync 只控制是否记录数据，不影响机器人移动

        # 读取摇杆值（归一化到[-1, 1]）
        rx_raw = self._axis_values.get(3, 0)
        ry_raw = self._axis_values.get(4, 0)
        
        # 如果轴值为0，尝试从设备直接读取
        if rx_raw == 0 and ry_raw == 0:
            try:
                rx_info = self._device.absinfo(3)
                ry_info = self._device.absinfo(4)
                if rx_info:
                    rx_raw = rx_info.value
                    self._axis_values[3] = rx_raw
                if ry_info:
                    ry_raw = ry_info.value
                    self._axis_values[4] = ry_raw
            except:
                pass
        
        rx = rx_raw / 32767.0
        ry = ry_raw / 32767.0

        # 应用校准偏移
        rx_cal = rx - self._joystick_calibration_offset[0]
        ry_cal = ry - self._joystick_calibration_offset[1]

        # 应用死区
        rx_cal = self._apply_deadzone(rx_cal)
        ry_cal = self._apply_deadzone(ry_cal)
        
        # 调试：定期打印摇杆值（每100次更新打印一次）
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 100 == 0 and (abs(rx_cal) > 0.01 or abs(ry_cal) > 0.01):
            print(f"🎮 摇杆值: RX={rx_cal:+.3f}, RY={ry_cal:+.3f} | "
                  f"原始值: RX={rx_raw}, RY={ry_raw} | "
                  f"Action: X={self._action[0]:.6f}, Y={self._action[1]:.6f}")

        # Z轴控制: 使用 L 和 R 按钮（因为右 Joy-Con 不支持 ZL/ZR）
        # R 按钮向上, L 按钮向下
        r_button = self._button_states.get(311, 0)  # BTN_TR (R)
        l_button = self._button_states.get(313, 0)  # BTN_TL (L)
        
        # 调试：检查按钮状态
        if not hasattr(self, '_last_z_debug'):
            self._last_z_debug = (0, 0)
        if (r_button, l_button) != self._last_z_debug:
            print(f"🔘 Z轴按钮状态: R={r_button}, L={l_button}")
            self._last_z_debug = (r_button, l_button)

        # 夹爪控制: X按钮关闭, B按钮打开
        # 注意：305是物理A按钮，但映射为B功能
        close_gripper = self._button_states.get(307, 0)  # BTN_NORTH (X)
        open_gripper = self._button_states.get(304, 0)  # BTN_SOUTH (物理B按钮)

        # 更新动作数组（增量更新）
        # 注意：即使摇杆没有移动（rx_cal=0, ry_cal=0），也会执行更新
        # 这样可以确保即使没有输入，action也会保持当前值
        
        # X/Y轴：始终更新（即使输入很小，也要保持响应）
        # 移除条件检查，让 action 始终更新，这样机器人可以响应控制
        # 增加系数以提高响应速度：从 0.000002 增加到 0.0001（50倍）
        # 这样摇杆完全推到底时，每秒可以移动约 0.01 单位（100Hz * 0.0001）
        self._action[0] -= rx_cal * 0.002  # X轴（左右）- 增加系数
        self._action[1] += ry_cal * 0.002  # Y轴（前后）- 反转，增加系数
        
        # Z轴：使用 L 和 R 按钮控制（右 Joy-Con 不支持 ZL/ZR）
        # 只有当按钮实际按下时才更新（值为 1 表示按下，0 表示释放）
        if r_button == 1:
            self._action[2] += 0.002  # R 按钮：向上
        if l_button == 1:  # 改为独立的 if，而不是 elif
            self._action[2] -= 0.002  # L 按钮：向下
        
        # 调试：如果 Z 轴一直在变化，打印信息
        if not hasattr(self, '_last_z_action'):
            self._last_z_action = self._action[2]
        if abs(self._action[2] - self._last_z_action) > 0.001:
            print(f"📊 Z轴变化: {self._last_z_action:.6f} -> {self._action[2]:.6f} (R={r_button}, L={l_button})")
            self._last_z_action = self._action[2]
        
        # 夹爪：按钮按下时更新
        if open_gripper == 1:
            self._action[3] += 0.01
        elif close_gripper == 1:
            self._action[3] -= 0.01
        
        self._action[3] = np.clip(self._action[3], 0.0, 1.0)
        
        # 检查action是否变化（减少打印频率，避免刷屏）
        action_changed = not np.allclose(self._action, self._last_action, atol=self._action_change_threshold)
        if action_changed:
            # 只在有显著变化时打印（减少输出）
            if np.max(np.abs(self._action - self._last_action)) > 0.0001:
                self._print_status("Action changed")
            self._last_action = self._action.copy()

    def close(self):
        """关闭设备"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._device:
            try:
                self._device.close()
            except:
                pass

    def print_info(self):
        """打印控制映射信息"""
        print("=" * 60)
        print("Joy-Con Controller Mapping:")
        print("=" * 60)
        print("⚠️  重要: 必须先按 A 按钮开始记录，机器人才会移动!")
        print()
        print("控制按钮:")
        print("  🟢 Start (开始记录):    按钮 304 或 305 (A/B)")
        print("  🟡 Pause (暂停记录):    Y (308)")
        print("  🔴 Stop (停止):         Home (316)")
        print()
        print("机器人控制:")
        print("  Movement (移动):        Right Stick (RX/RY)")
        print("  +Z (向上):              R (311) - 右 Joy-Con 不支持 ZR")
        print("  -Z (向下):              L (313) - 右 Joy-Con 不支持 ZL")
        print("  Open Gripper (打开):    A (304)")
        print("  Close Gripper (关闭):   X (307)")
        print()
        print("💡 提示: 如果按钮不响应，请查看上面的调试信息")
        print("   找到实际按下的按钮代码，然后告诉我以便调整映射")
        print("=" * 60)
        print(f"设备: {self._device.name if self._device else '未连接'}")
        print(f"设备路径: {self._device_path}")
        if self._device is None:
            print("⚠️  警告: 未检测到设备，请检查连接!")
        print("=" * 60)
    
    def _print_status(self, trigger=None):
        """打印当前状态（action值和按钮状态）"""
        print("------------------------------")
        if trigger:
            print(f"Trigger: {trigger}")
        print(f"Action: X={self._action[0]:.6f}, Y={self._action[1]:.6f}, "
              f"Z={self._action[2]:.6f}, Gripper={self._action[3]:.3f}")
        print(f"Sync: {self._sync}, Done: {self._done}")
        
        # 显示按下的按钮
        pressed_buttons = []
        for code, value in self._button_states.items():
            if value == 1:
                button_name = self._button_map.get(code, f"Btn{code}")
                pressed_buttons.append(f"{button_name}({code})")
        if pressed_buttons:
            print(f"Pressed buttons: {', '.join(pressed_buttons)}")
        
        # 显示摇杆值
        rx_raw = self._axis_values.get(3, 0)
        ry_raw = self._axis_values.get(4, 0)
        if rx_raw != 0 or ry_raw != 0:
            rx_norm = rx_raw / 32767.0
            ry_norm = ry_raw / 32767.0
            print(f"Stick: RX={rx_norm:+.3f}, RY={ry_norm:+.3f}")
        print("------------------------------")


def main():
    """测试主函数"""
    import sys
    
    device_path = None  # None表示自动检测
    if len(sys.argv) > 1:
        device_path = sys.argv[1]
    
    handler = PickBoxJoyconEvdevHandler(device_path=device_path)
    handler.print_info()
    handler.start()

    try:
        while not handler._done:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        handler.close()


if __name__ == "__main__":
    main()

