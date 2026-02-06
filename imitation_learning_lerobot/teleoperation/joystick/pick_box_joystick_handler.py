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

import abc


class Handler(abc.ABC):
    _name = ""

    def __init__(self):
        super().__init__()

        self._action: np.ndarray = None

        self._done = False
        self._sync = False

    @classmethod
    def name(cls) -> str:
        return cls._name

    def start(self):
        pass

    def close(self):
        pass

    @property
    def action(self):
        if self._action is None:
            return None
        return self._action.copy()

    @property
    def done(self):
        return self._done

    @property
    def sync(self):
        return self._sync

    def print_info(self):
        print("------------------------------")


class PickBoxJoyconHandler(Handler):
    _name = "pick_box_joycon"

    def __init__(self, device_path=None):
        super().__init__()

        self._timestep = 0.01
        self._action = np.zeros(4)
        self._device_path = device_path or "/dev/input/event16"  # 默认路径

        # Joy-Con按钮映射（根据evtest输出）
        # 注意：Joy-Con (R)的物理按钮映射
        # 304 (BTN_SOUTH) = 物理B按钮（下面）
        # 305 (BTN_EAST) = 物理A按钮（右边）
        self._button_map = {
            304: "B",      # BTN_SOUTH (物理B按钮)
            305: "A",      # BTN_EAST (物理A按钮)
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
                    else:
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
            return

        time.sleep(1.0)
        self._calibrate()

        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        """更新循环 - 在独立线程中运行"""
        if self._device is None:
            return
            
        rate_limiter = RateLimiter(frequency=1.0 / self._timestep)
        
        # 使用队列在线程间传递事件
        import queue
        event_queue = queue.Queue()
        first_event_received = False
        start_time = time.time()
        
        def read_events():
            """在单独线程中读取事件"""
            try:
                for event in self._device.read_loop():
                    if not self._running:
                        break
                    event_queue.put(event)
            except Exception:
                pass
        
        # 启动事件读取线程
        read_thread = threading.Thread(target=read_events, daemon=True)
        read_thread.start()
        
        while self._running:
            try:
                # 从队列中读取事件（非阻塞）
                try:
                    event = event_queue.get(timeout=self._timestep)
                    
                    if not first_event_received:
                        first_event_received = True
                    
                    self._process_event(event)
                    
                except queue.Empty:
                    pass  # 没有新事件，继续
                
                # 更新动作（每次循环都更新）
                self._joycon_update()
                
            except Exception:
                time.sleep(0.1)
            
            rate_limiter.sleep()

    def _process_event(self, event):
        """处理输入事件"""
        if event.type == ecodes.EV_KEY:
            # 按键事件
            button_code = event.code
            button_value = event.value
            
            self._button_states[button_code] = button_value
            
            # 处理按钮按下事件（立即处理sync状态）
            if button_value == 1:  # 按下
                button_name = self._button_map.get(button_code, f"Btn{button_code}")
                
                # A按钮（物理A按钮，代码305）：开始记录
                if button_code == 305:  # BTN_EAST (物理A按钮)
                    if not self._sync:
                        self._sync = True
                        self._print_status(f"Button {button_name} pressed - Started recording")
                    else:
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

        # 注意：sync状态现在在_process_event中处理，这里只检查
        # 如果未同步，不更新动作（但保持当前值）
        if not self._sync:
            return

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

        # Z轴控制: ZR向上, ZL向下
        zr_button = self._button_states.get(313, 0)  # BTN_TR2 (ZR)
        zl_button = self._button_states.get(312, 0)  # BTN_TL2 (ZL)

        # 夹爪控制: X按钮关闭, B按钮打开
        # 注意：305是物理A按钮，但映射为B功能
        close_gripper = self._button_states.get(307, 0)  # BTN_NORTH (X)
        open_gripper = self._button_states.get(304, 0)  # BTN_SOUTH (物理B按钮)

        # 更新动作数组（增量更新）
        # 注意：即使摇杆没有移动（rx_cal=0, ry_cal=0），也会执行更新
        # 这样可以确保即使没有输入，action也会保持当前值
        
        # X/Y轴：只有摇杆有输入时才更新
        if abs(rx_cal) > 0.001 or abs(ry_cal) > 0.001:
            self._action[0] -= rx_cal * 0.000002  # X轴（左右）
            self._action[1] += ry_cal * 0.000002  # Y轴（前后）- 反转
        
        # Z轴：按钮按下时更新
        if zr_button == 1:
            self._action[2] += 0.002
        elif zl_button == 1:
            self._action[2] -= 0.002
        
        # 夹爪：按钮按下时更新
        if open_gripper == 1:
            self._action[3] += 0.01
        elif close_gripper == 1:
            self._action[3] -= 0.01
        
        self._action[3] = np.clip(self._action[3], 0.0, 1.0)
        
        # 检查action是否变化
        action_changed = not np.allclose(self._action, self._last_action, atol=self._action_change_threshold)
        if action_changed:
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
        print("------------------------------")
        print("Joy-Con Controller Mapping:")
        print("Start:           A (305, BTN_EAST, 物理A按钮)")
        print("Pause:           Y (308)")
        print("Stop:            Home (316)")
        print("Movement:        Right Stick (RX/RY)")
        print("+Z (Up):         ZR (313)")
        print("-Z (Down):       ZL (312)")
        print("Open Gripper:    B (304, BTN_SOUTH, 物理B按钮)")
        print("Close Gripper:   X (307)")
        print("------------------------------")
        print(f"Device: {self._device.name if self._device else 'Not connected'}")
        print(f"Device path: {self._device_path}")
        print("------------------------------")
    
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
    
    handler = PickBoxJoyconHandler(device_path=device_path)
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

