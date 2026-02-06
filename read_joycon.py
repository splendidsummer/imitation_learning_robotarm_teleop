#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取Joy-Con手柄输入的Python脚本
需要安装: pip install evdev
"""

import sys
from evdev import InputDevice, categorize, ecodes
import time

# Joy-Con (R) 事件代码映射
BUTTON_MAP = {
    304: "BTN_SOUTH (A)",      # A按钮
    305: "BTN_EAST (B)",       # B按钮
    307: "BTN_NORTH (X)",      # X按钮
    308: "BTN_WEST (Y)",       # Y按钮
    310: "BTN_TL (L)",         # L按钮
    311: "BTN_TR (R)",         # R按钮
    312: "BTN_TL2 (ZL)",       # ZL按钮
    313: "BTN_TR2 (ZR)",       # ZR按钮
    315: "BTN_START (+)",      # +按钮
    316: "BTN_MODE (Home)",    # Home按钮
    318: "BTN_THUMBR (SR)",    # SR按钮
}

AXIS_MAP = {
    3: "ABS_RX (右摇杆X轴)",
    4: "ABS_RY (右摇杆Y轴)",
}

def read_joycon(device_path="/dev/input/event16"):
    """
    读取Joy-Con手柄输入
    
    Args:
        device_path: 输入设备路径，默认为event16 (Joy-Con R)
    """
    try:
        # 打开输入设备
        device = InputDevice(device_path)
        print(f"设备名称: {device.name}")
        print(f"设备路径: {device_path}")
        print(f"设备信息: {device.info}")
        print("\n支持的按键和轴:")
        print(f"  按键: {device.capabilities().get(ecodes.EV_KEY, [])}")
        print(f"  轴: {device.capabilities().get(ecodes.EV_ABS, [])}")
        print("\n开始读取输入... (按Ctrl+C退出)\n")
        print("-" * 60)
        
        # 存储当前状态
        button_states = {}
        axis_values = {}
        
        # 读取事件循环
        for event in device.read_loop():
            if event.type == ecodes.EV_KEY:
                # 按键事件
                button_name = BUTTON_MAP.get(event.code, f"未知按键({event.code})")
                button_states[event.code] = event.value
                
                if event.value == 1:
                    print(f"[按下] {button_name}")
                elif event.value == 0:
                    print(f"[释放] {button_name}")
                    
            elif event.type == ecodes.EV_ABS:
                # 摇杆/轴事件
                axis_name = AXIS_MAP.get(event.code, f"未知轴({event.code})")
                axis_values[event.code] = event.value
                
                # 只显示有意义的摇杆移动（避免噪声）
                if event.code in [3, 4]:  # ABS_RX, ABS_RY
                    # 归一化到[-1, 1]范围
                    normalized = event.value / 32767.0
                    if abs(normalized) > 0.01:  # 只显示大于1%的移动
                        print(f"[摇杆] {axis_name}: {event.value:6d} ({normalized:+.3f})")
                        
            elif event.type == ecodes.EV_SYN:
                # 同步事件，可以在这里更新显示
                pass
                
    except PermissionError:
        print(f"错误: 没有权限访问 {device_path}")
        print("请使用sudo运行此脚本，或者将用户添加到input组:")
        print("  sudo usermod -a -G input $USER")
        print("  然后重新登录")
        sys.exit(1)
    except FileNotFoundError:
        print(f"错误: 找不到设备 {device_path}")
        print("请检查设备是否存在，或使用evtest查看可用设备")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def list_devices():
    """列出所有可用的输入设备"""
    from evdev import list_devices
    
    print("可用的输入设备:")
    print("-" * 60)
    devices = list_devices()
    
    for i, device_path in enumerate(devices):
        try:
            device = InputDevice(device_path)
            print(f"{i}: {device_path}")
            print(f"   名称: {device.name}")
            print(f"   信息: {device.info}")
            print()
        except:
            pass


def read_joycon_with_state(device_path="/dev/input/event16", update_rate=0.1):
    """
    读取Joy-Con输入并持续显示当前状态
    
    Args:
        device_path: 输入设备路径
        update_rate: 状态更新频率（秒）
    """
    import select
    
    try:
        device = InputDevice(device_path)
        print(f"设备名称: {device.name}")
        print(f"开始读取输入... (按Ctrl+C退出)\n")
        
        # 当前状态
        buttons = {}
        axes = {}
        
        while True:
            # 使用select非阻塞读取
            r, w, x = select.select([device], [], [], update_rate)
            
            if device in r:
                try:
                    for event in device.read():
                        if event.type == ecodes.EV_KEY:
                            buttons[event.code] = event.value
                        elif event.type == ecodes.EV_ABS:
                            axes[event.code] = event.value
                except BlockingIOError:
                    pass
            
            # 显示当前状态
            print("\r" + " " * 80, end="")  # 清空行
            print("\r", end="")
            
            # 显示按键状态
            pressed_buttons = [BUTTON_MAP.get(code, f"B{code}") 
                             for code, value in buttons.items() 
                             if value == 1]
            if pressed_buttons:
                print(f"按键: {', '.join(pressed_buttons)} | ", end="")
            
            # 显示摇杆状态
            if 3 in axes:  # ABS_RX
                rx_norm = axes[3] / 32767.0
                print(f"RX: {rx_norm:+.3f} | ", end="")
            if 4 in axes:  # ABS_RY
                ry_norm = axes[4] / 32767.0
                print(f"RY: {ry_norm:+.3f}", end="")
            
            sys.stdout.flush()
            
    except PermissionError:
        print(f"错误: 没有权限访问 {device_path}")
        print("请使用sudo运行此脚本")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


def read_joycon_simple(device_path="/dev/input/event16"):
    """
    简化版本：只显示重要事件
    """
    try:
        device = InputDevice(device_path)
        print(f"读取设备: {device.name}")
        print("按任意按钮或移动摇杆... (Ctrl+C退出)\n")
        
        for event in device.read_loop():
            if event.type == ecodes.EV_KEY and event.value == 1:
                # 只显示按下事件
                button_name = BUTTON_MAP.get(event.code, f"按键{event.code}")
                print(f"按下: {button_name}")
                
            elif event.type == ecodes.EV_ABS:
                if event.code in [3, 4]:  # 摇杆
                    normalized = event.value / 32767.0
                    if abs(normalized) > 0.1:  # 只显示大于10%的移动
                        axis_name = AXIS_MAP.get(event.code, f"轴{event.code}")
                        print(f"{axis_name}: {normalized:+.2f}")
                        
    except PermissionError:
        print("错误: 需要root权限或加入input组")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n退出")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="读取Joy-Con手柄输入")
    parser.add_argument(
        "-d", "--device",
        default="/dev/input/event16",
        help="输入设备路径 (默认: /dev/input/event16)"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="列出所有可用设备"
    )
    parser.add_argument(
        "-s", "--simple",
        action="store_true",
        help="简化模式：只显示重要事件"
    )
    parser.add_argument(
        "-c", "--continuous",
        action="store_true",
        help="连续显示模式：持续显示当前状态"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_devices()
    elif args.simple:
        read_joycon_simple(args.device)
    elif args.continuous:
        read_joycon_with_state(args.device)
    else:
        read_joycon(args.device)

