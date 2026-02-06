#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 joycon_evdev Handler
"""

import time
import sys
from imitation_learning_lerobot.teleoperation import HandlerFactory


def test_joycon_evdev_handler(device_path=None):
    """测试 joycon_evdev Handler"""
    
    print("=" * 60)
    print("测试 joycon_evdev Handler")
    print("=" * 60)
    
    # 确保 HandlerFactory 已注册所有 Handler
    HandlerFactory.register_all()
    
    # 获取 Handler 类
    handler_cls = HandlerFactory.get_strategies("joycon_evdev")
    
    if handler_cls is None:
        print("错误: 找不到 'joycon_evdev' Handler!")
        print("已注册的 Handler 类型:")
        # 尝试列出所有已注册的 Handler
        print("  请检查 HandlerFactory._strategies")
        return
    
    print(f"成功获取 Handler 类: {handler_cls.__name__}")
    print(f"Handler 名称: {handler_cls.name}")
    print()
    
    # 创建 Handler 实例
    if device_path:
        print(f"使用指定设备路径: {device_path}")
        handler = handler_cls(device_path=device_path)
    else:
        print("自动检测设备...")
        handler = handler_cls()
    
    # 打印设备信息
    handler.print_info()
    
    # 启动 Handler
    print("\n启动 Handler...")
    handler.start()
    
    print("\n控制说明:")
    print("  - 按 A 按钮 (305) 开始记录 (sync=True)")
    print("  - 按 Y 按钮 (308) 暂停记录 (sync=False)")
    print("  - 按 Home 按钮 (316) 停止并退出")
    print("  - 使用右摇杆控制 X/Y 轴移动")
    print("  - ZR 按钮向上移动 Z 轴")
    print("  - ZL 按钮向下移动 Z 轴")
    print("  - B 按钮打开夹爪")
    print("  - X 按钮关闭夹爪")
    print("\n开始监控... (按 Ctrl+C 或 Home 按钮退出)\n")
    
    try:
        last_action = None
        last_sync = None
        
        while not handler.done:
            # 获取当前状态
            action = handler.action
            sync = handler.sync
            done = handler.done
            
            # 只在状态变化时打印
            if action is not None:
                action_changed = last_action is None or not all(
                    abs(a - b) < 1e-6 for a, b in zip(action, last_action)
                )
                
                if action_changed or sync != last_sync:
                    print(f"Action: X={action[0]:.6f}, Y={action[1]:.6f}, "
                          f"Z={action[2]:.6f}, Gripper={action[3]:.3f} | "
                          f"Sync: {sync} | Done: {done}")
                    last_action = action.copy()
                    last_sync = sync
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在退出...")
    finally:
        print("\n关闭 Handler...")
        handler.close()
        print("测试完成!")


def main():
    """主函数"""
    device_path = None
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        device_path = sys.argv[1]
        print(f"使用命令行指定的设备路径: {device_path}")
    
    test_joycon_evdev_handler(device_path)


if __name__ == "__main__":
    main()

